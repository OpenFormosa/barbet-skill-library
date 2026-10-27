"""Prepublication: immutable compiled arrays, semantic holds, exact requests/budgets.

Copies only newly accepted v2 rows. This script performs no remote mutation.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.sft import RESERVED, MiniCPMRenderer, pangolin_render
from barbet_inference.memory import canonical
from scripts.build_memory_sft_v2 import OUT, STATE, WORK, RAW, file_sha, save

PUBLIC=WORK/'exports/memory-harness-v2-public'
WAVE='wave-0001'


def init():
    global QWEN, CHAT, HOLD_IDS, HOLD_QUERIES, SEARCH_ALLOWED, MINI_RENDER
    os.environ['TOKENIZERS_PARALLELISM']='false'
    root=WORK/'inference/qwen38/model-fp8'
    QWEN=Tokenizer.from_file(str(root/'tokenizer.json'))
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    env.filters['tojson']=lambda x,**kw:json.dumps(x,**{'ensure_ascii':False,**kw})
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail
    CHAT=env.from_string((root/'chat_template.jinja').read_text())
    summary=json.loads((STATE/'semantic-audit-v1/summary.json').read_text())
    HOLD_IDS=set(summary['excluded_record_ids'])
    HOLD_QUERIES=set()
    for path in (STATE/'normalized').glob('pilot-*.jsonl'):
        if path.name.endswith('.excluded.jsonl'):continue
        for line in path.open():
            row=json.loads(line)
            if row['record_id'] in HOLD_IDS:HOLD_QUERIES.add(row['query_hash'])
    # The pilot found ambiguous multi-hop clues and unsupported conclusions in
    # 3 of 8 search examples. Keep only its first-pass accepted search examples
    # in this wave; the unreviewed search cohort stays local for future work.
    SEARCH_ALLOWED={r['record_id'] for r in summary['results'] if r['domain']=='search' and r['verdict']=='ACCEPT'}
    MINI_RENDER=MiniCPMRenderer((RAW/'minicpm5-tokenizer/chat_template.jinja').read_text())


def request_budget(row):
    chat=prepare_dataset_request(row,model='Qwen/Qwen3.8-Flash-Next-FP8')
    completion=prepare_dataset_request(row,model='Qwen/Qwen3.8-Flash-Next-FP8',mode='pangolin_completion')
    rendered=CHAT.render(messages=chat['messages'],tools=[],add_generation_prompt=True,
                         enable_thinking=False,bos_token='',eos_token='<|im_end|>')
    ct=len(QWEN.encode(rendered,add_special_tokens=False).ids)
    pt=len(QWEN.encode(completion['prompt'],add_special_tokens=False).ids)
    if max(ct,pt)+4096>1000000:
        raise ValueError('served Qwen budget exceeded')
    mini=prepare_dataset_request(row,model='openbmb/MiniCPM5-2B',mode='minicpm_completion')
    if hashlib.sha256(mini['prompt'].encode()).hexdigest()!=row['mini_prompt_sha256']:
        raise ValueError('native request prefix changed')
    return {'record_id':row['record_id'],'chat_tokens':ct,'completion_tokens':pt,
        'chat_rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
        'chat_payload_sha256':hashlib.sha256(canonical(chat['messages']).encode()).hexdigest(),
        'completion_sha256':hashlib.sha256(completion['prompt'].encode()).hexdigest(),
        'output_reserve_tokens':4096,'served_max_model_len':1000000}


def job(path):
    ident=Path(path).stem
    report=STATE/'public-package'/f'{ident}.json'
    if report.exists():return json.loads(report.read_text())
    source=json.loads(Path(path).read_text())
    if not source['all_exported_arrays_readback']:
        raise ValueError('missing original full-array readback')
    files=[];counts=Counter();rows_index=[];budgets=[]
    by_split={}
    for f in source['files']:
        if file_sha(OUT/f['path'])!=f['sha256']:
            raise ValueError('compiled bytes changed')
        by_split.setdefault(f['split'],{})[f['config']]=f
    for split,configs in by_split.items():
        if set(configs)!={'records','tokenized','minicpm5_tokenized'}:
            raise ValueError('missing aligned config')
        tables={k:pq.read_table(OUT/v['path']) for k,v in configs.items()}
        ids=tables['records'].column('record_id').to_pylist()
        if any(t.column('record_id').to_pylist()!=ids for t in tables.values()):
            raise ValueError('config record alignment')
        keep=[]
        for n,row in enumerate(tables['records'].to_pylist()):
            counts['seen']+=1
            if row['record_id'] in HOLD_IDS or row['query_hash'] in HOLD_QUERIES:
                counts['excluded:first_semantic_hold']+=1;continue
            if row['domain']=='search' and row['record_id'] not in SEARCH_ALLOWED:
                counts['excluded:unreviewed_search_after_pilot_failures']+=1;continue
            messages=json.loads(row['messages_json']);tools=json.loads(row['tools_json'])
            # A quoted JSON argument containing a native delimiter would still be
            # recognized by the tokenizer. Preserve the original row locally and
            # hold it rather than changing a literal argument during publication.
            literal_data=canonical(tools)+'\n'+canonical([m.get('tool_calls',[]) for m in messages])
            if RESERVED.search(literal_data):
                counts['excluded:reserved_delimiter_in_literal_tool_data']+=1;continue
            ptext,pspans=pangolin_render(messages,tools)
            mtext,mspans=MINI_RENDER.render(messages,tools)
            if ptext!=row['text'] or mtext!=row['mini_text'] or canonical(pspans)!=row['target_spans_json'] or canonical(mspans)!=row['mini_target_spans_json']:
                raise ValueError('renderer changed after compilation')
            budgets.append(request_budget(row))
            keep.append(n);counts['accepted']+=1;counts['split:'+split]+=1
            for key in ['pangolin_tokens','pangolin_supervised_tokens','minicpm_tokens','minicpm_supervised_tokens','assistant_turns','tool_observations']:
                counts[key]+=row[key]
            for key in ['domain','source','upstream_source','license','verification_tier','memory_placement','observation_origin']:
                counts[key+':'+str(row[key])]+=1
            for key in ['actual_retrieval','recorded_error_followup']:
                counts[key]+=int(row[key])
            length=max(row['pangolin_tokens'],row['minicpm_tokens'])
            bucket=next(b for b in [4096,8192,16384,32768,65536,131072] if length<=b)
            counts['length_bucket:'+str(bucket)]+=1
            rows_index.append({k:row[k] for k in ['record_id','family_id','query_hash','split','domain','source','upstream_source']})
        for name,table in tables.items():
            if not keep:continue
            selected=table.take(pa.array(keep,type=pa.int64()))
            dest=PUBLIC/'data'/WAVE/name/f'{split}-{ident}.parquet'
            dest.parent.mkdir(parents=True,exist_ok=True)
            if len(keep)==table.num_rows:
                # Immutable compiled bytes were fully read back, hash checked,
                # and aligned above. A hard link avoids an extra local copy.
                if not dest.exists():os.link(OUT/configs[name]['path'],dest)
            else:
                if dest.exists():
                    if not pq.read_table(dest).equals(selected):
                        raise ValueError('immutable public shard conflicts with resumed selection')
                else:
                    temporary=dest.with_suffix('.parquet.partial')
                    pq.write_table(selected,temporary,compression='zstd',compression_level=6,row_group_size=128)
                    os.replace(temporary,dest)
            if not pq.read_table(dest).equals(selected):
                raise ValueError('public export whole-table readback failed')
            files.append({'path':str(dest.relative_to(PUBLIC)),'sha256':file_sha(dest),
                'bytes':dest.stat().st_size,'rows':len(keep),'split':split,'config':name})
    proof=PUBLIC/'proofs'/WAVE/f'budgets-{ident}.parquet'
    if budgets:
        proof.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(budgets),proof,compression='zstd')
        files.append({'path':str(proof.relative_to(PUBLIC)),'sha256':file_sha(proof),
                      'bytes':proof.stat().st_size,'rows':len(budgets),'config':'budget-proof'})
    index=STATE/'public-package'/f'{ident}.index.parquet';index.parent.mkdir(parents=True,exist_ok=True)
    if rows_index:pq.write_table(pa.Table.from_pylist(rows_index),index,compression='zstd')
    result={'shard':ident,'counts':dict(counts),'files':files,'all_three_configs_exact_readback':True,
        'both_qwen_request_modes_measured':True,'no_truncation':True,'index':str(index) if rows_index else None}
    save(report,result);return result


def main():
    paths=sorted((STATE/'compiled').glob('[0-9]*.json'))
    if not (STATE/'compile-summary.json').exists():
        raise ValueError('full compilation not finished')
    if len(paths)!=json.loads((STATE/'dedup-summary.json').read_text())['shards']:
        raise ValueError('missing compiled shards')
    save(STATE/'publication-quality-policy-v1.json',{
        'after_pilot':'34 ACCEPT, 5 REVISE, 1 REJECT in 40 stratified pilot reviews; not a representative accuracy estimate.',
        'rules':['All first non-accept records held','Search: only first-pass accepted pilot examples in this wave; all unreviewed search held after observed clue/grounding errors',
                 'Hold literal tool data containing reserved chat delimiters; never mutate literal arguments to pass',
                 'Other rows retain honest upstream-verification labels and all mechanical gates; not all-row semantic verification'],
        'new_model_training':False})
    totals=Counter();files=[]
    with ProcessPoolExecutor(max_workers=24,initializer=init) as pool:
        for n,result in enumerate(pool.map(job,paths),1):
            totals.update(result['counts']);files.extend(result['files'])
            if n%10==0 or n==len(paths):print(canonical({'shards':n,'total':len(paths),'accepted':totals['accepted']}),flush=True)
    # Exact global identity/split checks over small metadata columns.
    db=sqlite3.connect(STATE/'public-index.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY, family_id TEXT, query_hash TEXT, split TEXT, domain TEXT, source TEXT, upstream_source TEXT)')
    db.execute('DELETE FROM records')
    for p in sorted((STATE/'public-package').glob('*.index.parquet')):
        rows=pq.read_table(p).to_pylist()
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',[(r['record_id'],r['family_id'],r['query_hash'],r['split'],r['domain'],r['source'],r['upstream_source']) for r in rows]);db.commit()
    for key in ['family_id','query_hash']:
        mixed=db.execute(f'SELECT COUNT(*) FROM (SELECT {key} FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
        if mixed:raise ValueError('cross-split family or exact-query leak')
    unique={k:db.execute(f'SELECT COUNT(DISTINCT {k}) FROM records').fetchone()[0] for k in ['record_id','family_id','query_hash']}
    db.close()
    result={'wave':WAVE,'status':'LOCAL_VERIFIED_NOT_YET_PUBLISHED','counts':dict(totals),
        'unique_counts':unique,'files':files,'accepted_episodes_target':10000000,
        'full_goal_complete':False,'legacy_active_rows':0,'model_efficacy_measured':False,
        'tokenizers':{'Pangolin':file_sha(WORK/'data/blobs/tokenizer/tokenizer.json'),
            'MiniCPM5-2B':file_sha(RAW/'minicpm5-tokenizer/tokenizer.json'),
            'served_Qwen':file_sha(WORK/'inference/qwen38/model-fp8/tokenizer.json')},
        'semantic_audit_scope':'40 stratified pilot samples, first non-accept held; not 1.4M semantic approvals',
        'no_source_program_executed':True,'all_records_and_arrays_reread':True,
        'all_answer_free_qwen_chat_and_completion_budgets_measured':True,
        'all_training_and_native_minicpm_budgets_with_4096_reserve':True}
    save(STATE/'public-package-summary.json',result)
    print(canonical({'accepted':totals['accepted'],'unique':unique,'files':len(files)}),flush=True)

if __name__=='__main__':main()
