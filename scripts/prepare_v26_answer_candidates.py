"""Index original pre-deduplication answers without copying large source payloads.

Every pointer binds immutable normalized bytes. No old semantic hold is revived.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts.build_memory_sft_wave2 import STATE as SECOND
from barbet_inference.answer_selection import input_identity
from barbet_inference.memory import canonical
from barbet_inference.sft import RESERVED,Rejected
from barbet_inference.tool_contract import trajectory_has_undeclared_arguments

FIRST=base.STATE
STATE=base.WORK/'reports/memory-harness-v2-v26-selection'
POLICY=base.REPO/'configs/memory_sft_v2_v26_selection.json'
STYLES=base.REPO/'configs/openorca_canonical_teacher_styles_v1.json'
SCENARIO='\n\nSCENARIO RULES (native tool formatting takes precedence):\n'


@lru_cache(maxsize=1)
def generic_styles():
    return set(json.loads(STYLES.read_text())['remove_exact_generic_teacher_styles'])


def canonicalize(row):
    """Only known generic teacher style; real task constraints stay visible."""
    styles=generic_styles()
    row.setdefault('source_family_id',row['family_id'])
    if row['source']=='Open-Orca/OpenOrca':
        for message in row['messages']:
            if message['role']=='system' and SCENARIO in message['content']:
                prefix,scenario=message['content'].rsplit(SCENARIO,1)
                if scenario in styles:
                    message['content']=prefix
                    row['source_refs']['canonical_teacher_style_sha256']=base.sha(scenario)
                    row['source_refs']['transformations'].append('exact generic teacher style removed for common-input v26 ranking')
    # Stable provider call aliases define duplicate identity, not rewritten text.
    identity=json.loads(canonical(row['messages']));aliases={}
    for m in identity:
        for c in m.get('tool_calls',[]):
            aliases[c['id']]='call_'+str(len(aliases));c['id']=aliases[c['id']]
        if 'tool_call_id' in m:m['tool_call_id']=aliases[m['tool_call_id']]
    row['record_id']=base.sha(canonical({'version':'memory-harness-sft-v2','messages':row['messages'],'tools':row['tools']}))
    row['candidate_identity']=base.sha(canonical({'messages':identity,'tools':row['tools']}))
    row['ppl_input_hash']=input_identity(row['messages'],row['tools'])
    return row


def init():
    global HOLDS,HELD_QUERIES,SEARCH_OK,EXTRA
    policy=json.loads((STATE/'candidate-holds.private.json').read_text())
    HOLDS=set(policy['record_ids']);HELD_QUERIES=set(policy['query_hashes'])
    SEARCH_OK=set(policy['search_allowed']);EXTRA=set(policy['supplemental_prior_queries'])


def eligible(row):
    if row['record_id'] in HOLDS or row['query_hash'] in HELD_QUERIES:raise Rejected('first_semantic_hold')
    if row['query_hash'] in EXTRA:raise Rejected('supplemental_prior_job_hold')
    if row['domain']=='search' and row['record_id'] not in SEARCH_OK:raise Rejected('unreviewed_search')
    if trajectory_has_undeclared_arguments(row['messages'],row['tools']):raise Rejected('undeclared_function_parameter')
    if RESERVED.search(canonical(row['tools'])+'\n'+canonical([m.get('tool_calls',[]) for m in row['messages']])):
        raise Rejected('reserved_literal_tool_delimiter')


def index_file(job):
    dest=STATE/'candidate-index-parts'/f"{job['file_id']:05d}.parquet"
    receipt=dest.with_suffix('.json')
    if receipt.exists():return json.loads(receipt.read_text())
    counts=Counter();rows=[];digest=hashlib.sha256()
    with Path(job['path']).open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            digest.update(line);counts['seen']+=1;row=json.loads(line);original=row['record_id']
            try:eligible(row)
            except Rejected as exc:counts['excluded:'+str(exc)]+=1;continue
            row=canonicalize(row)
            rows.append({'candidate_identity':row['candidate_identity'],'record_id':row['record_id'],
                'original_record_id':original,'input_hash':row['ppl_input_hash'],
                'bucket':int(row['ppl_input_hash'][:8],16)%8,'file_id':job['file_id'],'offset':offset,'length':len(line),
                'domain':row['domain'],'query_hash':row['query_hash'],'source_family_id':row['source_family_id'],
                'template_family_id':row.get('template_family_id',row['source_family_id']),'source':row['source']})
            counts['eligible']+=1
    if digest.hexdigest()!=job['sha256']:raise ValueError('normalized source bytes changed')
    dest.parent.mkdir(exist_ok=True)
    if rows:pq.write_table(pa.Table.from_pylist(rows),dest,compression='zstd')
    result={'file_id':job['file_id'],'counts':dict(counts),'path':str(dest) if rows else None,
            'sha256':base.file_sha(dest) if rows else None}
    base.save(receipt,result);return result


def prepare():
    path=STATE/'candidate-input-files.json'
    if path.exists():raise ValueError('candidate preparation already frozen')
    STATE.mkdir(parents=True,exist_ok=True)
    holds=set();queries=set();search=set()
    for state in [FIRST,SECOND]:
        audit=json.loads((state/'semantic-audit-v1/summary.json').read_text());bad=set(audit['excluded_record_ids']);holds.update(bad)
        for p in (state/'normalized').glob('pilot-*.jsonl'):
            if p.name.endswith('.excluded.jsonl'):continue
            for line in p.open():
                r=json.loads(line)
                if r['record_id'] in bad:queries.add(r['query_hash'])
        search.update(r['record_id'] for r in audit['results'] if r['domain']=='search' and r['verdict']=='ACCEPT')
    holds.update(json.loads((FIRST/'postpublication-tool-interface-audit-v2.json').read_text())['record_ids'])
    extra=set()
    for p in (base.WORK/'data/expansion/skill-prefill-v1').glob('*/prospective.json'):
        value=json.loads(p.read_text())
        for r in value.get('items',[]) if isinstance(value,dict) else []:
            for key in ['source_question','source_query','source_instruction']:
                if isinstance(r.get(key),str):extra.add(base.sha(base.norm(r[key])))
    files=[]
    for state in [FIRST,SECOND]:
        for p in sorted((state/'normalized').glob('[0-9]*.report.json')):
            report=json.loads(p.read_text())
            files.append({'file_id':len(files),'path':report['path'],'sha256':report['sha256'],'wave':state.name})
    base.save(STATE/'candidate-holds.private.json',{'record_ids':sorted(holds),'query_hashes':sorted(queries),'search_allowed':sorted(search),'supplemental_prior_queries':sorted(extra)})
    base.save(STATE/'candidate-policy-binding.json',{'policy_sha256':base.file_sha(POLICY),'style_policy_sha256':base.file_sha(STYLES),
        'source_order_compaction_superseded':True,'all_previous_first_holds_preserved':True,
        'recover_original_pre_deduplication_answers':True,'no_model_training':True,
        'limits':'This index covers normalized eligible candidates from waves 1 and 2. Raw rows excluded before normalization are not silently declared scored.'})
    base.save(path,files);print(json.dumps({'input_files':len(files),'first_hold_ids':len(holds),'first_hold_queries':len(queries)}),flush=True)


def index(workers):
    jobs=json.loads((STATE/'candidate-input-files.json').read_text());counts=Counter();parts=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,result in enumerate(pool.map(index_file,jobs),1):
            counts.update(result['counts']);parts.append(result)
            if n%100==0 or n==len(jobs):print(json.dumps({'indexed_files':n,'total':len(jobs),'counts':dict(counts)}),flush=True)
    db=sqlite3.connect(STATE/'candidates.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS candidates(candidate_identity TEXT PRIMARY KEY,record_id TEXT,original_record_id TEXT,input_hash TEXT,bucket INTEGER,file_id INTEGER,offset INTEGER,length INTEGER,domain TEXT,query_hash TEXT,source_family_id TEXT,template_family_id TEXT,source TEXT)')
    db.execute('DELETE FROM candidates')
    for part in parts:
        if not part['path']:continue
        if base.file_sha(part['path'])!=part['sha256']:raise ValueError('candidate index bytes changed')
        rows=pq.read_table(part['path']).to_pylist()
        db.executemany('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',[tuple(r.values()) for r in rows]);db.commit()
    db.execute('CREATE INDEX IF NOT EXISTS input_groups ON candidates(input_hash)')
    db.execute('DROP TABLE IF EXISTS groups')
    db.execute('CREATE TABLE groups AS SELECT input_hash,MIN(bucket) AS bucket,COUNT(*) AS candidates FROM candidates GROUP BY input_hash')
    db.execute('CREATE UNIQUE INDEX group_identity ON groups(input_hash)');db.execute('CREATE INDEX group_buckets ON groups(bucket,candidates)');db.commit()
    result={'counts':dict(counts),'distinct_answers':db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
        'input_groups':db.execute('SELECT COUNT(*) FROM groups').fetchone()[0],
        'competing_groups':db.execute('SELECT COUNT(*) FROM groups WHERE candidates>1').fetchone()[0],
        'answers_to_score':db.execute('SELECT SUM(candidates) FROM groups WHERE candidates>1').fetchone()[0],
        'status':'INDEXED_NOT_YET_PPL_SELECTED','all_score_inputs_are_eligible_only':True,'no_model_training':True}
    db.close();base.save(STATE/'candidate-index-summary.json',result);print(json.dumps(result),flush=True)


def load_candidate(pointer,files):
    with Path(files[pointer['file_id']]['path']).open('rb') as f:
        f.seek(pointer['offset']);raw=f.read(pointer['length'])
    row=canonicalize(json.loads(raw))
    if row['candidate_identity']!=pointer['candidate_identity'] or row['ppl_input_hash']!=pointer['input_hash']:
        raise ValueError('candidate pointer binding mismatch')
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','index']);p.add_argument('--workers',type=int,default=16);args=p.parse_args()
    if not 1<=args.workers<=32:raise ValueError('worker bound')
    prepare() if args.phase=='prepare' else index(args.workers)
