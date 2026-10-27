"""Coalesce verified PPL-selected rows without another filtering or ranking step."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts.build_v26_selected_release import STATE as SELECTED,PUBLIC as SOURCE,WAVE
from scripts import compact_memory_release_v2 as exact

STATE=SELECTED/'compact-v1'
DEST=base.WORK/'exports/memory-harness-v2-v26-public'
ROWS=20000


def configure():
    exact.DEST=DEST;exact.STATE=STATE


def job(split):
    configure();receipt=STATE/(split+'.json')
    if receipt.exists():return json.loads(receipt.read_text())
    configs=['records','tokenized','minicpm5_tokenized'];parts={n:[] for n in configs};pending=0;number=0;files=[];counts=Counter();ids=set()
    def flush(n):
        nonlocal pending,number
        for config in configs:
            whole=pa.concat_tables(parts[config]);current=whole.slice(0,n)
            files.append(exact.write_checked(current,DEST/'data'/WAVE/config/f'{split}-{number:05d}.parquet',wave=WAVE,config=config,split=split))
            rest=whole.slice(n);parts[config]=[rest] if len(rest) else []
        pending-=n;number+=1
    for p in sorted((SOURCE/'data'/WAVE/'records').glob(split+'-*.parquet')):
        tables={k:pq.read_table(SOURCE/'data'/WAVE/k/p.name) for k in configs}
        keys=tables['records'].column('record_id').to_pylist()
        if len(set(keys))!=len(keys) or set(keys)&ids:raise ValueError('duplicate compact record')
        if any(t.column('record_id').to_pylist()!=keys for t in tables.values()):raise ValueError('unaligned selected configs')
        ids.update(keys)
        for value in tables['records'].column('source_refs_json').to_pylist():
            selection=json.loads(value)['answer_selection'];counts['selection:'+selection['method']]+=1
        for k,t in tables.items():parts[k].append(t)
        counts['accepted']+=len(keys);pending+=len(keys)
        while pending>=ROWS:flush(ROWS)
    if pending:flush(pending)
    result={'split':split,'counts':dict(counts),'files':files,'all_values_equal_to_verified_source':True}
    base.save(receipt,result);return result


def main():
    STATE.mkdir(parents=True,exist_ok=True);configure()
    summary=json.loads((SELECTED/'public-package-summary.json').read_text())
    # Bind all input bytes before exact concatenation; no replacement of a row
    # or its token labels is permitted in this transport-only step.
    for f in summary['files']:
        if base.file_sha(SOURCE/f['path'])!=f['sha256']:raise ValueError('verified source package changed')
    files=[];counts=Counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(job,['train','validation']):
            files.extend(result['files']);counts.update(result['counts']);print(json.dumps({'compacted':result['split'],'rows':result['counts']['accepted']}),flush=True)
    if counts['accepted']!=summary['counts']['accepted']:raise ValueError('compaction changed episode count')
    pieces=[];pending=0;number=0;proof_ids=set()
    def flush(n):
        nonlocal pieces,pending,number
        whole=pa.concat_tables(pieces);current=whole.slice(0,n)
        files.append(exact.write_checked(current,DEST/'proofs'/WAVE/f'budgets-{number:05d}.parquet',wave=WAVE,config='budget-proof',split=None))
        rest=whole.slice(n);pieces=[rest] if len(rest) else [];pending-=n;number+=1
    for p in sorted((SOURCE/'proofs'/WAVE).glob('*.parquet')):
        table=pq.read_table(p);keys=table.column('record_id').to_pylist()
        if len(set(keys))!=len(keys) or set(keys)&proof_ids:raise ValueError('duplicate selected budget proof')
        proof_ids.update(keys);pieces.append(table);pending+=len(table)
        while pending>=ROWS:flush(ROWS)
    if pending:flush(pending)
    import sqlite3
    db=sqlite3.connect(f'file:{SELECTED / "public-index.sqlite"}?mode=ro',uri=True)
    accepted={r[0] for r in db.execute('SELECT record_id FROM records')};db.close()
    if accepted!=proof_ids:raise ValueError('budget-proof identity coverage changed')
    known=set(json.loads((base.STATE/'postpublication-tool-interface-audit-v2.json').read_text())['record_ids'])
    if accepted&known:raise ValueError('prior function-parameter holds were revived')
    result={**{k:v for k,v in summary.items() if k!='files'},'files':files,
        'counts':{**summary['counts'],**{k:v for k,v in counts.items() if k.startswith('selection:')},'excluded:new_declared_argument_guard':len(known)},
        'all_retained_records_and_token_arrays_equal':True,'all_budget_proofs_preserved':True,
        'rows_per_compact_shard':ROWS,'all_known_first_wave_tool_holds_absent':True}
    base.save(STATE/'summary.json',result);print(json.dumps({'accepted':counts['accepted'],'files':len(files),'selection_counts':dict(counts)}),flush=True)


if __name__=='__main__':main()
