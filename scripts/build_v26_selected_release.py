"""Compile and recheck the cumulative PPL-selected release with frozen formats."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts import package_memory_sft_v2 as package
from scripts.finalize_v26_answer_selection import STATE,OUT
from scripts.build_memory_sft_wave2 import STATE as SECOND
from barbet_inference.memory import canonical

FIRST=base.STATE
PUBLIC=base.WORK/'exports/memory-harness-v2-v26-selected-public'
WAVE='wave-0003'


def compile_all(workers):
    if (STATE/'SUPERSEDED.json').exists():raise ValueError('selected release was superseded before compilation')
    if not (STATE/'dedup-summary.json').exists():raise ValueError('PPL selection incomplete')
    base.STATE=STATE;base.OUT=OUT;base.compile_all(workers,False)


def prepare_audit():
    if (STATE/'SUPERSEDED.json').exists():raise ValueError('selected release was superseded before review')
    if not (STATE/'dedup-summary.json').exists():raise ValueError('PPL selection incomplete')
    path=STATE/'normalized/pilot-selected-winners.jsonl';path.parent.mkdir(exist_ok=True)
    if path.exists():return
    reviewed=set();source_files=[]
    for state in [FIRST,SECOND]:
        j=json.loads((state/'semantic-audit-v1/summary.json').read_text());reviewed.update(r['record_id'] for r in j['results'])
        source_files+=json.loads((state/'build-source-lock.json').read_text())['files']
    pools=defaultdict(list)
    for p in sorted((STATE/'selected').glob('*.jsonl')):
        for line in p.open():
            r=json.loads(line)
            if r['record_id'] in reviewed:continue
            values=pools[r['domain']];values.append((base.sha('selected-winner-audit:'+r['record_id']),r))
            if len(values)>16:values.sort(key=lambda x:x[0]);del values[8:]
    with path.open('w') as f:
        for domain,values in sorted(pools.items()):
            for _,r in sorted(values,key=lambda x:x[0])[:8]:f.write(canonical(r)+'\n')
    base.save(STATE/'build-source-lock.json',{'files':source_files,'scope':'Pinned sources reused from the two earlier waves; no new source downloads or answers generated for PPL selection.'})


def audit():
    prepare_audit()
    from scripts import audit_memory_sft_v2 as reviewer
    reviewer.STATE=STATE;reviewer.main()


def package_init():
    package.STATE=STATE;package.OUT=OUT;package.PUBLIC=PUBLIC;package.WAVE=WAVE;package.init()
    # Five search examples already had a positive first review. They are not
    # rereviewed or relabeled as fresh approvals in the new winner diagnosis.
    original=json.loads((FIRST/'semantic-audit-v1/summary.json').read_text())
    package.SEARCH_ALLOWED.update(r['record_id'] for r in original['results'] if r['domain']=='search' and r['verdict']=='ACCEPT')


def package_all(workers):
    if (STATE/'SUPERSEDED.json').exists():raise ValueError('selected release was superseded before packaging')
    if not (STATE/'compile-summary.json').exists():raise ValueError('compiled data incomplete')
    paths=sorted((STATE/'compiled').glob('[0-9]*.json'))
    if len(paths)!=json.loads((STATE/'dedup-summary.json').read_text())['shards']:raise ValueError('missing compiled shard')
    totals=Counter();files=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=package_init) as pool:
        for n,result in enumerate(pool.map(package.job,paths),1):
            totals.update(result['counts']);files.extend(result['files'])
            if n%25==0 or n==len(paths):print(json.dumps({'shards':n,'total':len(paths),'counts':dict(totals)}),flush=True)
    db=sqlite3.connect(STATE/'public-index.sqlite')
    db.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)')
    for p in sorted((STATE/'public-package').glob('*.index.parquet')):
        rows=pq.read_table(p).to_pylist()
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',[(r['record_id'],r['family_id'],r['query_hash'],r['split'],r['domain'],r['source'],r['upstream_source']) for r in rows]);db.commit()
    unique={k:db.execute(f'SELECT COUNT(DISTINCT {k}) FROM records').fetchone()[0] for k in ['record_id','family_id','query_hash']}
    for key in ['family_id','query_hash']:
        if db.execute(f'SELECT COUNT(*) FROM (SELECT {key} FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1)').fetchone()[0]:raise ValueError('selected release split leakage')
    db.execute('ATTACH DATABASE ? AS previous',(str(FIRST/'public-index.sqlite'),))
    if db.execute('SELECT COUNT(*) FROM records r JOIN previous.records p USING(query_hash) WHERE r.split != p.split').fetchone()[0]:raise ValueError('published question fold changed')
    db.close();audit_scope=json.loads((STATE/'semantic-audit-v1/summary.json').read_text())
    base.save(STATE/'public-package-summary.json',{'wave':WAVE,'status':'LOCAL_VERIFIED_NOT_YET_PUBLISHED','counts':dict(totals),'unique_counts':unique,'files':files,
        'selection':'One minimum v26 answer-PPL candidate per complete eligible input group; singleton groups require no ranking.',
        'ppl_binding_sha256':base.file_sha(STATE/'ppl-score-binding.json'),'old_source_order_release_superseded':True,
        'all_records_and_arrays_reread':True,'all_answer_free_qwen_chat_and_completion_budgets_measured':True,
        'all_training_and_native_minicpm_budgets_with_4096_reserve':True,'all_published_query_folds_preserved':True,
        'new_winner_audit_scope':{'samples':len(audit_scope['results']),'verdicts':audit_scope['counts'],'all_rows_semantically_verified':False},
        'legacy_active_rows':0,'accepted_episodes_target':10000000,'full_goal_complete':False,'model_weights_trained':False,'model_efficacy_measured':False})
    print(json.dumps({'accepted':totals['accepted'],'unique':unique,'files':len(files)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','compile','package']);p.add_argument('--workers',type=int,default=24);args=p.parse_args()
    if not 1<=args.workers<=32:raise ValueError('CPU worker bound')
    if args.phase=='audit':audit()
    elif args.phase=='compile':compile_all(args.workers)
    else:package_all(args.workers)
