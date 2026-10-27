"""Verify the second wave with the frozen renderers and explicit wave-2 scope."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import sqlite3
import pyarrow.parquet as pq
from scripts import package_memory_sft_v2 as package
from scripts.build_memory_sft_wave2 import STATE, OUT, FIRST
from scripts.build_memory_sft_v2 import WORK, RAW, save, file_sha

PUBLIC=WORK/'exports/memory-harness-v2-wave2-public'
WAVE='wave-0002'


def configure():
    package.STATE=STATE;package.OUT=OUT;package.PUBLIC=PUBLIC;package.WAVE=WAVE


def init():
    configure();package.init()


def main():
    configure()
    if not (STATE/'compile-summary.json').exists():raise ValueError('wave compilation incomplete')
    paths=sorted((STATE/'compiled').glob('[0-9]*.json'))
    if len(paths)!=json.loads((STATE/'dedup-summary.json').read_text())['shards']:raise ValueError('missing compiled shard')
    totals=Counter();files=[]
    with ProcessPoolExecutor(max_workers=24,initializer=init) as pool:
        for n,r in enumerate(pool.map(package.job,paths),1):
            totals.update(r['counts']);files.extend(r['files'])
            if n%20==0 or n==len(paths):print(json.dumps({'shards':n,'total':len(paths),'accepted':totals['accepted']}),flush=True)
    db=sqlite3.connect(STATE/'public-index.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)')
    db.execute('DELETE FROM records')
    for p in sorted((STATE/'public-package').glob('*.index.parquet')):
        rows=pq.read_table(p).to_pylist()
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',[(r['record_id'],r['family_id'],r['query_hash'],r['split'],r['domain'],r['source'],r['upstream_source']) for r in rows]);db.commit()
    unique={k:db.execute(f'SELECT COUNT(DISTINCT {k}) FROM records').fetchone()[0] for k in ['record_id','family_id','query_hash']}
    db.execute('ATTACH DATABASE ? AS previous',(str(FIRST/'public-index.sqlite'),))
    for key in ['record_id','query_hash']:
        overlap=db.execute(f'SELECT COUNT(*) FROM records JOIN previous.records USING ({key})').fetchone()[0]
        if overlap:raise ValueError('unexpected prior-wave identity overlap: '+key)
    for key in ['family_id','query_hash']:
        mixed=db.execute(f'SELECT COUNT(*) FROM (SELECT {key} FROM (SELECT {key},split FROM records UNION ALL SELECT {key},split FROM previous.records) GROUP BY {key} HAVING COUNT(DISTINCT split)>1)').fetchone()[0]
        if mixed:raise ValueError('cross-wave split leakage: '+key)
    db.close()
    audit=json.loads((STATE/'semantic-audit-v1/summary.json').read_text())
    save(STATE/'public-package-summary.json',{'wave':WAVE,'status':'LOCAL_VERIFIED_NOT_YET_PUBLISHED',
        'counts':dict(totals),'unique_counts':unique,'files':files,'accepted_episodes_target':10000000,
        'full_goal_complete':False,'legacy_active_rows':0,'model_efficacy_measured':False,
        'all_records_and_arrays_reread':True,'all_answer_free_qwen_chat_and_completion_budgets_measured':True,
        'all_training_and_native_minicpm_budgets_with_4096_reserve':True,
        'cross_wave_record_and_query_overlap':0,'cross_wave_family_and_query_split_conflicts':0,
        'semantic_audit_scope':{'samples':len(audit['results']),'verdicts':audit['counts'],'representative_population_accuracy':False},
        'new_guard':'Reject unconfirmed root function parameters despite permissive JSON Schema defaults; preserve explicit open interfaces',
        'tokenizers':{'Pangolin':file_sha(WORK/'data/blobs/tokenizer/tokenizer.json'),
                      'MiniCPM5-2B':file_sha(RAW/'minicpm5-tokenizer/tokenizer.json'),
                      'served_Qwen':file_sha(WORK/'inference/qwen38/model-fp8/tokenizer.json')}})
    print(json.dumps({'accepted':totals['accepted'],'unique':unique,'files':len(files)}),flush=True)


if __name__=='__main__':main()
