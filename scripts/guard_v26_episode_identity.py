"""Do not count source dialogue-prefix views or simulated result variants as tasks."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import prepare_v26_answer_candidates as candidates
from scripts import build_memory_sft_v2 as base
from barbet_inference.answer_selection import input_identity
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected

STATE=candidates.STATE/'episode-identity-v1'


def identities(row):
    static=[m for m in row['messages'] if m['role']!='tool']
    task=input_identity(static,row['tools'])
    messages=json.loads(canonical(row['messages']));aliases={};prefixes=[]
    for n,m in enumerate(messages):
        for call in m.get('tool_calls',[]):
            aliases[call['id']]='call_'+str(len(aliases));call['id']=aliases[call['id']]
        if 'tool_call_id' in m:m['tool_call_id']=aliases[m['tool_call_id']]
        if m['role']=='assistant' and n<len(messages)-1:
            prefixes.append(base.sha(canonical({'messages':messages[:n+1],'tools':row['tools']})))
    return task,prefixes


def job(spec):
    receipt=STATE/'parts'/f"{spec['file_id']:05d}.json"
    if receipt.exists():return json.loads(receipt.read_text())
    counts=Counter();rows=[];prefixes=set();digest=hashlib.sha256()
    with Path(spec['path']).open('rb') as f:
        for line in f:
            digest.update(line);r=json.loads(line)
            try:candidates.eligible(r)
            except Rejected:continue
            r=candidates.canonicalize(r);task,previous=identities(r)
            rows.append({'candidate_identity':r['candidate_identity'],'input_hash':r['ppl_input_hash'],'base_input_hash':task});prefixes.update(previous)
            counts['eligible_rows']+=1
    if digest.hexdigest()!=spec['sha256']:raise ValueError('normalized bytes changed during episode guard')
    receipt.parent.mkdir(exist_ok=True);case_path=receipt.with_suffix('.cases.parquet');prefix_path=receipt.with_suffix('.prefixes.parquet')
    if rows:pq.write_table(pa.Table.from_pylist(rows),case_path,compression='zstd')
    if prefixes:pq.write_table(pa.table({'candidate_identity':sorted(prefixes)}),prefix_path,compression='zstd')
    result={'file_id':spec['file_id'],'counts':dict(counts),'cases':str(case_path) if rows else None,'prefixes':str(prefix_path) if prefixes else None}
    base.save(receipt,result);return result


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    if (STATE/'summary.json').exists():raise ValueError('episode guard already frozen')
    base.save(STATE/'policy.json',{'version':'strict-episode-identity-v1','reason':'Recovered answer alternatives must not reintroduce training-prefix views or pretend that different simulated results for the same task are new independent tasks.',
        'strict_dialogue_prefixes':'Drop an exact provider-ID-normalized message prefix when a longer eligible recorded dialogue exists.',
        'task_conditions':'For the same fixed user/scenario/tool-registry input, hold competing complete trajectories with different observation conditions. Do not compare likelihoods as if they share an identical condition.',
        'quality_reranking':'Recompute the minimum from saved scores among remaining eligible answers. Keep every first semantic hold. Never invent a score or choose a first-source fallback.',
        'weights_trained':False})
    jobs=json.loads((candidates.STATE/'candidate-input-files.json').read_text());parts=[]
    with ProcessPoolExecutor(max_workers=16,initializer=candidates.init) as pool:
        for n,result in enumerate(pool.map(job,jobs),1):
            parts.append(result)
            if n%250==0 or n==len(jobs):print(json.dumps({'guarded_files':n,'total':len(jobs)}),flush=True)
    db=sqlite3.connect(STATE/'identity.sqlite')
    db.execute('CREATE TABLE cases(candidate_identity TEXT PRIMARY KEY,input_hash TEXT,base_input_hash TEXT)')
    db.execute('CREATE TABLE prefixes(candidate_identity TEXT PRIMARY KEY)')
    for part in parts:
        if part['cases']:
            db.executemany('INSERT OR IGNORE INTO cases VALUES (?,?,?)',[(r['candidate_identity'],r['input_hash'],r['base_input_hash']) for r in pq.read_table(part['cases']).to_pylist()])
        if part['prefixes']:
            db.executemany('INSERT OR IGNORE INTO prefixes VALUES (?)',[(r,) for r in pq.read_table(part['prefixes']).column(0).to_pylist()])
        db.commit()
    db.execute('ATTACH DATABASE ? AS original',(str(candidates.STATE/'candidates.sqlite'),))
    if db.execute('SELECT COUNT(*) FROM original.candidates LEFT JOIN cases USING(candidate_identity) WHERE cases.candidate_identity IS NULL').fetchone()[0]:raise ValueError('episode identity coverage incomplete')
    db.execute('CREATE INDEX static_task_inputs ON cases(base_input_hash,input_hash)')
    db.execute('CREATE TABLE ambiguous AS SELECT base_input_hash FROM cases c WHERE NOT EXISTS(SELECT 1 FROM prefixes p WHERE p.candidate_identity=c.candidate_identity) GROUP BY base_input_hash HAVING COUNT(DISTINCT input_hash)>1')
    db.execute('CREATE UNIQUE INDEX ambiguous_tasks ON ambiguous(base_input_hash)')
    db.execute("CREATE TABLE holds AS SELECT c.candidate_identity, CASE WHEN p.candidate_identity IS NOT NULL THEN 'strict_dialogue_prefix_view' ELSE 'same_task_different_observation_conditions' END AS reason FROM cases c LEFT JOIN prefixes p USING(candidate_identity) LEFT JOIN ambiguous a USING(base_input_hash) WHERE p.candidate_identity IS NOT NULL OR a.base_input_hash IS NOT NULL")
    db.execute('CREATE UNIQUE INDEX held_candidates ON holds(candidate_identity)');db.commit()
    result={'counts':{r[0]:r[1] for r in db.execute('SELECT reason,COUNT(*) FROM holds GROUP BY reason')},
        'distinct_candidate_answers':db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],
        'ambiguous_static_task_inputs':db.execute('SELECT COUNT(*) FROM ambiguous').fetchone()[0],
        'all_original_candidates_covered':True,'status':'COMPLETE','original_scores_not_rewritten':True}
    db.close();base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
