"""Create a cumulative, smaller-file release without altering any retained row.

The new declared-argument holds are removed from wave 1. Each retained record and
both unshifted token arrays must equal the concatenated source values exactly.
Twenty-thousand-row shards reduce Hub object count and anonymous readback latency.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.build_memory_sft_v2 import WORK, STATE as FIRST, file_sha, save
from scripts.build_memory_sft_wave2 import STATE as SECOND

DEST=WORK/'exports/memory-harness-v2-cumulative-0002'
STATE=WORK/'reports/memory-harness-v2-cumulative-0002'
ROOTS={'wave-0001':WORK/'exports/memory-harness-v2-public',
       'wave-0002':WORK/'exports/memory-harness-v2-wave2-public'}
ROWS=20000


def write_checked(table,path,*,wave,config,split):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not pq.read_table(path).equals(table):raise ValueError('immutable compact shard conflict')
    else:
        temp=path.with_suffix('.partial')
        pq.write_table(table,temp,compression='zstd',compression_level=6,row_group_size=128)
        if not pq.read_table(temp).equals(table):raise ValueError('compact full-array readback failed')
        temp.replace(path)
    return {'path':str(path.relative_to(DEST)),'wave':wave,'config':config,'split':split,
            'rows':table.num_rows,'bytes':path.stat().st_size,'sha256':file_sha(path)}


def job(task):
    wave,split=task;report=STATE/(wave+'-'+split+'.json')
    if report.exists():return json.loads(report.read_text())
    root=ROOTS[wave];names=['records','tokenized','minicpm5_tokenized']
    holds=set(json.loads((FIRST/'postpublication-tool-interface-audit-v2.json').read_text())['record_ids']) if wave=='wave-0001' else set()
    parts={k:[] for k in names};pending=0;shard=0;files=[];counts=Counter();indices=[]
    def flush(n):
        nonlocal pending,shard
        values={k:pa.concat_tables(parts[k]) for k in names}
        for name in names:
            current=values[name].slice(0,n)
            files.append(write_checked(current,DEST/'data'/wave/name/f'{split}-{shard:05d}.parquet',wave=wave,config=name,split=split))
            remainder=values[name].slice(n);parts[name]=[remainder] if remainder.num_rows else []
        pending-=n;shard+=1
    for p in sorted((root/'data'/wave/'records').glob(split+'-*.parquet')):
        tables={k:pq.read_table(root/'data'/wave/k/p.name) for k in names}
        ids=tables['records'].column('record_id').to_pylist()
        if any(t.column('record_id').to_pylist()!=ids for t in tables.values()):raise ValueError('source config alignment failed')
        keep=[i for i,rid in enumerate(ids) if rid not in holds]
        counts['seen']+=len(ids);counts['excluded:new_declared_argument_guard']+=len(ids)-len(keep)
        if not keep:continue
        pick=pa.array(keep,type=pa.int64())
        chosen={k:t.take(pick) for k,t in tables.items()}
        for name in names:parts[name].append(chosen[name])
        pending+=len(keep)
        for row in chosen['records'].to_pylist():
            counts['accepted']+=1;counts['split:'+split]+=1
            for key in ['pangolin_tokens','pangolin_supervised_tokens','minicpm_tokens','minicpm_supervised_tokens','assistant_turns','tool_observations']:
                counts[key]+=row[key]
            for key in ['domain','source','upstream_source','license','verification_tier','memory_placement','observation_origin']:
                counts[key+':'+str(row[key])]+=1
            for key in ['actual_retrieval','recorded_error_followup']:counts[key]+=int(row[key])
            length=max(row['pangolin_tokens'],row['minicpm_tokens'])
            bucket=next(b for b in [4096,8192,16384,32768,65536,131072] if length<=b)
            counts['length_bucket:'+str(bucket)]+=1
            indices.append({k:row[k] for k in ['record_id','family_id','query_hash','split','domain','source','upstream_source']})
        while pending>=ROWS:flush(ROWS)
    if pending:flush(pending)
    index=STATE/(wave+'-'+split+'.index.parquet')
    if indices:pq.write_table(pa.Table.from_pylist(indices),index,compression='zstd')
    result={'wave':wave,'split':split,'counts':dict(counts),'files':files,
        'all_retained_records_and_token_arrays_equal':True,'index':str(index)}
    save(report,result);return result


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    if not (SECOND/'public-package-summary.json').exists():raise ValueError('second-wave verification incomplete')
    save(STATE/'policy.json',{'version':'compact-cumulative-v2-0002','rows_per_shard':ROWS,
        'changes':['Concatenate exact retained rows into larger Parquet shards','Exclude wave-1 unconfirmed root arguments; retain original release in history'],
        'training_text_or_label_rewrite':False,'new_model_training':False,'target':10000000,'target_complete':False})
    totals=Counter();files=[];wave_counts={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(job,[(wave,split) for wave in ROOTS for split in ['train','validation']]):
            totals.update(result['counts']);files.extend(result['files'])
            wave_counts.setdefault(result['wave'],Counter()).update(result['counts'])
            print(json.dumps({'wave':result['wave'],'split':result['split'],'accepted':result['counts'].get('accepted',0)}),flush=True)
    db=sqlite3.connect(STATE/'public-index.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)');db.execute('DELETE FROM records')
    for p in sorted(STATE.glob('*.index.parquet')):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=10000):
            db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)',[(r['record_id'],r['family_id'],r['query_hash'],r['split'],r['domain'],r['source'],r['upstream_source']) for r in batch.to_pylist()]);db.commit()
    for key in ['family_id','query_hash']:
        if db.execute(f'SELECT COUNT(*) FROM (SELECT {key} FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1)').fetchone()[0]:raise ValueError('cumulative split leakage')
    unique={k:db.execute(f'SELECT COUNT(DISTINCT {k}) FROM records').fetchone()[0] for k in ['record_id','family_id','query_hash']}
    accepted={r[0] for r in db.execute('SELECT record_id FROM records')};db.close()
    # Preserve every per-row budget binding; compact proof files too.
    for wave,root in ROOTS.items():
        pieces=[];size=0;number=0;seen=set()
        def flush_proof(n):
            nonlocal pieces,size,number
            table=pa.concat_tables(pieces);now=table.slice(0,n)
            files.append(write_checked(now,DEST/'proofs'/wave/f'budgets-{number:05d}.parquet',wave=wave,config='budget-proof',split=None))
            rest=table.slice(n);pieces=[rest] if rest.num_rows else [];size-=n;number+=1
        for p in sorted((root/'proofs'/wave).glob('*.parquet')):
            table=pq.read_table(p);ids=table.column('record_id').to_pylist();keep=[i for i,r in enumerate(ids) if r in accepted]
            if not keep:continue
            for i in keep:
                if ids[i] in seen:raise ValueError('duplicate budget proof')
                seen.add(ids[i])
            pieces.append(table.take(pa.array(keep,type=pa.int64())));size+=len(keep)
            while size>=ROWS:flush_proof(ROWS)
        if size:flush_proof(size)
        if len(seen)!=wave_counts[wave]['accepted']:raise ValueError('budget proof coverage mismatch')
    summary={'version':'memory-harness-sft-v2-cumulative-0002','status':'LOCAL_VERIFIED_NOT_YET_PUBLISHED',
        'counts':dict(totals),'wave_counts':{k:dict(v) for k,v in wave_counts.items()},'unique_counts':unique,
        'files':files,'all_retained_records_and_token_arrays_equal':True,'all_budget_proofs_preserved':True,
        'cross_split_family_and_query_conflicts':0,'accepted_new_episode_target':10000000,'goal_complete':False,
        'active_legacy_rows':0,'model_efficacy_measured':False}
    save(STATE/'summary.json',summary);print(json.dumps({'accepted':totals['accepted'],'unique':unique,'files':len(files)}),flush=True)


if __name__=='__main__':main()
