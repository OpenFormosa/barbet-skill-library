"""Exhaustive split/identity check over the already byte-bound full source index."""
from pathlib import Path
import json,sqlite3,time
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,sha_file

STATE=Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
SOURCE=Path('/root/barbet-context-sft-data/reports/semantic-curriculum-v3/corpus/summary.json')

def main():
    STATE.mkdir(parents=True,exist_ok=True)
    dest=STATE/'split-index.sqlite'
    if dest.exists():raise FileExistsError(dest)
    db=sqlite3.connect(dest);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA cache_size=-1048576')
    db.execute('CREATE TABLE records (record_id TEXT, root_id TEXT, query_hash TEXT, split TEXT)')
    manifest=read_json(SOURCE);count=0;start=time.time()
    for i,item in enumerate(manifest['files'],1):
        path=item['output_path'];assert sha_file(path)==item['sha256']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['record_id','root_task_id','query_hash','split']):
            rows=batch.to_pylist();db.executemany('INSERT INTO records VALUES (?,?,?,?)',[(r['record_id'],r['root_task_id'],r['query_hash'],r['split']) for r in rows]);count+=len(rows)
        if i%100==0:db.commit();print(json.dumps({'indexed_shards':i,'rows':count}),flush=True)
    assert count==10018967;db.commit()
    conflicts={}
    for column in ['record_id','root_id','query_hash']:
        db.execute('CREATE INDEX idx_'+column+' ON records ('+column+',split)')
        if column=='record_id':
            found=db.execute('SELECT record_id,COUNT(*) FROM records GROUP BY record_id HAVING COUNT(*)>1').fetchall()
        else:
            found=db.execute('SELECT '+column+',COUNT(DISTINCT split) FROM records GROUP BY '+column+' HAVING COUNT(DISTINCT split)>1').fetchall()
        conflicts[column]=found
        print(json.dumps({'checked':column,'conflicts':len(found)}),flush=True)
    db.commit();db.close()
    save_new(STATE/'split-preflight.json',{'rows':count,'source_summary_sha256':sha_file(SOURCE),'conflicts':conflicts,'seconds':time.time()-start,'cross_split_conflicts_are_holds':True,'driver_sha256':sha_file(__file__)})

if __name__=='__main__':main()
