"""Run every source shard; no pilot row limit and no training step."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys
from transformers import AutoTokenizer
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json,shutil,time
from barbet_pipeline.full_corpus import prepare,init_worker,process_job,STATE
from barbet_pipeline.full_publication import publish_group
from barbet_pipeline.common import read_json,save_new,sha_file

def main():
    assert read_json(STATE/'live-tokenizer-controls.json')['passed']
    assert read_json(STATE/'actual-source-integration-controls-v2.json')['all_completed']
    plan=read_json(STATE/'plan.json') if (STATE/'plan.json').exists() else prepare()
    import barbet_pipeline.full_corpus as driver
    assert sha_file(driver.__file__)==plan['driver_sha256']
    jobs=plan['jobs'];counts=Counter();sources=Counter();parts=[];started=time.time()
    workers=128;group_size=128
    with ProcessPoolExecutor(workers,initializer=init_worker,mp_context=get_context('fork')) as pool:
        for start in range(0,len(jobs),group_size):
            if shutil.disk_usage(STATE).free<50*1024**3:raise RuntimeError('Free-space floor reached before group; no original files removed')
            results=[]
            futures={pool.submit(process_job,job):job for job in jobs[start:start+group_size]}
            for future in as_completed(futures):
                result=future.result();results.append(result)
                print(json.dumps({'event':'shard_processed','source_path':result['source_path'],'counts':result['counts'],
                                  'source_shards_completed_in_group':len(results),'group_size':len(futures)}),flush=True)
            results.sort(key=lambda r:r['source_path'])
            publication=publish_group(start//group_size,results)
            for r in results:counts.update(r['counts']);sources.update(r['by_source']);parts.append(r['source_path'])
            progress={'event':'group_published','source_shards':len(parts),'total_source_shards':len(jobs),'counts':dict(counts),
                'seconds_this_invocation':time.time()-started,'head':publication['head'],'complete':len(parts)==len(jobs)}
            save_new(STATE/'progress'/f'{start//group_size:04d}.json',progress)
            print(json.dumps(progress),flush=True)
    assert counts['active_rows_processed']==plan['active_rows_required']==10018967
    assert counts['exported']+counts['quarantined']==counts['active_rows_processed']
    old=read_json(STATE.parent/'semantic-curriculum-v3/corpus/summary.json')['counts']
    for k,value in old.items():
        if k.startswith('source:'):assert sources['processed:'+k[7:]]==value
    summary={'status':'FULL_CORPUS_PROCESSED_DATA_PUBLISHED_AWAITING_FINAL_CARD','counts':dict(counts),'by_source':dict(sources),
       'source_shards':len(parts),'all_sources_accounted_for':True,'no_sample_limit':True,
       'plan_sha256':sha_file(STATE/'plan.json'),'no_model_training':True,'world_truth_or_learning_proven':False}
    save_new(STATE/'processing-summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
