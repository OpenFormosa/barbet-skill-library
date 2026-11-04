"""Continue full processing with a receipt-bound ranked-input correction."""
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
import json,shutil,time,traceback
import barbet_pipeline.full_corpus as initial
import barbet_pipeline.full_corpus_ranked as driver
from barbet_pipeline.full_publication_streamed import publish_group
from barbet_pipeline.common import read_json,save_new,sha_file,digest


def main():
    state=driver.STATE
    plan=read_json(state/'plan.json');amendment=read_json(state/'ranked-input-amendment.json')
    assert sha_file(initial.__file__)==plan['driver_sha256']==amendment['initial_driver_sha256']
    assert sha_file(driver.__file__)==amendment['corrected_driver_sha256']
    assert read_json(state/'ranked-input-controls.json')['passed']
    assert read_json(state/'initial-attempt-closure.json')['all_initial_processes_stopped']
    transport=read_json(state/'ranged-readback-amendment.json')
    assert transport['previous_coordinator_stopped']
    assert read_json(state/'ranged-readback-control.json')['passed']
    import barbet_pipeline.anonymous_readback_ranged as readback
    assert sha_file(readback.__file__)==transport['readback_sha256']
    jobs=plan['jobs'];counts=Counter();sources=Counter();parts=[];started=time.time()
    with ProcessPoolExecutor(128,initializer=driver.init_worker,mp_context=get_context('fork')) as pool:
        prefetched={}
        for start in range(0,len(jobs),128):
            if shutil.disk_usage(state).free<50*1024**3:raise RuntimeError('Free-space floor reached')
            results=[];failures=[]
            futures={}
            for job in jobs[start:start+128]:
                future=prefetched.pop(job['path'],None)
                if future is None:future=pool.submit(driver.process_job,job)
                futures[future]=job
            # Bounded overlap, not an unbounded queue of disk-writing shards.
            upcoming=jobs[start+128:start+192]
            remaining=[j for j in jobs[start:start+128] if not (state/'receipts'/(digest(j['path'])+'.json')).exists()]
            estimate=lambda js:sum(os.path.getsize(j['local'])+os.path.getsize(j['token_local']) for j in js)
            if upcoming and shutil.disk_usage(state).free - 1.5*estimate(remaining) - 2*estimate(upcoming) >= 50*1024**3:
                for job in upcoming:prefetched[job['path']]=pool.submit(driver.process_job,job)
                print(json.dumps({'event':'bounded_prefetch','source_shards':len(upcoming),'max_workers':128}),flush=True)
            for future in as_completed(futures):
                job=futures[future]
                try:result=future.result()
                except Exception:
                    failure={'source_path':job['path'],'traceback':traceback.format_exc(),'not_a_source_quality_verdict':True}
                    save_new(state/'resume-errors'/(digest(job['path'])+'.json'),failure)
                    failures.append(failure);print(json.dumps({'event':'engineering_exception',**failure}),flush=True);continue
                results.append(result)
                print(json.dumps({'event':'shard_processed','source_path':result['source_path'],'counts':result['counts'],
                    'source_shards_completed_in_group':len(results),'group_size':len(futures)}),flush=True)
            if failures:raise RuntimeError(f'{len(failures)} engineering failures preserved; incomplete group not published')
            results.sort(key=lambda r:r['source_path']);publication=publish_group(start//128,results)
            for r in results:counts.update(r['counts']);sources.update(r['by_source']);parts.append(r['source_path'])
            progress={'event':'group_published','source_shards':len(parts),'total_source_shards':len(jobs),'counts':dict(counts),
                'seconds_this_invocation':time.time()-started,'head':publication['head'],'complete':len(parts)==len(jobs)}
            path=state/'progress'/f'{start//128:04d}.json'
            if path.exists():
                previous=read_json(path)
                assert all(previous[k]==progress[k] for k in ['source_shards','counts','head','complete'])
            else:save_new(path,progress)
            print(json.dumps(progress),flush=True)
    assert counts['active_rows_processed']==plan['active_rows_required']==10018967
    assert counts['exported']+counts['quarantined']==counts['active_rows_processed']
    old=read_json(state.parent/'semantic-curriculum-v3/corpus/summary.json')['counts']
    for k,value in old.items():
        if k.startswith('source:'):assert sources['processed:'+k[7:]]==value
    summary={'status':'FULL_CORPUS_PROCESSED_DATA_PUBLISHED_AWAITING_FINAL_CARD','counts':dict(counts),'by_source':dict(sources),
       'source_shards':len(parts),'all_sources_accounted_for':True,'no_sample_limit':True,
       'plan_sha256':sha_file(state/'plan.json'),'code_amendment_sha256':sha_file(state/'ranked-input-amendment.json'),
       'no_model_training':True,'world_truth_or_learning_proven':False}
    save_new(state/'processing-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
