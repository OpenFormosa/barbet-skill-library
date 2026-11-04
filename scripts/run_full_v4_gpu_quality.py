"""Exhaustive frozen-v26 scoring; each input/output file has a resume receipt."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
import sys
from pathlib import Path
import argparse,json,time,math,shutil,hashlib,tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from barbet_pipeline.common import read_json,save_new,sha_file,digest

STATE=Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
GPU=STATE/'gpu-quality'
MODEL=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
REPO='OpenFormosa/barbet-long-context-sft'


def download(item,destination):
    from barbet_pipeline.anonymous_readback_ranged import _transfer,TransferError,CHUNK_BYTES,RANGE_THRESHOLD
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{item["revision"]}/'+quote(item['remote'],safe='/')
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory(dir=destination.parent,prefix='ranges-') as temporary:
                temporary=Path(temporary)
                if item['bytes']<=RANGE_THRESHOLD:
                    paths=[_transfer(url,temporary/'body',item['bytes'])]
                else:
                    spans=[(start,min(start+CHUNK_BYTES,item['bytes'])-1) for start in range(0,item['bytes'],CHUNK_BYTES)]
                    def fetch(pair):
                        index,span=pair
                        return _transfer(url,temporary/f'{index:06d}',item['bytes'],span)
                    with ThreadPoolExecutor(4) as pool:paths=list(pool.map(fetch,enumerate(spans)))
                with destination.open('xb') as target:
                    for path in paths:
                        with path.open('rb') as source:shutil.copyfileobj(source,target)
            if destination.stat().st_size!=item['bytes'] or sha_file(destination)!=item['sha256']:
                raise ValueError('GPU input file differs from published bytes')
            return
        except TransferError as error:
            delay=error.retry_after if error.retry_after is not None else 2+attempt
            if attempt==2 or delay>600:raise
            time.sleep(delay)


def prepare():
    while not (STATE/'processing-summary.json').exists() or not (GPU/'anchors.json').exists():time.sleep(5)
    summary=read_json(STATE/'processing-summary.json');jobs=[]
    for path in sorted((STATE/'publication/groups').glob('*.json')):
        group=read_json(path)
        assert group['all_anonymous_hashes_match']
        for f in group['files']:
            if '/tokenized/' in f['remote']:jobs.append({**f,'revision':group['head']})
    assert sum(j['rows'] for j in jobs)==summary['counts']['exported']
    assignments=[[] for _ in range(8)];loads=[0]*8
    for item in sorted(jobs,key=lambda j:(-j['bytes'],j['remote'])):
        rank=min(range(8),key=lambda i:loads[i]);assignments[rank].append(item);loads[rank]+=item['bytes']
    model_files={p.name:sha_file(p) for p in [MODEL/'model.safetensors',MODEL/'config.json',MODEL/'modeling_barbet.py',MODEL/'configuration_barbet.py',MODEL/'load_barbet.py',MODEL/'tokenizer.json']}
    assert model_files['model.safetensors']==read_json(GPU/'plan.json')['weight_sha256']
    assert model_files['tokenizer.json']=='f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176'
    plan={'source_processing_summary_sha256':sha_file(STATE/'processing-summary.json'),'rows':summary['counts']['exported'],'input_tokens':summary['counts']['input_tokens'],'supervised_tokens':summary['counts']['supervised_tokens'],'files':len(jobs),'assignments':assignments,'model_files':model_files,'anchor_sha256':sha_file(GPU/'anchors.json'),'scorer_sha256':sha_file('barbet_pipeline/gpu_quality.py'),'runner_sha256':sha_file(__file__),'max_gpus':8,'batch_size':4,'max_padded_batch_tokens':16384,'no_truncation':True,'training_updates':0,'score_quality_is_not_answer_correctness':True}
    save_new(GPU/'full-score-plan.json',plan)
    print(json.dumps({'rows':plan['rows'],'files':len(jobs),'assigned_files':[len(a) for a in assignments]}),flush=True)


def worker(rank):
    os.environ['CUDA_VISIBLE_DEVICES']=str(rank)
    import torch
    torch.set_num_threads(2)
    sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from barbet_pipeline.gpu_quality import FrozenScorer
    while not (GPU/'full-score-plan.json').exists():time.sleep(5)
    plan=read_json(GPU/'full-score-plan.json')
    assert sha_file(__file__)==plan['runner_sha256']
    assert sha_file('barbet_pipeline/gpu_quality.py')==plan['scorer_sha256']
    assert sha_file(GPU/'anchors.json')==plan['anchor_sha256']
    for name,sha in plan['model_files'].items():assert sha_file(MODEL/name)==sha
    sys.path.insert(0,str(MODEL));from load_barbet import load_barbet
    model,tokenizer=load_barbet(MODEL,'cuda:0');scorer=FrozenScorer(model)
    anchor=read_json(GPU/'anchors.json');families=sorted(anchor['prototypes'])
    prototypes=torch.tensor([anchor['prototypes'][f] for f in families])
    scratch=GPU/'scratch'/str(rank);scratch.mkdir(parents=True,exist_ok=True)
    output=GPU/'scores';output.mkdir(exist_ok=True)
    schema=pa.schema([('record_id',pa.string()),('input_tokens',pa.int64()),('supervised_tokens',pa.int64()),('nll_sum',pa.float64()),('nll',pa.float64()),('teacher_forced_top1_correct',pa.int64()),('readout_gradient_sketch_norm',pa.float64()),('semantic_family_cosines',pa.list_(pa.float32(),len(families)))])
    total_rows=0;start=time.monotonic()
    for item in plan['assignments'][rank]:
        key=digest(item['remote']);receipt=GPU/'score-receipts'/f'{key}.json';target=output/f'{key}.parquet'
        if receipt.exists():
            old=read_json(receipt);assert old['input_sha256']==item['sha256'] and old['scorer_sha256']==plan['scorer_sha256']
            assert sha_file(target)==old['output_sha256'];total_rows+=old['rows'];continue
        if target.exists():
            archive=GPU/'incomplete-attempts';archive.mkdir(exist_ok=True)
            target.rename(archive/f'{key}-{time.time_ns()}.parquet')
        if shutil.disk_usage(GPU).free<50*1024**3:raise RuntimeError('GPU scoring free-space floor reached')
        local=scratch/f'{key}.parquet'
        if local.exists():assert sha_file(local)==item['sha256']
        else:download(item,local)
        count=0;tokens=0;supervised=0;loss=0.;correct=0;seen=set();tic=time.monotonic()
        with pq.ParquetWriter(target,schema,compression='zstd') as writer:
            for arrow_batch in pq.ParquetFile(local).iter_batches(batch_size=64):
                rows=sorted(arrow_batch.to_pylist(),key=lambda r:len(r['input_ids']))
                batches=[];batch=[]
                for row in rows:
                    if batch and (len(batch)>=4 or (len(batch)+1)*len(row['input_ids'])>16384):batches.append(batch);batch=[]
                    batch.append(row)
                if batch:batches.append(batch)
                for batch in batches:
                    results=scorer.score(batch);values=[]
                    for row,result in zip(batch,results):
                        assert row['record_id']==result['record_id'] and result['supervised_tokens']==row['pangolin_supervised_tokens']
                        assert row['record_id'] not in seen;seen.add(row['record_id'])
                        feature=result.pop('_feature');result['semantic_family_cosines']=(prototypes@feature).tolist()
                        values.append(result);count+=1;tokens+=result['input_tokens'];supervised+=result['supervised_tokens'];loss+=result['nll_sum'];correct+=result['teacher_forced_top1_correct']
                    writer.write_table(pa.Table.from_pylist(values,schema=schema))
        assert count==item['rows'] and pq.ParquetFile(target).metadata.num_rows==count
        save_new(receipt,{'input_path':item['remote'],'input_sha256':item['sha256'],'input_revision':item['revision'],'rows':count,'input_tokens':tokens,'supervised_tokens':supervised,'nll_sum':loss,'teacher_forced_top1_correct':correct,'output_path':str(target),'output_sha256':sha_file(target),'output_bytes':target.stat().st_size,'model_revision':MODEL.name,'scorer_sha256':plan['scorer_sha256'],'anchor_sha256':plan['anchor_sha256'],'semantic_family_order':families,'seconds':time.monotonic()-tic,'gpu_rank':rank,'training_updates':0,'no_truncation':True})
        assert local.is_relative_to(scratch) and sha_file(local)==item['sha256'];local.unlink()
        total_rows+=count
        print(json.dumps({'rank':rank,'rows_scored':total_rows,'last_file_rows':count,'seconds':time.monotonic()-start}),flush=True)
    save_new(GPU/'workers'/f'{rank}.json',{'complete':True,'rows':total_rows,'seconds':time.monotonic()-start,'plan_sha256':sha_file(GPU/'full-score-plan.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--rank',type=int);args=parser.parse_args()
    if args.prepare:prepare()
    else:worker(args.rank)
