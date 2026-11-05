"""Same frozen per-file computation, with explicit physical recovery slots."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
import argparse,sys,time,json,shutil
from pathlib import Path
from barbet_pipeline.common import read_json,save_new,sha_file,digest
from barbet_pipeline.parallel_recovery import verify_recovery_contract,writer_core
from scripts.run_full_v4_gpu_quality import download,MODEL,GPU
CODE=Path('/root/barbet-full-corpus-v4')
RECOVERY=GPU/'parallel-recovery-20261105'


def process_file(item,slot,scorer,prototypes,families,schema,pq,pa,plan,scratch,output):
    key=digest(item['remote']);receipt=GPU/'score-receipts'/f'{key}.json';target=output/f'{key}.parquet'
    if receipt.exists():
        from barbet_pipeline.gpu_release import validate_score_receipt
        old=read_json(receipt);validate_score_receipt(item,old,plan,GPU)
        return old['rows']
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
    save_new(receipt,{'input_path':item['remote'],'input_sha256':item['sha256'],'input_revision':item['revision'],'rows':count,'input_tokens':tokens,'supervised_tokens':supervised,'nll_sum':loss,'teacher_forced_top1_correct':correct,'output_path':str(target),'output_sha256':sha_file(target),'output_bytes':target.stat().st_size,'model_revision':MODEL.name,'scorer_sha256':plan['scorer_sha256'],'anchor_sha256':plan['anchor_sha256'],'semantic_family_order':families,'seconds':time.monotonic()-tic,'gpu_rank':slot,'original_logical_rank':0,'recovery_plan_sha256':sha_file(RECOVERY/'plan.json'),'recovery_runner_sha256':sha_file(__file__),'training_updates':0,'no_truncation':True})
    assert local.is_relative_to(scratch) and sha_file(local)==item['sha256'];local.unlink()
    return count


def worker(slot):
    os.environ['CUDA_VISIBLE_DEVICES']=str(slot)
    import torch
    torch.set_num_threads(2)
    sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
    import pyarrow as pa
    import pyarrow.parquet as pq
    from barbet_pipeline.gpu_quality import FrozenScorer
    recovery=read_json(RECOVERY/'plan.json');plan=verify_recovery_contract(CODE,GPU,recovery)
    assert sha_file(GPU/'anchors.json')==plan['anchor_sha256']
    for name,sha in plan['model_files'].items():assert sha_file(MODEL/name)==sha
    sys.path.insert(0,str(MODEL));from load_barbet import load_barbet
    model,tokenizer=load_barbet(MODEL,'cuda:0');scorer=FrozenScorer(model)
    anchor=read_json(GPU/'anchors.json');families=sorted(anchor['prototypes'])
    prototypes=torch.tensor([anchor['prototypes'][f] for f in families])
    scratch=RECOVERY/'scratch'/str(slot);scratch.mkdir(parents=True,exist_ok=True)
    output=GPU/'scores'
    schema=pa.schema([('record_id',pa.string()),('input_tokens',pa.int64()),('supervised_tokens',pa.int64()),('nll_sum',pa.float64()),('nll',pa.float64()),('teacher_forced_top1_correct',pa.int64()),('readout_gradient_sketch_norm',pa.float64()),('semantic_family_cosines',pa.list_(pa.float32(),len(families)))])
    rows=0;start=time.monotonic()
    for item in recovery['assignments'][slot]:
        rows+=process_file(item,slot,scorer,prototypes,families,schema,pq,pa,plan,scratch,output)
        print(json.dumps({'slot':slot,'rows':rows,'seconds':time.monotonic()-start}),flush=True)
    assert rows==sum(i['rows'] for i in recovery['assignments'][slot])
    save_new(RECOVERY/'slots'/f'{slot}.json',{'slot':slot,'complete':True,'rows':rows,'seconds':time.monotonic()-start,'recovery_plan_sha256':sha_file(RECOVERY/'plan.json'),'training_updates':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--slot',type=int,required=True);args=parser.parse_args()
    if not 0<=args.slot<8:raise ValueError('Invalid physical GPU slot')
    worker(args.slot)
