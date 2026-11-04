import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,time,json
from pathlib import Path
import torch
from barbet_pipeline.common import read_json,save_new,sha_file
from barbet_pipeline.gpu_quality import FrozenScorer

torch.set_num_threads(2)
s=Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
folder=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
assert sha_file('barbet_pipeline/gpu_quality.py')==read_json(s/'plan.json')['scorer_sha256']
sys.path.insert(0,str(folder));from load_barbet import load_barbet
model,tokenizer=load_barbet(folder,'cuda:0');scorer=FrozenScorer(model)
rows=sorted(read_json(s/'anchor-inputs.json')['rows'],key=lambda x:len(x['input_ids']))
sums={};counts={};results=[];start=time.monotonic()
for offset in range(0,len(rows),4):
 batch=rows[offset:offset+4]
 for row,result in zip(batch,scorer.score(batch)):
  feature=result.pop('_feature');family=row['semantic_family_id']
  sums[family]=sums.get(family,torch.zeros_like(feature))+feature;counts[family]=counts.get(family,0)+1
  results.append({**result,'semantic_family_id':family})
 if offset%200==0:print(json.dumps({'roots':len(results),'seconds':time.monotonic()-start}),flush=True)
prototypes={k:(v/v.norm()).tolist() for k,v in sums.items()}
save_new(s/'anchors.json',{'model_revision':folder.name,'input_sha256':sha_file(s/'anchor-inputs.json'),'family_counts':counts,'prototypes':prototypes,'all_train_roots':True,'heldout_used':False,'scorer_sha256':sha_file('barbet_pipeline/gpu_quality.py')})
save_new(s/'anchor-scores.json',{'rows':results,'seconds':time.monotonic()-start,'training_updates':0})
print(json.dumps({'completed_roots':len(results),'seconds':time.monotonic()-start}),flush=True)
