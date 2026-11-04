import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,time,json
from pathlib import Path
from collections import defaultdict
import torch
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json,save_new,sha_file
from barbet_pipeline.gpu_quality import FrozenScorer

torch.set_num_threads(2)
s=Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
plan=read_json(s/'context-probe-plan.json');assert sha_file(s/'context-probes.parquet')==plan['file_sha256']
folder=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
sys.path.insert(0,str(folder));from load_barbet import load_barbet
model,tokenizer=load_barbet(folder,'cuda:0');scorer=FrozenScorer(model,sketch=False)
rows=sorted(pq.read_table(s/'context-probes.parquet').to_pylist(),key=lambda r:len(r['input_ids']))
results=[];start=time.monotonic()
for offset in range(0,len(rows),4):
 batch=rows[offset:offset+4]
 for row,score in zip(batch,scorer.score(batch,projection=False)):
  results.append({**score,**{k:row[k] for k in ['root_task_id','semantic_family_id','condition','correct_candidate','source_record_id']}})
 if offset%500==0:print(json.dumps({'decisions':len(results),'total':len(rows),'seconds':time.monotonic()-start}),flush=True)
assert len(results)==plan['rows']
byroot=defaultdict(dict)
for row in results:byroot[row['root_task_id']][(row['condition'],row['correct_candidate'])]=row
families=defaultdict(list)
for rid,items in byroot.items():
 original=items['original',True];r={'root_task_id':rid,'family':original['semantic_family_id'],'full_nll':original['nll'],'skill_gain_nats':items['no_skill',True]['nll']-original['nll'],'evidence_gain_nats':items['no_evidence',True]['nll']-original['nll'],'ifd_like_ratio':original['nll']/items['answer_only',True]['nll'],'invariance_nll_change':items['invariance',True]['nll']-original['nll']}
 if ('original',False) in items:
  r['pair_correct_sum_logprob']=all(items[c,True]['nll_sum']<items[c,False]['nll_sum'] for c in ['original','counterfactual'])
  r['pair_correct_mean_logprob']=all(items[c,True]['nll']<items[c,False]['nll'] for c in ['original','counterfactual'])
  r['pair_minimum_mean_nll_margin']=min(items[c,False]['nll']-items[c,True]['nll'] for c in ['original','counterfactual'])
 families[r['family']].append(r)
summary=[]
for name,values in sorted(families.items()):
 pairs=[v for v in values if 'pair_correct_sum_logprob' in v]
 summary.append({'family':name,'roots':len(values),'discriminating_first_action_pairs':len(pairs),'sum_logprob_pairs_correct':sum(v['pair_correct_sum_logprob'] for v in pairs),'mean_logprob_pairs_correct':sum(v['pair_correct_mean_logprob'] for v in pairs),**{k:sum(v[k] for v in values)/len(values) for k in ['full_nll','skill_gain_nats','evidence_gain_nats','ifd_like_ratio','invariance_nll_change']}})
save_new(s/'context-probe-scores.json',{'model_revision':folder.name,'input_sha256':plan['file_sha256'],'scorer_sha256':sha_file('barbet_pipeline/gpu_quality.py'),'runner_sha256':sha_file(__file__),'rows':results,'per_root':list(v for values in families.values() for v in values),'seconds':time.monotonic()-start})
save_new(s/'context-probe-summary.json',{'complete':True,'roots':len(byroot),'decisions':len(results),'families':summary,'score_sha256':sha_file(s/'context-probe-scores.json'),'scope':plan['scope'],'training_updates':0,'free_generation':'NOT_RUN','no_missing_evidence_answer_correctness_claim':True,'thresholds_used_to_accept_training_rows':False})
print(json.dumps({'completed':True,'roots':len(byroot),'families':summary},ensure_ascii=False),flush=True)
