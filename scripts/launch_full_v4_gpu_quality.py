import os,subprocess,time,json
from pathlib import Path
from barbet_pipeline.common import read_json,save_new,sha_file
s=Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
while not (s/'full-score-plan.json').exists():time.sleep(5)
plan=read_json(s/'full-score-plan.json')
assert plan['max_gpus']==8
# Respect other users' GPU processes; no process is cancelled by this launcher.
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if processes:raise RuntimeError('GPU processes already present; preserve them and inspect before launch')
children=[];handles=[]
for rank in range(8):
 handle=(s/f'worker-{rank}.log').open('x');handles.append(handle)
 env=dict(os.environ,PYTHONPATH='/root/barbet-full-corpus-v4',CUDA_VISIBLE_DEVICES=str(rank),TOKENIZERS_PARALLELISM='false')
 p=subprocess.Popen(['/root/Barbet/.venv/bin/python','-u','scripts/run_full_v4_gpu_quality.py','--rank',str(rank)],cwd='/root/barbet-full-corpus-v4',env=env,stdout=handle,stderr=subprocess.STDOUT)
 children.append(p)
save_new(s/'launch.json',{'pids':[p.pid for p in children],'rank_order':list(range(8)),'time':time.time(),'plan_sha256':sha_file(s/'full-score-plan.json'),'training_updates':0})
print(json.dumps({'launched':len(children),'pids':[p.pid for p in children]}),flush=True)
status={}
while len(status)<len(children):
 for rank,p in enumerate(children):
  code=p.poll()
  if code is not None and rank not in status:
   status[rank]=code;print(json.dumps({'finished_rank':rank,'exit_code':code}),flush=True)
 time.sleep(5)
for handle in handles:handle.close()
save_new(s/'workers-exited.json',{'exit_codes':status,'all_success':all(code==0 for code in status.values())})
if not all(code==0 for code in status.values()):raise RuntimeError('One or more GPU worker failed; successful file receipts are preserved')
