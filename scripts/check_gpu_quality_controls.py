import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,time,json,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from barbet_pipeline.common import read_json,save_new,sha_file
from barbet_pipeline.gpu_quality import FrozenScorer,readout_sketch

torch.set_num_threads(2)
state=Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
folder=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
sys.path.insert(0,str(folder))
from load_barbet import load_barbet
start=time.monotonic();model,tokenizer=load_barbet(folder,'cuda:0')
scorer=FrozenScorer(model)
rows=read_json(state/'control-inputs.json')['rows']
print(json.dumps({'event':'loaded','seconds':time.monotonic()-start}),flush=True)
# The statistical comparison tolerances are set before any measurements.
tolerance={'nll_absolute':0.02,'nll_relative':0.005,'sketch_cosine_minimum':0.995}
save_new(state/'control-tolerances.json',tolerance)
small=rows[:4]
one=[scorer.score([row])[0] for row in small]
batch=scorer.score(small)
errors=[abs(a['nll']-b['nll']) for a,b in zip(one,batch)]
cosines=[float(a['_feature']@b['_feature']) for a,b in zip(one,batch)]
assert all(e<=tolerance['nll_absolute']+tolerance['nll_relative']*abs(a['nll']) for e,a in zip(errors,one)),errors
assert min(cosines)>=tolerance['sketch_cosine_minimum'],cosines
# Compare analytic detached-readout gradient against actual autograd on CPU.
g=torch.Generator().manual_seed(7)
h=torch.randn(5,7,generator=g,dtype=torch.float64)
w=torch.randn(11,7,generator=g,dtype=torch.float64,requires_grad=True)
y=torch.tensor([1,2,3,4,5]);p1=torch.randn(11,3,generator=g,dtype=torch.float64);p2=torch.randn(7,4,generator=g,dtype=torch.float64)
logits=h@w.T
loss=torch.nn.functional.cross_entropy(logits,y,reduction='sum');loss.backward()
analytic=readout_sketch(logits.detach().softmax(-1),y,h,p1,p2)
expected=p1.T@w.grad@p2
assert torch.allclose(analytic,expected,atol=1e-10,rtol=1e-10)
# Compare chunked selected-position loss with dense released-model loss.
row=small[0];ids=torch.tensor([row['input_ids']],device='cuda:0');labels=torch.tensor([row['labels']],device='cuda:0')
from torch.nn.attention import SDPBackend,sdpa_kernel
with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
 dense=model(input_ids=ids,labels=labels,use_cache=False,return_dict=True)
 dense_nll=float(dense.loss)
assert abs(dense_nll-one[0]['nll'])<0.02,(dense_nll,one[0]['nll'])
performance=[]
for size in [1,4,8,16,32]:
 torch.cuda.synchronize();tic=time.monotonic()
 result=scorer.score(rows[:size]);torch.cuda.synchronize();elapsed=time.monotonic()-tic
 performance.append({'batch':size,'input_tokens':sum(x['input_tokens'] for x in result),'supervised_tokens':sum(x['supervised_tokens'] for x in result),'seconds':elapsed,'input_tokens_per_second':sum(x['input_tokens'] for x in result)/elapsed})
 print(json.dumps(performance[-1]),flush=True)
result={'passed':True,'revision':folder.name,'weight_sha256':sha_file(folder/'model.safetensors'),'loader_sha256':sha_file(folder/'load_barbet.py'),'scorer_sha256':sha_file('barbet_pipeline/gpu_quality.py'),'control_input_sha256':sha_file(state/'control-inputs.json'),'tolerances':tolerance,'batch_single_nll_absolute_errors':errors,'batch_single_sketch_cosines':cosines,'dense_nll':dense_nll,'selected_chunked_nll':one[0]['nll'],'analytic_gradient_matches_autograd':True,'performance':performance,'training_updates':0,'actual_generation_requests':0,'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),'scope':'numerical and throughput controls; no data quality verdict or learning claim'}
save_new(state/'numerical-controls.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='performance'}),flush=True)
