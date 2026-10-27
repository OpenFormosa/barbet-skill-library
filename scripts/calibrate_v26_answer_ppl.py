"""Inference-only calibration of answer masks and native v26 likelihood."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import torch
from tokenizers import Tokenizer
from barbet_inference.answer_selection import NativeAnswerRenderer, score_batch

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66')
STATE=Path('/root/barbet-context-sft-data/reports/memory-harness-v2-v26-selection')


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def load():
    policy=json.loads((ROOT/'configs/memory_sft_v2_v26_selection.json').read_text())
    for filename,key in [('model.safetensors','weights_sha256'),('tokenizer.json','tokenizer_sha256'),
                         ('chat_template.jinja','native_template_sha256'),('modeling_barbet.py','modeling_sha256'),('load_barbet.py','loader_sha256')]:
        if sha_file(MODEL/filename)!=policy[key]:raise ValueError('model artifact changed: '+filename)
    torch.set_num_threads(4);torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0]<8*1024**3:raise ValueError('insufficient unreserved GPU memory')
    sys.path.insert(0,str(MODEL))
    from load_barbet import load_barbet
    model,tokenizer=load_barbet(str(MODEL),device='cuda:0')
    return model,NativeAnswerRenderer((MODEL/'chat_template.jinja').read_text(),tokenizer),policy


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    report=STATE/'live-calibration-diagnostic-v3.json'
    if report.exists():raise ValueError('calibration receipt already exists')
    model,renderer,policy=load()
    prefix=[{'role':'system','content':'Answer from the supplied record. Give a concise answer with a checkable explanation.'},
            {'role':'user','content':'Inventory record: the blue box contains 17 bolts and the red box contains 9 bolts. How many bolts do these two boxes contain together?'}]
    fixtures=[prefix+[{'role':'assistant','content':answer,'loss':True}] for answer in
              ['26 bolts.','The total is 17 + 9 = 26 bolts.','There are 29 bolts.']]
    items=[renderer.render(m) for m in fixtures]
    start=time.monotonic();single=[score_batch(model,[x])[0] for x in items];single_seconds=time.monotonic()-start
    start=time.monotonic();batch=score_batch(model,items);batch_seconds=time.monotonic()-start
    max_delta=max(abs(a['mean_nll']-b['mean_nll']) for a,b in zip(single,batch))
    # Independently use the published model loss, with all prompt labels masked.
    x=items[0]
    with torch.inference_mode():
        ids=torch.tensor([x['input_ids']],device='cuda');labels=torch.tensor([x['labels']],device='cuda')
        h=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state
        reference=float(model._chunked_shifted_loss(h,labels,chunk_size=64))
    receipt={'model':policy['model'],'revision':policy['revision'],'policy_sha256':sha_file(ROOT/'configs/memory_sft_v2_v26_selection.json'),
             'single':single,'padded_batch':batch,'maximum_padded_nll_delta':max_delta,
             'published_loss_reference':reference,'published_loss_delta':abs(reference-single[0]['mean_nll']),
             'single_seconds':single_seconds,'batch_seconds':batch_seconds,
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'native_single_score_matches_published_loss':abs(reference-single[0]['mean_nll'])<=1e-4,'padded_batch_gate_passed':max_delta<=0.01,
             'maximum_padded_nll_tolerance':0.01,'model_weights_trained':False,'fixtures_are_not_training_data':True,
             'first_attempt':'Original v1 calibration stopped at padded delta gate; v2 stopped at published loss comparison. Both logs retained. This diagnostic records all values and unchanged gates.',
             'quality_note':'PPL is not a correctness metric. The third answer is deliberately wrong.',
             'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','mamba-ssm','causal-conv1d']}}
    report.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
