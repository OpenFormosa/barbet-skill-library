"""Canonical unbatched scoring after the retained padded/GEMM diagnostics."""
import json
import time
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scripts.calibrate_v26_answer_ppl import ROOT,STATE,load,sha_file
from barbet_inference.answer_selection import score_batch


def main():
    path=STATE/'canonical-single-v3-validation.json'
    if path.exists():raise ValueError('immutable validation receipt already exists')
    model,renderer,policy=load();items=[]
    for answer in ['26 bolts.','The total is 17 + 9 = 26 bolts.','There are 29 bolts.']:
        items.append(renderer.render([{'role':'system','content':'Answer from the supplied record. Give a concise answer with a checkable explanation.'},
            {'role':'user','content':'Inventory record: the blue box contains 17 bolts and the red box contains 9 bolts. How many bolts do these two boxes contain together?'},
            {'role':'assistant','content':answer,'loss':True}]))
    results=[]
    for item in items:
        score=score_batch(model,[item])[0];repeat=score_batch(model,[item])[0]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ids=torch.tensor([item['input_ids']],device='cuda');labels=torch.tensor([item['labels']],device='cuda')
            h=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state
            reference=float(model._chunked_shifted_loss(h,labels,chunk_size=128))
        results.append({'score':score,'repeat_nll_delta':abs(score['mean_nll']-repeat['mean_nll']),
                        'published_same_kernel_loss':reference,'reference_nll_delta':abs(score['mean_nll']-reference)})
    # Warm model throughput, excluding compilation/model loading.
    start=time.monotonic()
    for _ in range(30):score_batch(model,[items[1]])
    seconds=time.monotonic()-start
    passed=all(r['repeat_nll_delta']<=1e-7 and r['reference_nll_delta']<=1e-4 for r in results)
    receipt={'policy_sha256':sha_file(ROOT/'configs/memory_sft_v2_v26_selection.json'),
        'scorer_sha256':sha_file(ROOT/'barbet_inference/answer_selection.py'),'results':results,
        'canonical_batch_size':1,'weights_trained':False,'all_checks_passed':passed,
        'warm_30_short_fixture_seconds':seconds,'memory_peak_bytes':torch.cuda.max_memory_allocated(),
        'failure_diagnostics_preserved':['v26-ppl-calibration.log','v26-ppl-calibration-diagnostic-v2.log','live-calibration-diagnostic-v3.json'],
        'scope':'Exact unbatched FlashAttention inference scores, same BF16 vocabulary projection shapes as published loss. Multi-sequence padding is not used. Short-fixture throughput is not a corpus ETA.'}
    path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if not passed:raise ValueError('canonical score validation failed')


if __name__=='__main__':main()
