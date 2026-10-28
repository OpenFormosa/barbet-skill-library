"""Two real retained rows, no-grad v26 loss/shift smoke. Never trains weights."""
import json
from pathlib import Path
import sys

import torch  # Use the pinned Barbet environment before optional data helpers.

sys.path.append('/usr/local/lib/python3.12/dist-packages')
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from scripts.calibrate_v26_answer_ppl import load, MODEL
from scripts import publish_barbet_only_v1 as release
from scripts import audit_barbet_continuation_v1 as audit
from scripts import publish_barbet_readiness_v1 as correction
from barbet_inference.sft import pangolin_render
from barbet_inference.training import BarbetSFTCollator


def main():
    out = audit.STATE / 'native-collated-forward-v1.json'
    policy_path = audit.STATE / 'native-collated-forward-policy-v1.json'
    if out.exists() or policy_path.exists(): raise ValueError('First forward smoke already started')
    excluded = {rid for job in release.read(correction.STATE / 'plan.json')['jobs'] for rid in job['removed_ids']}
    pointers = {p['record_id']: p for p in release.read(audit.STATE / 'sample-pointers.json')}
    tok = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    candidates = []
    for packet in release.read(audit.STATE / 'content-sample-packets.private.json'):
        if packet['record_id'] in excluded: continue
        text, _ = pangolin_render(packet['messages'], packet['tools'])
        length = len(tok.encode(text, add_special_tokens=False).ids)
        if length <= 1024: candidates.append((length, packet['record_id']))
    candidates.sort(); chosen = [candidates[0]]
    chosen.append(next(p for p in candidates[1:] if p[0] != chosen[0][0]))
    rows = []
    for length, rid in chosen:
        pointer = pointers[rid]
        row = pq.read_table(release.parent.DEST / pointer['path'].replace('/records/', '/tokenized/')).slice(pointer['row'], 1).to_pylist()[0]
        if row['record_id'] != rid or row['total_tokens'] != length: raise ValueError('Stored smoke row differs')
        rows.append(row)
    release.base.save(policy_path, {'driver_sha256': release.base.file_sha(Path(__file__)),
        'selection': 'Shortest fixed stratified nonheld rows with different full lengths, at most 1024 tokens',
        'records': [{'record_id': rid, 'tokens': length} for length, rid in chosen],
        'model_revision': release.MODEL_REVISION, 'gradient_updates': 0, 'generation_requests': 0,
        'max_loss_delta': 1e-4, 'no_answer_selection_or_old_calibration_changes': True,
        'source_programs_executed': False})
    model, _, native_policy = load(); model.eval()
    batch = {k: v.to('cuda:0') for k, v in BarbetSFTCollator()(rows).items()}
    anchor = model.model.embed_tokens.weight[:1].detach().clone()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = model(**batch, use_cache=False, logits_to_keep=1, loss_chunk_size=128, return_dict=True)
        loss = float(output.loss)
        hidden = model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                             use_cache=False, return_dict=True).last_hidden_state
        target = torch.full_like(batch['labels'], -100); target[:, :-1] = batch['labels'][:, 1:]
        total = torch.zeros((), device='cuda:0', dtype=torch.float32)
        for start in range(0, hidden.shape[1], 128):
            stop = min(start + 128, hidden.shape[1])
            logits = model.lm_head(hidden[:, start:stop]).float()
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target[:, start:stop].reshape(-1),
                                     ignore_index=-100, reduction='sum')
        reference = float(total / (target != -100).sum())
    if not torch.isfinite(torch.tensor([loss, reference])).all() or abs(loss - reference) > 1e-4:
        raise ValueError('Native masked loss / independent explicit single shift differs')
    if model.training or any(p.grad is not None for p in model.parameters()) or not torch.equal(anchor, model.model.embed_tokens.weight[:1]):
        raise ValueError('No-grad smoke changed training/gradient state')
    result = {'records': [rid for _, rid in chosen], 'lengths': [length for length, _ in chosen],
              'native_masked_loss': loss, 'independent_single_shift_CE': reference,
              'absolute_delta': abs(loss-reference), 'all_checks_passed': True,
              'right_padding_tokens': int((batch['attention_mask'] == 0).sum()),
              'supervised_tokens': int((batch['labels'] != -100).sum()),
              'model_revision': native_policy['revision'], 'model_eval_mode': True,
              'gradient_updates': 0, 'parameter_grads_present': False,
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'not_training_or_model_improvement_evidence': True}
    release.base.save(out, result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
