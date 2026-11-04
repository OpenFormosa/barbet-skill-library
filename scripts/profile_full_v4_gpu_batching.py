"""Measure batching on real short graph rows; never mutate active scoring workers."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import json
import time
from pathlib import Path
import torch
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, save_new, sha_file
from barbet_pipeline.gpu_quality import FrozenScorer
from scripts.run_full_v4_gpu_quality import download, MODEL, GPU, STATE


def main():
    torch.set_num_threads(2)
    receipts = [read_json(p) for p in (STATE / 'receipts').glob('*.json')]
    source = next(r for r in receipts if r['source_path'] == 'data/wave-0010/records/train-00000.parquet')
    target = next(f['remote'] for f in source['files'] if '/verified/tokenized/train-' in f['remote'])
    plan = read_json(GPU / 'full-score-plan.json')
    item = next(i for a in plan['assignments'] for i in a if i['remote'] == target)
    path = GPU / 'batching-control-input.parquet'
    if path.exists():
        assert sha_file(path) == item['sha256']
    else:
        download(item, path)
    rows = next(pq.ParquetFile(path).iter_batches(batch_size=128)).to_pylist()
    rows = sorted(rows, key=lambda r: len(r['input_ids']))
    assert len(rows) == 128 and max(len(r['input_ids']) for r in rows) < 8192
    sys.path.insert(0, str(MODEL))
    from load_barbet import load_barbet
    model, tokenizer = load_barbet(MODEL, 'cuda:0')
    scorer = FrozenScorer(model)
    scorer.score(rows[:4]); torch.cuda.synchronize()
    results = []
    reference = None
    for size in [4, 16, 1, 32, 8, 64, 4]:
        torch.cuda.synchronize()
        start = time.monotonic()
        values = []
        for begin in range(0, len(rows), size):
            values.extend(scorer.score(rows[begin:begin + size]))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if reference is None:
            reference = values
        nll_errors = [abs(a['nll'] - b['nll']) for a, b in zip(reference, values)]
        cosines = [float(a['_feature'] @ b['_feature']) for a, b in zip(reference, values)]
        passed = all(e <= .02 + .005 * abs(a['nll']) for a, e in zip(reference, nll_errors)) and min(cosines) >= .995
        result = {'batch_size': size, 'rows': len(rows), 'seconds': elapsed,
            'input_tokens_per_second': sum(len(r['input_ids']) for r in rows) / elapsed,
            'maximum_nll_absolute_difference': max(nll_errors), 'minimum_sketch_cosine': min(cosines),
            'preexisting_numerical_tolerances_passed': passed}
        results.append(result)
        print(json.dumps(result), flush=True)
    save_new(GPU / 'batching-profile.json', {'source_input_path': item['remote'], 'source_input_sha256': item['sha256'],
        'record_ids': [r['record_id'] for r in rows], 'model_revision': MODEL.name,
        'scorer_sha256': sha_file('barbet_pipeline/gpu_quality.py'), 'profiler_sha256': sha_file(__file__),
        'results': results, 'minimum_length': min(len(r['input_ids']) for r in rows),
        'maximum_length': max(len(r['input_ids']) for r in rows),
        'scope': 'Engineering control on 128 actual verified graph rows; not model quality validation.',
        'timing_limit': 'Shares GPU 7 with one ongoing full-corpus scorer; throughput is not an isolated hardware benchmark.',
        'training_updates': 0, 'active_worker_configuration_changed': False})


if __name__ == '__main__':
    main()
