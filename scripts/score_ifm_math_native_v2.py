"""Isolate native scoring from publication/auditor-only Python dependencies.

The first launcher failed during import in all eight workers, before any model
load or score. Its logs and bindings remain intact. Numerical scorer, tokenizer,
renderer, candidate files, and native runtime versions are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/root/barbet-memory-harness-v2')
STATE = Path('/root/barbet-context-sft-data/reports/memory-harness-v2-ifm-v1/math-full-v1/v26-selection-v1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def configure():
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    if binding['driver_sha256'] != sha(ROOT / 'scripts/rank_ifm_math_v26_v1.py'):
        raise ValueError('original candidate orchestration changed')
    if binding['structural_guard_driver_sha256'] != sha(ROOT / 'scripts/guard_v26_episode_identity.py'):
        raise ValueError('identity guard changed')
    proof = json.loads((STATE / 'episode-identity-v1/summary.json').read_text())
    if proof['status'] != 'COMPLETE' or not proof['all_original_candidates_covered']:
        raise ValueError('complete identity proof required')


def launch():
    configure(); folder = STATE / 'scores/canonical-v2'
    receipt = STATE / 'score-launch-native-v2.json'
    if receipt.exists(): raise ValueError('isolated launch already recorded; inspect per-bucket progress')
    failure = json.loads((STATE / 'first-launch-import-failure-v1.json').read_text())
    if failure['model_requests_or_scores'] != 0: raise ValueError('first launch had scoring activity')
    for item in failure['logs']:
        if sha(item['path']) != item['sha256']: raise ValueError('first failure evidence changed')
    if list(folder.glob('*.sqlite')) or list(folder.glob('*.binding.json')):
        raise ValueError('unexpected existing score state before prospective repair')
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    children = []; handles = []
    for bucket in range(8):
        handle = (folder / f'bucket-{bucket}-native-v2.log').open('w'); handles.append(handle)
        process = subprocess.Popen(['/root/Barbet/.venv/bin/python', str(Path(__file__).resolve()), 'score', '--bucket', str(bucket)],
            cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(bucket)}, stdout=handle, stderr=subprocess.STDOUT)
        children.append((bucket, process))
    receipt.write_text(json.dumps({'driver_sha256': sha(Path(__file__)),
        'original_import_failure_sha256': sha(STATE / 'first-launch-import-failure-v1.json'),
        'scorer_sha256': sha(ROOT / 'scripts/score_v26_answer_candidates.py'),
        'native_answer_selection_sha256': sha(ROOT / 'barbet_inference/answer_selection.py'),
        'pids': [{'bucket': b, 'pid': p.pid} for b, p in children],
        'runtime_dependencies_modified': False, 'canonical_unpadded_batch_size': 1,
        'candidate_manifest_sha256': sha(STATE / 'candidate-input-files.json')}, indent=2) + '\n')
    failed = []
    for bucket, process in children:
        code = process.wait(); print(json.dumps({'bucket': bucket, 'exit_code': code}), flush=True)
        if code: failed.append(bucket)
    for handle in handles: handle.close()
    if failed: raise RuntimeError('isolated native scoring incomplete: ' + str(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['launch', 'score'])
    parser.add_argument('--bucket', type=int); args = parser.parse_args()
    if args.phase == 'launch': launch()
    else:
        configure()
        if args.bucket is None or not 0 <= args.bucket < 8: raise ValueError('bucket required')
        from scripts import score_v26_answer_candidates as scorer
        scorer.STATE = STATE; scorer.main(args.bucket, 0, 'canonical-v2')
