#!/usr/bin/env python3
"""Freeze named final comparisons before generating any final test answer."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import ROOT, canonical, runtime_hashes, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comparisons', nargs='+', required=True, help='model:policy pairs')
    p.add_argument('--reason', required=True)
    p.add_argument('--config', type=Path, default=HERE / 'policies.json')
    args = p.parse_args()
    target = HERE / 'FINAL_SELECTION.json'
    if target.exists():
        raise RuntimeError('final selection already exists; never overwrite an opened test selection')
    existing = list((HERE / 'work/runs').glob('*/test/**/*.jsonl'))
    if existing:
        raise RuntimeError('test trajectories exist before sealing')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('commit the reviewed implementation before creating the final seal')
    configs = json.loads(args.config.read_text())['policies']
    pairs = [x.split(':') for x in args.comparisons]
    if any(len(x) != 2 or x[0] not in {'mini', 'qwen'} or x[1] not in configs for x in pairs):
        raise ValueError('invalid registered model/policy pair')
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'runtime_hashes': runtime_hashes(),
        'scorer_sha256': sha((HERE / 'score.py').read_bytes()),
        'comparison_script_sha256': sha((HERE / 'compare.py').read_bytes()),
        'goal_sha256': sha((HERE / 'goal.json').read_bytes()),
        'input_sha256': sha((HERE / 'work/test.inputs.jsonl').read_bytes()),
        'label_sha256': sha((HERE / 'work/test.labels.jsonl').read_bytes()),
        'corpus_sha256': sha((HERE / 'work/corpus.jsonl').read_bytes()),
        'policies': {name: configs[name] for _, name in pairs},
        'comparisons': pairs, 'seeds': [20261027, 20261028, 20261029],
        'serving': {'batch_size': 128, 'max_seqs': 128, 'max_model_len': 49152, 'eager': True},
        'selection_reason': args.reason,
        'policy_file_sha256': sha(args.config.read_bytes()),
        'stopping_rule': json.loads((HERE / 'goal.json').read_text())['search']['stopping'],
        'validation_summary_sha256': sha((HERE / 'work/scores/confirm/validation-summary.json').read_bytes()),
        'no_post_test_policy_selection': True}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sealed': str(target), 'seal_sha256': sha(canonical(result)), 'comparisons': pairs}))


if __name__ == '__main__':
    main()
