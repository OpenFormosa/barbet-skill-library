"""Bounded anonymous transport recovery for an already committed manifest.

This only reads frozen bytes. Stop the earlier owned verifier before running it;
the original publisher remains responsible for the final proof commit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path

from scripts import build_memory_sft_v2 as base
from scripts import resume_v26_recovery_readback_v2 as bounded


def main(state):
    state = state.resolve()
    if not state.is_relative_to((base.WORK / 'reports').resolve()):
        raise ValueError('readback state outside the dataset reports')
    if (state / 'publication-metadata-commit.json').exists():
        raise ValueError('data readback already finished; use original metadata resume')
    plan_path = state / 'publication-plan.json'
    plan = json.loads(plan_path.read_text())
    data = json.loads((state / 'publication-data-commit.json').read_text())
    if data['plan_sha256'] != base.file_sha(plan_path):
        raise ValueError('committed manifest changed')
    folder = state / 'anonymous-readback/files'
    if not folder.is_dir(): raise ValueError('original readback has not started')
    policy = {'data_commit': data['commit'], 'plan_sha256': data['plan_sha256'],
              'helper_sha256': base.file_sha(Path(bounded.__file__)),
              'driver_sha256': base.file_sha(Path(__file__)),
              'workers': 2, 'per_attempt_wall_seconds': 180, 'max_attempts': 3,
              'chunk_bytes': 65536, 'anonymous': True,
              'same_revision_full_bytes_required': True,
              'model_requests': 0, 'data_or_metadata_publications': 0}
    p = state / 'anonymous-readback/bounded-transport-v3.json'
    if p.exists():
        if json.loads(p.read_text()) != policy: raise ValueError('transport policy changed')
    else: base.save(p, policy)
    bounded.STATE = state
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: bounded.verify(item, data['commit'], folder), plan['manifest']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--state', required=True, type=Path)
    args = parser.parse_args()
    with (base.WORK / 'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.state)
