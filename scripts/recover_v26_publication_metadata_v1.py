"""Restore missing recipe metadata in a new commit, retaining the first failure.

No training payload changes. The repaired commit receives a new full anonymous
readback; previous receipts are never relabeled as verification of a new commit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi, CommitOperationAdd
from scripts import build_memory_sft_v2 as base
from scripts.prune_v26_query_aliases_v1 import STATE as ORIGINAL, DEST as SOURCE
from scripts import publish_memory_cumulative_v2 as transport

STATE = ORIGINAL / 'publication-recovery-v1'
DEST = base.WORK / 'exports/memory-harness-v2-v26-complete-public-v2-recovered'
FIRST = base.WORK / 'reports/memory-harness-v2'


def metadata_only_paths(missing):
    if any(p.startswith(('data/', 'proofs/')) for p in missing):
        raise ValueError('training payload gap cannot be repaired as metadata only')
    if set(missing) != {'recipe/memory_sft_v2.json', 'recipe/memory_sft_v2_scale_10m.json'}:
        raise ValueError('unexpected missing metadata scope')
    return sorted(missing)


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists(): raise ValueError('recovery plan already frozen')
    failure = json.loads((ORIGINAL / 'publication-tree-failure-v1.json').read_text())
    missing = metadata_only_paths(failure['missing'])
    if failure['extra'] or failure['missing_training_payload_files']: raise ValueError('unexpected original tree defect')
    old = json.loads((ORIGINAL / 'publication-plan.json').read_text())
    receipts = [json.loads(p.read_text()) for p in (ORIGINAL / 'anonymous-readback/files').glob('*.json')]
    verified = {r['path']: r for r in receipts}
    for item in old['manifest']:
        if item['path'] in missing: continue
        if verified.get(item['path']) != {**item, 'revision': failure['commit'], 'anonymous': True}:
            raise ValueError('original nonmissing file lacks its actual first readback receipt')
        # A separate destination preserves all original files and first-failure
        # manifests. Hard links are never opened for mutation.
    for item in old['manifest']:
        dst = DEST / item['path']; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): os.link(SOURCE / item['path'], dst)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    report = {'first_data_commit': failure['commit'], 'missing_metadata_paths': missing,
        'first_commit_training_files_missing': 0, 'first_commit_other_files_anonymously_verified': len(verified),
        'cause': 'The client removes unchanged Add operations as no-ops. The old publisher also deleted their parent recipe folder, so two unchanged configuration files were removed without being re-added.',
        'fix': 'Only remove obsolete paths or subtrees that contain no planned retained files. Preserve shared ancestors of unchanged files.',
        'original_failure_receipt_sha256': base.file_sha(ORIGINAL / 'publication-tree-failure-v1.json'),
        'original_first_readback_log_sha256': base.file_sha(base.WORK / 'reports/v26-query-pruned-anonymous-verify-v1.log'),
        'original_plan_sha256': base.file_sha(ORIGINAL / 'publication-plan.json'),
        'transport_library_commit': code, 'data_recipe_library_commit': old['library_commit'],
        'no_training_payload_values_changed': True, 'previous_failed_revision_and_receipts_preserved': True,
        'new_complete_anonymous_readback_required': True}
    report_path = DEST / 'reports/v26-selected/publication-transport-recovery-v1.json'
    base.save(report_path, report)
    test_path = DEST / 'reports/v26-selected/cumulative-publisher-tests-v3.log'
    shutil.copyfile(FIRST / 'cumulative-publisher-tests-v3.log', test_path)
    additions = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                 for p in [report_path, test_path]]
    plan = {**old, 'parent_commit': failure['commit'], 'manifest': old['manifest'] + additions,
        'metadata_recovery_of': failure['commit'], 'metadata_only_write_paths': missing + [a['path'] for a in additions],
        'publication_transport_library_commit': code,
        'goal_phase': 'v26_minimum_ppl_release_metadata_repaired_and_fully_verified_continuing_to_10m'}
    base.save(STATE / 'summary.json', json.loads((ORIGINAL / 'summary.json').read_text()))
    base.save(STATE / 'publication-plan.json', plan)
    print(json.dumps({'files_in_repaired_manifest': len(plan['manifest']), 'metadata_writes': plan['metadata_only_write_paths'],
                      'old_verified_files_preserved': len(verified)}), flush=True)


def publish():
    if (STATE / 'publication-data-commit.json').exists(): raise ValueError('recovery already committed')
    plan = json.loads((STATE / 'publication-plan.json').read_text()); api = HfApi()
    if api.repo_info(transport.DATASET, repo_type='dataset').sha != plan['parent_commit']:
        raise ValueError('remote changed before metadata recovery')
    actual = set(HfApi(token=False).list_repo_files(transport.DATASET, repo_type='dataset', revision=plan['parent_commit']))
    expected = {f['path'] for f in plan['manifest']} | {'.gitattributes'}
    if actual - expected or expected - actual != set(plan['metadata_only_write_paths']):
        raise ValueError('current remote defect differs from the frozen recovery scope')
    by_path = {f['path']: f for f in plan['manifest']}; operations = []
    for path in plan['metadata_only_write_paths']:
        item = by_path[path]
        if base.file_sha(DEST / path) != item['sha256']: raise ValueError('planned recovery metadata changed')
        operations.append(CommitOperationAdd(path, DEST / path))
    result = api.create_commit(transport.DATASET, repo_type='dataset', operations=operations,
        parent_commit=plan['parent_commit'], commit_message='Restore two unchanged recipe files omitted by folder replacement and preserve failure evidence')
    base.save(STATE / 'publication-data-commit.json', {'commit': result.oid, 'parent': plan['parent_commit'],
        'plan_sha256': base.file_sha(STATE / 'publication-plan.json'), 'metadata_only_recovery': True})
    print(json.dumps({'repaired_data_commit': result.oid, 'training_payload_unchanged': True}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'publish', 'verify']); args = p.parse_args()
    with (FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare': prepare()
        elif args.phase == 'publish': publish()
        else:
            transport.STATE = STATE; transport.DEST = DEST; transport.FIRST = FIRST; transport.verify()
