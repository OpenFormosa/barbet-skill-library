"""Frozen native-v26 scoring of all eligible OpenCode answer alternatives."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from scripts import rank_natural_instructions_v26_v1 as structural
from scripts.inventory_opencode_v1 import STATE as SOURCE

STATE = SOURCE / 'v26-selection-v1'
CALIBRATION = base.WORK / 'reports/memory-harness-v2-v26-selection'


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'candidate-input-files.json').exists(): raise ValueError('OpenCode scoring manifest already frozen')
    summary = json.loads((SOURCE / 'normalize-summary.json').read_text())
    policy = json.loads((SOURCE / 'full-normalization-policy-v1.json').read_text())
    if summary['full_normalization_policy_sha256'] != base.file_sha(SOURCE / 'full-normalization-policy-v1.json'):
        raise ValueError('normalization policy changed')
    for key, script in [('driver_sha256', 'normalize_opencode_memory_v1.py'), ('adapter_sha256', 'opencode_memory_adapter_v1.py')]:
        if policy[key] != base.file_sha(base.REPO / 'scripts' / script): raise ValueError('source transformation changed')
    files = []; rows = 0
    for spec in json.loads((SOURCE / 'full-normalize-jobs-v1.json').read_text()):
        report = json.loads((SOURCE / 'normalized' / ('full-' + spec['job_id'] + '.report.json')).read_text())
        if report['policy_sha256'] != summary['full_normalization_policy_sha256'] or base.file_sha(report['path']) != report['sha256']:
            raise ValueError('candidate shard binding differs')
        rows += report['counts'].get('candidate_answers', 0)
        if report['counts'].get('candidate_answers', 0):
            files.append({'file_id': len(files), 'path': report['path'], 'sha256': report['sha256'], 'wave': 'opencode-contextual-memory-v1'})
    if rows != summary['counts']['candidate_answers']: raise ValueError('candidate coverage differs')
    audit = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    base.save(STATE / 'candidate-holds.private.json', {'record_ids': audit['excluded_record_ids'],
        'query_hashes': policy['held_query_hashes'], 'search_allowed': [],
        'supplemental_prior_queries': policy['unequal_condition_alias_query_hashes']})
    for name in ['canonical-single-v3-validation.json', 'live-calibration-diagnostic-v3.json', 'remote-v26-tag-check.json']:
        shutil.copyfile(CALIBRATION / name, STATE / name)
    base.save(STATE / 'candidate-input-files.json', files)
    base.save(STATE / 'candidate-policy-binding.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'policy_sha256': base.file_sha(candidates.POLICY), 'normalization_summary_sha256': base.file_sha(SOURCE / 'normalize-summary.json'),
        'first_source_audit_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'structural_guard_driver_sha256': base.file_sha(base.REPO / 'scripts/rank_natural_instructions_v26_v1.py'),
        'all_first_holds_preserved': True, 'native_v26_minimum_conditional_answer_ppl': True,
        'source_test_metadata_not_in_prompt': True, 'source_programs_executed': False, 'weights_trained': False})
    print(json.dumps({'files': len(files), 'candidate_answers': rows}), flush=True)


def configure():
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    if binding['driver_sha256'] != base.file_sha(Path(__file__)):
        raise ValueError('frozen scoring driver changed')
    if binding['structural_guard_driver_sha256'] != base.file_sha(base.REPO / 'scripts/rank_natural_instructions_v26_v1.py'):
        raise ValueError('structural proof code changed')
    candidates.STATE = STATE; structural.STATE = STATE


def score_all():
    if json.loads((STATE / 'episode-identity-v1/summary.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('all-row structural guard required')
    folder = STATE / 'scores/canonical-v2'; folder.mkdir(parents=True, exist_ok=True)
    if (STATE / 'score-launch-v1.json').exists(): raise ValueError('scoring launch already recorded')
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    children = []; handles = []
    for bucket in range(8):
        handle = (folder / f'bucket-{bucket}.log').open('w'); handles.append(handle)
        command = ['/root/Barbet/.venv/bin/python', str(Path(__file__).resolve()), 'score', '--bucket', str(bucket)]
        proc = subprocess.Popen(command, cwd=base.REPO, env={**env, 'CUDA_VISIBLE_DEVICES': str(bucket)}, stdout=handle, stderr=subprocess.STDOUT)
        children.append((bucket, proc))
    base.save(STATE / 'score-launch-v1.json', {'pids': [{'bucket': b, 'pid': p.pid} for b, p in children],
        'native_unpadded_batch_size': 1, 'max_gpu_workers': 8, 'weights_trained': False})
    failed = []
    for bucket, process in children:
        code = process.wait(); print(json.dumps({'bucket': bucket, 'exit_code': code}), flush=True)
        if code: failed.append(bucket)
    for handle in handles: handle.close()
    if failed: raise RuntimeError('incomplete scoring buckets: ' + str(failed))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'index', 'guard', 'score-all', 'score'])
    p.add_argument('--workers', type=int, default=4); p.add_argument('--bucket', type=int); args = p.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('index worker bound')
    if args.phase == 'prepare': prepare()
    else:
        configure()
        if args.phase == 'index':
            if (STATE / 'candidate-index-summary.json').exists(): raise ValueError('index already frozen')
            candidates.index(args.workers)
        elif args.phase == 'guard': structural.guard()
        elif args.phase == 'score-all': score_all()
        else:
            if args.bucket is None: raise ValueError('score bucket required')
            from scripts import score_v26_answer_candidates as scorer
            scorer.STATE = STATE; scorer.main(args.bucket, 0, 'canonical-v2')
