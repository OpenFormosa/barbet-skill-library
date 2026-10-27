"""Rank the frozen NI reference alternatives with the calibrated v26 scorer.

No answers are generated here. Candidate normalization and the numerical scorer
remain immutable; the new manifest supplies a separate scoring protocol binding.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts import prepare_v26_answer_candidates as candidates

STATE = ni.STATE / 'v26-selection-v1'
ORIGINAL = base.WORK / 'reports/memory-harness-v2-v26-selection'


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'candidate-input-files.json').exists():
        raise ValueError('NI ranking preparation already frozen')
    first = json.loads((ni.STATE / 'preparation-summary.json').read_text())
    for key, path in [('adapter_sha256', base.REPO / 'scripts/natural_instructions_memory_adapter.py'),
                      ('builder_sha256', base.REPO / 'scripts/build_natural_instructions_memory_v1.py')]:
        if first[key] != base.file_sha(path): raise ValueError('source transformation changed')
    full = json.loads((ni.STATE / 'normalize-summary.json').read_text())
    policy_path = ni.STATE / 'full-normalization-policy-v2.json'
    policy = json.loads(policy_path.read_text())
    if full['full_normalization_policy_sha256'] != base.file_sha(policy_path):
        raise ValueError('full normalization policy mismatch')
    if policy['script_sha256'] != base.file_sha(base.REPO / 'scripts/normalize_natural_instructions_memory_v2.py'):
        raise ValueError('full normalization code changed')
    files = []; rows = 0
    for spec in json.loads((ni.STATE / 'full-normalize-jobs-v2.json').read_text()):
        report = json.loads((ni.STATE / 'normalized' / (spec['job_id'] + '.report.json')).read_text())
        if base.file_sha(report['path']) != report['sha256']:
            raise ValueError('full candidate bytes changed')
        rows += report['counts'].get('candidate_answers', 0)
        if report['counts'].get('candidate_answers', 0):
            files.append({'file_id': len(files), 'path': report['path'], 'sha256': report['sha256'],
                          'wave': 'natural-instructions-memory-v2'})
    if rows != full['counts']['candidate_answers']: raise ValueError('candidate file coverage differs')
    audit = json.loads((ni.STATE / 'semantic-audit-v1/summary.json').read_text())
    base.save(STATE / 'candidate-holds.private.json', {
        'record_ids': audit['excluded_record_ids'], 'query_hashes': policy['held_query_hashes'],
        'search_allowed': [], 'supplemental_prior_queries': []})
    for name in ['canonical-single-v3-validation.json', 'live-calibration-diagnostic-v3.json',
                 'remote-v26-tag-check.json']:
        shutil.copyfile(ORIGINAL / name, STATE / name)
    base.save(STATE / 'candidate-input-files.json', files)
    base.save(STATE / 'candidate-policy-binding.json', {
        'policy_sha256': base.file_sha(candidates.POLICY),
        'full_normalization_policy_sha256': base.file_sha(policy_path),
        'full_normalization_summary_sha256': base.file_sha(ni.STATE / 'normalize-summary.json'),
        'first_source_diagnosis_sha256': base.file_sha(ni.STATE / 'semantic-audit-v1/summary.json'),
        'driver_sha256': base.file_sha(Path(__file__)), 'all_first_holds_preserved': True,
        'one_memory_placement_per_task': True, 'minimum_eligible_native_v26_answer_ppl': True,
        'accepted_training_rows': 0, 'no_model_training': True})
    print(json.dumps({'files': len(files), 'candidate_answers': rows}), flush=True)


def configure():
    candidates.STATE = STATE
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    if binding['driver_sha256'] != base.file_sha(Path(__file__)):
        raise ValueError('frozen NI ranking driver changed')


def guard():
    """Verify the structural special case rather than assume it from the source.

    Every eligible NI row has exactly one final assistant, no tool messages and
    no unsupervised assistants. Therefore no dialogue-prefix or observation-view
    alternative exists. Hash-bound full bytes are checked again before scoring.
    """
    folder = STATE / 'episode-identity-v1'; folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists(): raise ValueError('identity guard already frozen')
    candidates.init(); seen = 0
    for item in json.loads((STATE / 'candidate-input-files.json').read_text()):
        if base.file_sha(item['path']) != item['sha256']: raise ValueError('candidate bytes changed')
        for line in Path(item['path']).open():
            row = json.loads(line); candidates.eligible(row)
            messages = row['messages']
            if row['tools'] or messages[-1]['role'] != 'assistant' or not messages[-1].get('loss', True):
                raise ValueError('NI is no longer a one-target no-tool condition')
            if any(m['role'] not in ['user', 'system'] for m in messages[:-1]):
                raise ValueError('unexpected prefix assistant or observation')
            seen += 1
    index = json.loads((STATE / 'candidate-index-summary.json').read_text())
    if seen != index['counts']['eligible']: raise ValueError('identity coverage mismatch')
    db = sqlite3.connect(folder / 'identity.sqlite')
    db.execute('CREATE TABLE holds(candidate_identity TEXT PRIMARY KEY,reason TEXT)'); db.close()
    base.save(folder / 'policy.json', {'version': 'single-final-assistant-no-observation-v1',
        'all_manifest_rows_checked': True, 'source_prefixes_cannot_be_episodes': True,
        'driver_sha256': base.file_sha(Path(__file__))})
    base.save(folder / 'summary.json', {'status': 'COMPLETE', 'counts': {},
        'rows_checked': seen, 'distinct_candidate_answers': index['distinct_answers'],
        'all_original_candidates_covered': True, 'ambiguous_static_task_inputs': 0,
        'original_scores_not_rewritten': True})
    print(json.dumps({'identity_checked_rows': seen, 'holds': 0}), flush=True)


def score_all():
    if json.loads((STATE / 'episode-identity-v1/summary.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('complete identity coverage required')
    folder = STATE / 'scores/canonical-v2'; folder.mkdir(parents=True, exist_ok=True)
    if (STATE / 'score-launch-v1.json').exists(): raise ValueError('launch already recorded; inspect individual buckets')
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    children = []; handles = []
    for bucket in range(8):
        handle = (folder / f'bucket-{bucket}.log').open('w'); handles.append(handle)
        command = ['/root/Barbet/.venv/bin/python', str(Path(__file__).resolve()), 'score', '--bucket', str(bucket)]
        proc = subprocess.Popen(command, cwd=base.REPO, env={**env, 'CUDA_VISIBLE_DEVICES': str(bucket)},
                                stdout=handle, stderr=subprocess.STDOUT)
        children.append((bucket, proc))
    base.save(STATE / 'score-launch-v1.json', {'pids': [{'bucket': b, 'pid': p.pid} for b, p in children],
        'native_unpadded_batch_size': 1, 'max_gpu_workers': 8, 'weights_trained': False})
    failed = []
    for bucket, proc in children:
        code = proc.wait()
        print(json.dumps({'bucket': bucket, 'exit_code': code}), flush=True)
        if code: failed.append(bucket)
    for handle in handles: handle.close()
    if failed: raise RuntimeError('scoring workers failed: ' + str(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'index', 'guard', 'score', 'score-all'])
    parser.add_argument('--workers', type=int, default=8); parser.add_argument('--bucket', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('index worker bound')
    if args.phase == 'prepare': prepare()
    else:
        configure()
        if args.phase == 'index':
            if (STATE / 'candidate-index-summary.json').exists(): raise ValueError('index already frozen')
            candidates.index(args.workers)
        elif args.phase == 'guard': guard()
        elif args.phase == 'score-all': score_all()
        else:
            if args.bucket is None: raise ValueError('bucket is required')
            from scripts import score_v26_answer_candidates as scorer
            scorer.STATE = STATE; scorer.main(args.bucket, 0, 'canonical-v2')
