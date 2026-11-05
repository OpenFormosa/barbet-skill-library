"""Resume only the failed rank after the recorded anonymous-download failure.

The scorer, worker, original assignments, successful outputs and receipts remain
unchanged.  A new exit summary is written only after the real resumed process
exits; the failed exit summary and original logs remain auditable.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import time

from barbet_pipeline.common import read_json, save_new, sha_file, digest
from barbet_pipeline.gpu_release import validate_score_receipt

CODE = Path('/root/barbet-full-corpus-v4')
GPU = Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
RESUME = GPU / 'resume-network-20261105'


def main():
    plan = read_json(GPU / 'full-score-plan.json')
    original = read_json(GPU / 'workers-exited.json')
    if original['all_success'] or original['exit_codes'] != {str(i): int(i == 0) for i in range(8)}:
        raise ValueError('This recovery applies only to the recorded rank-0 failure')
    if sha_file(CODE / 'scripts/run_full_v4_gpu_quality.py') != plan['runner_sha256']:
        raise ValueError('Frozen worker changed')
    if sha_file(CODE / 'barbet_pipeline/gpu_quality.py') != plan['scorer_sha256']:
        raise ValueError('Frozen scorer changed')
    failed_log = (GPU / 'worker-0.log').read_text()
    if 'Anonymous curl transfer failed (exit 7)' not in failed_log:
        raise ValueError('Recorded failure is not the investigated connection error')
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True, timeout=60).strip()
    if active:
        raise RuntimeError('Preserve existing GPU processes; do not start a duplicate worker')

    missing = []
    existing_files = existing_rows = 0
    for rank, items in enumerate(plan['assignments']):
        for item in items:
            receipt_path = GPU / 'score-receipts' / (digest(item['remote']) + '.json')
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                validate_score_receipt(item, receipt, plan, GPU)
                existing_files += 1
                existing_rows += receipt['rows']
            else:
                if rank != 0:
                    raise ValueError('A successful rank is missing a score receipt')
                missing.append(item)
        worker_path = GPU / 'workers' / (str(rank) + '.json')
        if rank:
            worker = read_json(worker_path)
            if not worker['complete'] or worker['rows'] != sum(i['rows'] for i in items):
                raise ValueError('Successful rank completion does not cover its assignment')
            if worker['plan_sha256'] != sha_file(GPU / 'full-score-plan.json'):
                raise ValueError('Successful rank used a different plan')
        elif worker_path.exists():
            raise ValueError('Rank 0 already has a completion receipt; investigate first')
    if len(missing) != 155 or sum(i['rows'] for i in missing) != 551276:
        raise ValueError('Missing workload differs from the investigated failure')
    preflight = read_json(GPU / 'network-resume-preflight-20261105.json')
    first = missing[0]
    cached = GPU / 'scratch/0' / (digest(first['remote']) + '.parquet')
    if (preflight['input_path'] != first['remote'] or preflight['sha256'] != first['sha256']
            or sha_file(cached) != first['sha256']):
        raise ValueError('The failed immutable input has not been recovered byte-for-byte')

    RESUME.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    investigation = {
        'status': 'INVESTIGATED_NETWORK_FAILURE_RESUME_NOT_COMPLETION',
        'started_at_utc': started,
        'reason': 'Rank 0 exhausted bounded anonymous curl retries with exit 7 before scoring its next file.',
        'rank': 0, 'existing_scored_files': existing_files,
        'existing_scored_rows': existing_rows, 'missing_files': len(missing),
        'missing_rows': sum(i['rows'] for i in missing),
        'plan_sha256': sha_file(GPU / 'full-score-plan.json'),
        'runner_sha256': plan['runner_sha256'], 'scorer_sha256': plan['scorer_sha256'],
        'resume_implementation_sha256': sha_file(__file__),
        'prior_exit_sha256': sha_file(GPU / 'workers-exited.json'),
        'original_worker_log_sha256': sha_file(GPU / 'worker-0.log'),
        'recovered_input_receipt_sha256': sha_file(GPU / 'network-resume-preflight-20261105.json'),
        'successful_scores_hash_verified_before_resume': True,
        'changed_model_or_scorer_or_assignments': False,
        'training_updates': 0,
    }
    save_new(RESUME / 'investigation.json', investigation)
    (GPU / 'workers-exited.json').rename(RESUME / 'prior-workers-exited.json')
    env = dict(os.environ, PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    with (RESUME / 'worker-0.log').open('x') as log:
        process = subprocess.Popen(
            ['/root/Barbet/.venv/bin/python', '-u', 'scripts/run_full_v4_gpu_quality.py', '--rank', '0'],
            cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = {'pid': process.pid, 'rank': 0, 'started_at_utc': started,
            'plan_sha256': investigation['plan_sha256'], 'training_updates': 0}
        save_new(RESUME / 'launch.json', launch)
        print(json.dumps({'status': 'resumed_rank_0', **launch}), flush=True)
        code = process.wait()
    codes = dict(original['exit_codes'])
    codes['0'] = code
    if code == 0:
        worker = read_json(GPU / 'workers/0.json')
        if (not worker['complete'] or worker['rows'] != sum(i['rows'] for i in plan['assignments'][0])
                or worker['plan_sha256'] != investigation['plan_sha256']):
            raise ValueError('Actual rank-0 completion does not cover its original assignment')
    result = {'rank': 0, 'pid': process.pid, 'exit_code': code,
        'elapsed_seconds': time.monotonic() - start,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'investigation_sha256': sha_file(RESUME / 'investigation.json'),
        'prior_workers_exited_sha256': sha_file(RESUME / 'prior-workers-exited.json'),
        'resumed_worker_log_sha256': sha_file(RESUME / 'worker-0.log'),
        'scope': 'Only the failed rank resumed; seven successful rank completions were retained.'}
    save_new(RESUME / 'completion.json', result)
    save_new(GPU / 'workers-exited.json', {'exit_codes': codes,
        'all_success': all(v == 0 for v in codes.values()),
        'resume_completion_sha256': sha_file(RESUME / 'completion.json'),
        'prior_failure_preserved': True})
    print(json.dumps(result), flush=True)
    if code:
        raise RuntimeError('Resumed worker failed; preserve both attempts and investigate')


if __name__ == '__main__':
    main()
