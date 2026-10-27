#!/usr/bin/env python3
"""Run frozen final jobs on explicitly assigned GPUs, keeping resumable logs."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    seal = json.loads((HERE / 'FINAL_SELECTION.json').read_text())
    schedule_path = HERE / 'work/final_job_schedule.json'
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text())
    else:
        jobs = []
        for family, policy in seal['comparisons']:
            cost = (6 if family == 'qwen' else 3) * (2.4 if policy == 'long' else 1.3 if policy == 'self_review' else 1.)
            for seed in seal['seeds']:
                jobs.append({'model': family, 'policy': policy, 'seed': seed, 'weight': cost, 'status': 'pending'})
        loads = [0.] * 8
        for job in sorted(jobs, key=lambda j: (-j['weight'], j['model'], j['policy'], j['seed'])):
            gpu = min(range(8), key=lambda i: (loads[i], i))
            job['gpu'] = gpu
            loads[gpu] += job['weight']
        schedule = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'jobs': jobs,
            'scheduling_notice': 'Static greedy balance by declared heuristic weights; not a measured latency prediction.'}
        schedule_path.write_text(json.dumps(schedule, indent=2) + '\n')
    expected = {(m, p, s) for m, p in seal['comparisons'] for s in seal['seeds']}
    if {(j['model'], j['policy'], j['seed']) for j in schedule['jobs']} != expected:
        raise RuntimeError('schedule differs from sealed comparisons')
    lock = threading.Lock()
    def save():
        temporary = schedule_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(schedule, indent=2) + '\n')
        temporary.replace(schedule_path)
    def worker(gpu):
        jobs = sorted([j for j in schedule['jobs'] if j['gpu'] == gpu], key=lambda j: -j['weight'])
        for job in jobs:
            if job['status'] == 'completed':
                continue
            log = HERE / 'work/logs' / f'final-{job["model"]}-{job["policy"]}-{job["seed"]}.log'
            command = [sys.executable, str(HERE / 'run.py'), '--model', job['model'], '--split', 'test',
                '--policies', job['policy'], '--config', str(HERE / 'final_policies.json'),
                '--seed', str(job['seed']), '--run-tag', 'final', '--batch-size', '128', '--max-seqs', '128', '--eager']
            with lock:
                job.update(status='running', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), command=command, log=str(log.relative_to(HERE)))
                save()
                print(json.dumps({'event': 'launched', 'gpu': gpu, 'model': job['model'], 'policy': job['policy'], 'seed': job['seed']}), flush=True)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
            with log.open('a') as stream:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with lock:
                job.update(status='completed' if process.returncode == 0 else 'failed', returncode=process.returncode,
                    finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                save()
                print(json.dumps({'event': job['status'], 'gpu': gpu, 'model': job['model'], 'policy': job['policy'], 'seed': job['seed']}), flush=True)
            if process.returncode:
                raise RuntimeError('final job failed; inspect ' + str(log))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(worker, range(8)):
            pass


if __name__ == '__main__':
    main()
