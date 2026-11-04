"""Finish the authorized release only after all CPU/GPU evidence really exists."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from barbet_pipeline.common import read_json, save_new, sha_file

CODE = Path('/root/barbet-full-corpus-v4')
STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
GPU = STATE / 'gpu-quality'
PYTHON = '/root/barbet-context-sft-data/.venv/bin/python'


def run(script, log):
    env = dict(os.environ, PYTHONPATH=str(CODE))
    with (STATE / log).open('x') as output:
        return subprocess.run([PYTHON, '-u', str(CODE / 'scripts' / script)], cwd=CODE,
            env=env, stdout=output, stderr=subprocess.STDOUT).returncode


def main():
    started = datetime.now(timezone.utc).isoformat()
    print(json.dumps({'status': 'waiting_for_complete_gpu_census', 'started_at_utc': started}), flush=True)
    while not (GPU / 'complete-summary.json').exists():
        if (GPU / 'workers-exited.json').exists() and not read_json(GPU / 'workers-exited.json')['all_success']:
            raise RuntimeError('GPU worker failure requires investigation; no final publication attempted')
        time.sleep(30)
    from barbet_pipeline.full_release import collect_complete_release
    from barbet_pipeline.gpu_release import collect_complete_gpu_release
    summary, stats, files = collect_complete_release(STATE)
    gpu, measured = collect_complete_gpu_release(STATE, files)
    print(json.dumps({'status': 'all_cpu_and_gpu_receipts_verified', 'rows': gpu['totals']['rows']}), flush=True)
    package = STATE / 'release-package'
    if package.exists():
        # Never discard a partial package or change its already frozen contents.
        manifest = read_json(package / 'release-manifest.json')
        assert manifest['all_exported_rows_gpu_measured'] and manifest['gpu_measurement_rows'] == gpu['totals']['rows']
        for item in read_json(package / 'upload-plan.json')['items']:
            assert sha_file(item['local']) == item['sha256']
    elif run('package_full_v4_release.py', 'final-package.log') != 0:
        raise RuntimeError('Release packaging failed; evidence retained in final-package.log')
    receipt_path = STATE / 'publication/final/final-receipt.json'
    for attempt in range(1, 4):
        if receipt_path.exists():
            break
        log = 'final-publish-attempt-' + str(attempt) + '.log'
        result = run('publish_full_v4_release.py', log)
        if result == 0:
            break
        error = (STATE / log).read_text()[-12000:]
        transient = any(x in error.lower() for x in ['502', '503', '504', '500 server error', '429', 'readtimeout', 'connectionerror', 'connection reset', 'timed out'])
        if not transient or attempt == 3:
            raise RuntimeError('Publication failed; no completion claim: ' + log)
        print(json.dumps({'status': 'retrying_transient_publication_failure', 'attempt': attempt, 'log': log}), flush=True)
        time.sleep(30)
    receipt = read_json(receipt_path)
    assert receipt['exported_rows'] == gpu['totals']['rows']
    docs = CODE / 'docs/reports/full-corpus-v4'
    outputs = [(package / 'report.html', docs / 'release-complete.html'),
               (package / 'gpu-report.html', docs / 'gpu-complete.html'),
               (package / 'publication-complete.json', docs / 'publication-complete.json')]
    for source, target in outputs:
        if target.exists():
            assert sha_file(target) == sha_file(source)
        else:
            shutil.copyfile(source, target)
    paths = [str(t.relative_to(CODE)) for _, t in outputs]
    subprocess.run(['git', 'add', '--', *paths], cwd=CODE, check=True)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=CODE, text=True).splitlines()
    if staged:
        if set(staged) != set(paths):
            raise RuntimeError('Unrelated staged work appeared; preserve it without committing')
        subprocess.run(['git', 'commit', '-m', 'Publish completed full corpus and exhaustive frozen-v26 quality evidence'], cwd=CODE, check=True)
    for attempt in range(3):
        result = subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'feat/full-corpus-reprocess-v4'], cwd=CODE)
        if result.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError('HF publication is complete, but GitHub report push failed; local reports retained')
        time.sleep(30)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    result = {'status': 'FULL_CORPUS_GPU_AND_PUBLICATION_COMPLETE', 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'started_at_utc': started, 'source_rows': summary['counts']['active_rows_processed'],
        'exported_and_gpu_scored_rows': gpu['totals']['rows'], 'quarantined_rows': summary['counts']['quarantined'],
        'hf_final_revision': receipt['final_revision'], 'hf_branch': receipt['branch'], 'github_commit': commit,
        'cpu_summary_sha256': sha_file(STATE / 'processing-summary.json'),
        'gpu_summary_sha256': sha_file(GPU / 'complete-summary.json'),
        'hf_final_receipt_sha256': sha_file(receipt_path), 'model_training': 'NOT_RUN',
        'learnability_transfer_retention': 'NOT_ESTABLISHED_BY_PROXY_MEASUREMENTS'}
    save_new(STATE / 'pipeline-complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
