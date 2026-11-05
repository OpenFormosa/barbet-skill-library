"""Disclose and perform the controlled switch to eight unfinished-file workers."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import subprocess
import time
import psutil
from barbet_pipeline.common import read_json, save_new, sha_file, digest
from barbet_pipeline.gpu_release import validate_score_receipt
from barbet_pipeline.parallel_recovery import partition_missing, verify_recovery_contract, writer_core

CODE = Path('/root/barbet-full-corpus-v4')
GPU = Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')
OUTPUT = GPU / 'parallel-recovery-20261105'


def main():
    original = read_json(GPU / 'full-score-plan.json')
    control_path = GPU / 'parallel-recovery-controls-20261105/result.json'
    controls = read_json(control_path)
    if (not controls['passed'] or controls['rows'] != 32
            or controls['original_plan_sha256'] != sha_file(GPU / 'full-score-plan.json')
            or controls['recovery_runner_sha256'] != sha_file(CODE / 'scripts/run_full_v4_gpu_parallel_recovery.py')
            or controls['scorer_sha256'] != original['scorer_sha256']):
        raise ValueError('Matching real-v26 recovery controls must pass before the switch')
    if writer_core(CODE / 'scripts/run_full_v4_gpu_quality.py') != writer_core(CODE / 'scripts/run_full_v4_gpu_parallel_recovery.py'):
        raise ValueError('Original scoring loop changed')
    if (GPU / 'workers/0.json').exists() or (GPU / 'workers-exited.json').exists():
        raise ValueError('Investigate the current worker state before a controlled switch')
    network = GPU / 'resume-network-20261105'
    old_launch = read_json(network / 'launch.json')
    owned = psutil.Process(old_launch['pid'])
    expected = ['/root/Barbet/.venv/bin/python', '-u', 'scripts/run_full_v4_gpu_quality.py', '--rank', '0']
    if owned.cmdline() != expected:
        raise ValueError('The process is not the owned original rank-0 continuation')
    OUTPUT.mkdir(exist_ok=False)
    intent = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Seven local GPUs are idle after completing their original assignments; partition only unfinished rank-0 files across eight GPUs.',
        'controlled_interruption_pid': owned.pid, 'actual_cmdline': owned.cmdline(),
        'old_network_launch_sha256': sha_file(network / 'launch.json'),
        'original_plan_sha256': sha_file(GPU / 'full-score-plan.json'),
        'real_v26_controls_sha256': sha_file(control_path),
        'launcher_sha256': sha_file(__file__), 'expected_signal': 'SIGTERM',
        'successful_receipts_preserved': True, 'partial_outputs_archived_before_rescoring': True,
        'no_model_or_scorer_or_per_file_batch_change': True, 'training_updates': 0}
    save_new(OUTPUT / 'switch-intent.json', intent)
    os.kill(owned.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while not (network / 'completion.json').exists() or not (GPU / 'workers-exited.json').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned supervisor has not recorded the controlled interruption; preserve state')
        time.sleep(.2)
    stopped = read_json(network / 'completion.json')
    stopped_exits = read_json(GPU / 'workers-exited.json')
    if stopped['pid'] != owned.pid or stopped['exit_code'] != -signal.SIGTERM or stopped_exits['all_success']:
        raise ValueError('Actual termination differs from the recorded controlled switch')
    (GPU / 'workers-exited.json').rename(OUTPUT / 'network-attempt-workers-exited.json')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=60).strip()
    if active:
        raise RuntimeError('Another GPU process is active; preserve it and investigate before launch')
    retained, pending, retained_hashes = [], [], {}
    for rank, items in enumerate(original['assignments']):
        for item in items:
            path = GPU / 'score-receipts' / (digest(item['remote']) + '.json')
            if path.exists():
                validate_score_receipt(item, read_json(path), original, GPU)
                retained.append(item);retained_hashes[item['remote']] = sha_file(path)
            else:
                if rank != 0:
                    raise ValueError('A successful original rank lost a receipt')
                pending.append(item)
    recovery = {'status': 'EXACT_UNFINISHED_FILE_PARTITION_NOT_NEW_SCORER',
        'original_plan_sha256': sha_file(GPU / 'full-score-plan.json'),
        'runner_sha256': sha_file(CODE / 'scripts/run_full_v4_gpu_parallel_recovery.py'),
        'helper_sha256': sha_file(CODE / 'barbet_pipeline/parallel_recovery.py'),
        'controls_sha256': sha_file(control_path), 'switch_intent_sha256': sha_file(OUTPUT / 'switch-intent.json'),
        'missing_inputs': pending, 'retained_inputs': retained,
        'retained_receipt_sha256': retained_hashes, 'assignments': partition_missing(pending),
        'original_logical_rank': 0, 'physical_gpu_slots': list(range(8)), 'max_gpus': 8,
        'original_scoring_core_ast_identical': True, 'per_file_batch_size': 4,
        'per_file_max_padded_tokens': 16384, 'no_truncation': True, 'training_updates': 0}
    verify_recovery_contract(CODE, GPU, recovery)
    save_new(OUTPUT / 'plan.json', recovery)
    handles, children = [], []
    start = time.monotonic()
    for slot in range(8):
        handle = (OUTPUT / f'worker-{slot}.log').open('x');handles.append(handle)
        env = dict(os.environ, PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES=str(slot), TOKENIZERS_PARALLELISM='false')
        child = subprocess.Popen(['/root/Barbet/.venv/bin/python', '-u', 'scripts/run_full_v4_gpu_parallel_recovery.py', '--slot', str(slot)],
            cwd=CODE, env=env, stdout=handle, stderr=subprocess.STDOUT)
        children.append(child)
    save_new(OUTPUT / 'launch.json', {'pids': [p.pid for p in children], 'slots': list(range(8)),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'recovery_plan_sha256': sha_file(OUTPUT / 'plan.json')})
    print(json.dumps({'launched': 8, 'pids': [p.pid for p in children],
        'remaining_files': len(pending), 'remaining_rows': sum(i['rows'] for i in pending)}), flush=True)
    exits = {}
    while len(exits) < 8:
        for slot, child in enumerate(children):
            code = child.poll()
            if code is not None and str(slot) not in exits:
                exits[str(slot)] = code
                print(json.dumps({'slot_finished': slot, 'actual_process_exit_code': code}), flush=True)
        if len(exits) < 8:time.sleep(2)
    for handle in handles:handle.close()
    save_new(OUTPUT / 'partition-exits.json', {'actual_process_exit_codes': exits,
        'all_success': all(code == 0 for code in exits.values()), 'launch_sha256': sha_file(OUTPUT / 'launch.json')})
    if not all(code == 0 for code in exits.values()):
        save_new(GPU / 'workers-exited.json', {'exit_codes': {str(i): int(i == 0) for i in range(8)},
            'all_success': False, 'failure_evidence': str(OUTPUT / 'partition-exits.json')})
        raise RuntimeError('Parallel recovery has a failed process; retain every successful receipt')
    for slot in range(8):
        part = read_json(OUTPUT / 'slots' / f'{slot}.json')
        assert part['complete'] and part['rows'] == sum(i['rows'] for i in recovery['assignments'][slot])
        assert part['recovery_plan_sha256'] == sha_file(OUTPUT / 'plan.json')
    for assignment in original['assignments']:
        for item in assignment:
            path = GPU / 'score-receipts' / (digest(item['remote']) + '.json')
            validate_score_receipt(item, read_json(path), original, GPU)
    for slot, items in enumerate(recovery['assignments']):
        for item in items:
            receipt = read_json(GPU / 'score-receipts' / (digest(item['remote']) + '.json'))
            assert receipt['gpu_rank'] == slot and receipt['original_logical_rank'] == 0
            assert receipt['recovery_plan_sha256'] == sha_file(OUTPUT / 'plan.json')
            assert receipt['recovery_runner_sha256'] == recovery['runner_sha256']
    seconds = time.monotonic() - start
    save_new(GPU / 'workers/0.json', {'complete': True,
        'rows': sum(i['rows'] for i in original['assignments'][0]), 'seconds': seconds,
        'plan_sha256': sha_file(GPU / 'full-score-plan.json'),
        'execution': 'Original logical assignment completed by eight real recovery processes plus preserved original receipts.',
        'recovery_plan_sha256': sha_file(OUTPUT / 'plan.json')})
    result = {'complete': True, 'actual_process_exit_codes': exits, 'elapsed_seconds': seconds,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha_file(OUTPUT / 'plan.json'), 'partition_exits_sha256': sha_file(OUTPUT / 'partition-exits.json'),
        'retained_files': len(retained), 'retained_rows': sum(i['rows'] for i in retained),
        'new_files': len(pending), 'new_rows': sum(i['rows'] for i in pending),
        'logical_rank0_completion_sha256': sha_file(GPU / 'workers/0.json'),
        'source_quality_or_target_changes': 0, 'training_updates': 0}
    save_new(OUTPUT / 'completion.json', result)
    save_new(GPU / 'workers-exited.json', {'exit_codes': {str(i): 0 for i in range(8)}, 'all_success': True,
        'exit_codes_scope': 'Original logical assignments; rank 0 completed by the actual eight successful recovery processes.',
        'original_rank0_single_resume_exit_code': stopped['exit_code'], 'prior_failure_preserved': True,
        'parallel_recovery_completion_sha256': sha_file(OUTPUT / 'completion.json')})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
