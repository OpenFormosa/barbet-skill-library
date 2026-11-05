from pathlib import Path
import pytest
from barbet_pipeline.common import save_new, read_json, sha_file, digest
from barbet_pipeline.parallel_recovery import partition_missing
from barbet_pipeline.parallel_recovery_evidence import collect_recovery_history

CODE = Path(__file__).resolve().parents[1]


def fixture(gpu):
    net = gpu / 'resume-network-20261105'
    par = gpu / 'parallel-recovery-20261105'
    items = [{'remote': f'fixture-{i}', 'rows': 1, 'bytes': 10} for i in range(9)]
    original = {'assignments': [items[:2]] + [[i] for i in items[2:]], 'rows': 9, 'files': 9,
        'runner_sha256': sha_file(CODE / 'scripts/run_full_v4_gpu_quality.py'),
        'scorer_sha256': sha_file(CODE / 'barbet_pipeline/gpu_quality.py')}
    save_new(gpu / 'full-score-plan.json', original)
    plan_sha = sha_file(gpu / 'full-score-plan.json')
    save_new(net / 'prior-workers-exited.json', {'exit_codes': {str(i): int(i == 0) for i in range(8)}, 'all_success': False})
    (gpu / 'worker-0.log').write_text('Fixture: original connection failure\n')
    (net / 'worker-0.log').write_text('Fixture: controlled interruption\n')
    save_new(gpu / 'network-resume-preflight-20261105.json', {'fixture': True})
    inv = {'plan_sha256': plan_sha, 'runner_sha256': original['runner_sha256'],
        'scorer_sha256': original['scorer_sha256'], 'changed_model_or_scorer_or_assignments': False,
        'training_updates': 0, 'prior_exit_sha256': sha_file(net / 'prior-workers-exited.json'),
        'original_worker_log_sha256': sha_file(gpu / 'worker-0.log'),
        'recovered_input_receipt_sha256': sha_file(gpu / 'network-resume-preflight-20261105.json')}
    save_new(net / 'investigation.json', inv)
    save_new(net / 'launch.json', {'pid': 123, 'fixture': True})
    save_new(net / 'completion.json', {'pid': 123, 'exit_code': -15,
        'investigation_sha256': sha_file(net / 'investigation.json'),
        'prior_workers_exited_sha256': sha_file(net / 'prior-workers-exited.json'),
        'resumed_worker_log_sha256': sha_file(net / 'worker-0.log')})
    save_new(par / 'network-attempt-workers-exited.json', {'exit_codes': {str(i): -15 if i == 0 else 0 for i in range(8)},
        'all_success': False, 'resume_completion_sha256': sha_file(net / 'completion.json')})
    control = {'passed': True, 'rows': 32, 'original_scoring_core_ast_identical': True,
        'original_plan_sha256': plan_sha, 'scorer_sha256': original['scorer_sha256'],
        'recovery_runner_sha256': sha_file(CODE / 'scripts/run_full_v4_gpu_parallel_recovery.py')}
    save_new(gpu / 'parallel-recovery-controls-20261105/result.json', control)
    intent = {'controlled_interruption_pid': 123, 'expected_signal': 'SIGTERM',
        'old_network_launch_sha256': sha_file(net / 'launch.json'),
        'launcher_sha256': sha_file(CODE / 'scripts/launch_full_v4_parallel_recovery.py'),
        'real_v26_controls_sha256': sha_file(gpu / 'parallel-recovery-controls-20261105/result.json'),
        'training_updates': 0, 'no_model_or_scorer_or_per_file_batch_change': True}
    save_new(par / 'switch-intent.json', intent)
    retained = [items[0]] + items[2:]
    hashes = {}
    for item in retained:
        p = gpu / 'score-receipts' / (digest(item['remote']) + '.json')
        save_new(p, {'fixture': item['remote']});hashes[item['remote']] = sha_file(p)
    recovery = {'original_plan_sha256': plan_sha,
        'runner_sha256': control['recovery_runner_sha256'],
        'helper_sha256': sha_file(CODE / 'barbet_pipeline/parallel_recovery.py'),
        'controls_sha256': sha_file(gpu / 'parallel-recovery-controls-20261105/result.json'),
        'switch_intent_sha256': sha_file(par / 'switch-intent.json'),
        'missing_inputs': [items[1]], 'retained_inputs': retained, 'retained_receipt_sha256': hashes,
        'assignments': partition_missing([items[1]])}
    save_new(par / 'plan.json', recovery)
    recovery_sha = sha_file(par / 'plan.json')
    save_new(par / 'launch.json', {'pids': list(range(1000, 1008)), 'slots': list(range(8)), 'recovery_plan_sha256': recovery_sha})
    wanted = {str(i): 0 for i in range(8)}
    save_new(par / 'partition-exits.json', {'actual_process_exit_codes': wanted, 'all_success': True,
        'launch_sha256': sha_file(par / 'launch.json')})
    save_new(gpu / 'workers/0.json', {'complete': True, 'rows': 2})
    save_new(par / 'completion.json', {'complete': True, 'actual_process_exit_codes': wanted,
        'plan_sha256': recovery_sha, 'partition_exits_sha256': sha_file(par / 'partition-exits.json'),
        'logical_rank0_completion_sha256': sha_file(gpu / 'workers/0.json'),
        'retained_files': 8, 'retained_rows': 8, 'new_files': 1, 'new_rows': 1,
        'source_quality_or_target_changes': 0, 'training_updates': 0})
    save_new(gpu / 'workers-exited.json', {'exit_codes': wanted, 'all_success': True,
        'original_rank0_single_resume_exit_code': -15,
        'parallel_recovery_completion_sha256': sha_file(par / 'completion.json')})
    for slot, tasks in enumerate(recovery['assignments']):
        save_new(par / 'slots' / f'{slot}.json', {'slot': slot, 'complete': True, 'rows': len(tasks), 'recovery_plan_sha256': recovery_sha})
        for item in tasks:
            save_new(gpu / 'score-receipts' / (digest(item['remote']) + '.json'),
                {'gpu_rank': slot, 'original_logical_rank': 0, 'recovery_plan_sha256': recovery_sha,
                    'recovery_runner_sha256': recovery['runner_sha256']})
    return net, par


def rewrite(path, mutate):
    value = read_json(path);mutate(value);path.unlink();save_new(path, value)


def test_full_history_preserves_both_interruptions(tmp_path):
    fixture(tmp_path)
    network, parallel = collect_recovery_history(tmp_path, CODE)
    assert network['completion']['exit_code'] == -15
    assert parallel['partition_exits']['all_success'] and parallel['all_history_hashes_verified']


def test_old_log_cannot_be_silently_rewritten(tmp_path):
    net, par = fixture(tmp_path)
    (net / 'worker-0.log').write_text('erased interruption')
    with pytest.raises(ValueError, match='history hash'):
        collect_recovery_history(tmp_path, CODE)


def test_completed_logical_rank_does_not_hide_a_failed_process(tmp_path):
    net, par = fixture(tmp_path)
    rewrite(par / 'partition-exits.json', lambda v: v['actual_process_exit_codes'].update({'2': 1}))
    # Keep the file link valid so the negative control exercises actual exit semantics.
    rewrite(par / 'completion.json', lambda v: v.update(partition_exits_sha256=sha_file(par / 'partition-exits.json')))
    rewrite(tmp_path / 'workers-exited.json', lambda v: v.update(parallel_recovery_completion_sha256=sha_file(par / 'completion.json')))
    with pytest.raises(ValueError, match='Eight actual successful'):
        collect_recovery_history(tmp_path, CODE)


def test_retained_score_receipt_is_immutable(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'score-receipts' / (digest('fixture-0') + '.json')).write_text('{}')
    with pytest.raises(ValueError, match='retained successful'):
        collect_recovery_history(tmp_path, CODE)


def test_no_completion_receipt_means_no_completed_history(tmp_path):
    net, par = fixture(tmp_path)
    (par / 'completion.json').unlink()
    with pytest.raises(FileNotFoundError):
        collect_recovery_history(tmp_path, CODE)
