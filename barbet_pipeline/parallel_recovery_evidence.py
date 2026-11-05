"""Publication evidence for the disclosed network and parallel recovery attempts."""
from pathlib import Path
from .common import read_json, sha_file, digest
from .parallel_recovery import verify_recovery_contract
from .resume_evidence import collect_network_resume_evidence


def collect_recovery_history(gpu, code):
    gpu, code = Path(gpu), Path(code)
    parallel = gpu / 'parallel-recovery-20261105'
    if not (parallel / 'plan.json').exists():
        return collect_network_resume_evidence(gpu), None
    recovery = read_json(parallel / 'plan.json')
    original = verify_recovery_contract(code, gpu, recovery)
    network = gpu / 'resume-network-20261105'
    inv = read_json(network / 'investigation.json')
    prior = read_json(network / 'prior-workers-exited.json')
    net_launch = read_json(network / 'launch.json')
    net_complete = read_json(network / 'completion.json')
    net_exits = read_json(parallel / 'network-attempt-workers-exited.json')
    intent = read_json(parallel / 'switch-intent.json')
    launch = read_json(parallel / 'launch.json')
    exits = read_json(parallel / 'partition-exits.json')
    completed = read_json(parallel / 'completion.json')
    final = read_json(gpu / 'workers-exited.json')
    controls = read_json(gpu / 'parallel-recovery-controls-20261105/result.json')
    pairs = [
        (inv['prior_exit_sha256'], network / 'prior-workers-exited.json'),
        (inv['original_worker_log_sha256'], gpu / 'worker-0.log'),
        (inv['recovered_input_receipt_sha256'], gpu / 'network-resume-preflight-20261105.json'),
        (net_complete['investigation_sha256'], network / 'investigation.json'),
        (net_complete['prior_workers_exited_sha256'], network / 'prior-workers-exited.json'),
        (net_complete['resumed_worker_log_sha256'], network / 'worker-0.log'),
        (net_exits['resume_completion_sha256'], network / 'completion.json'),
        (intent['old_network_launch_sha256'], network / 'launch.json'),
        (recovery['switch_intent_sha256'], parallel / 'switch-intent.json'),
        (recovery['helper_sha256'], code / 'barbet_pipeline/parallel_recovery.py'),
        (intent['launcher_sha256'], code / 'scripts/launch_full_v4_parallel_recovery.py'),
        (intent['real_v26_controls_sha256'], gpu / 'parallel-recovery-controls-20261105/result.json'),
        (recovery['controls_sha256'], gpu / 'parallel-recovery-controls-20261105/result.json'),
        (launch['recovery_plan_sha256'], parallel / 'plan.json'),
        (exits['launch_sha256'], parallel / 'launch.json'),
        (completed['plan_sha256'], parallel / 'plan.json'),
        (completed['partition_exits_sha256'], parallel / 'partition-exits.json'),
        (completed['logical_rank0_completion_sha256'], gpu / 'workers/0.json'),
        (final['parallel_recovery_completion_sha256'], parallel / 'completion.json'),
    ]
    if any(expected != sha_file(path) for expected, path in pairs):
        raise ValueError('Parallel recovery history hash differs')
    if (inv['plan_sha256'] != recovery['original_plan_sha256']
            or inv['runner_sha256'] != original['runner_sha256']
            or inv['scorer_sha256'] != original['scorer_sha256']
            or inv['changed_model_or_scorer_or_assignments'] or inv['training_updates'] != 0
            or intent['training_updates'] != 0 or not intent['no_model_or_scorer_or_per_file_batch_change']):
        raise ValueError('The recovery chain changed the frozen measurement scope')
    if (prior != {'exit_codes': {str(i): int(i == 0) for i in range(8)}, 'all_success': False}
            or net_exits['exit_codes'] != {str(i): -15 if i == 0 else 0 for i in range(8)}
            or net_exits['all_success'] or net_complete['exit_code'] != -15
            or intent['controlled_interruption_pid'] != net_complete['pid']
            or net_complete['pid'] != net_launch['pid'] or intent['expected_signal'] != 'SIGTERM'):
        raise ValueError('Original failure or controlled interruption is not preserved')
    wanted = {str(i): 0 for i in range(8)}
    if (not exits['all_success'] or exits['actual_process_exit_codes'] != wanted
            or completed['actual_process_exit_codes'] != wanted or not completed['complete']
            or not final['all_success'] or final['exit_codes'] != wanted
            or final['original_rank0_single_resume_exit_code'] != -15
            or len(launch['pids']) != 8 or len(set(launch['pids'])) != 8
            or launch['slots'] != list(range(8))):
        raise ValueError('Eight actual successful recovery processes are required')
    if (not controls['passed'] or controls['rows'] != 32
            or not controls['original_scoring_core_ast_identical']
            or controls['original_plan_sha256'] != recovery['original_plan_sha256']
            or controls['recovery_runner_sha256'] != recovery['runner_sha256']
            or controls['scorer_sha256'] != original['scorer_sha256']):
        raise ValueError('Actual recovery numerical controls do not match')
    if (completed['retained_files'] != len(recovery['retained_inputs'])
            or completed['retained_rows'] != sum(i['rows'] for i in recovery['retained_inputs'])
            or completed['new_files'] != len(recovery['missing_inputs'])
            or completed['new_rows'] != sum(i['rows'] for i in recovery['missing_inputs'])
            or completed['training_updates'] != 0 or completed['source_quality_or_target_changes'] != 0):
        raise ValueError('Recovery completion changes the data or coverage scope')
    for item in recovery['retained_inputs']:
        path = gpu / 'score-receipts' / (digest(item['remote']) + '.json')
        if sha_file(path) != recovery['retained_receipt_sha256'][item['remote']]:
            raise ValueError('A retained successful score receipt changed')
    slots = []
    for slot, items in enumerate(recovery['assignments']):
        part = read_json(parallel / 'slots' / f'{slot}.json')
        if (not part['complete'] or part['slot'] != slot or part['rows'] != sum(i['rows'] for i in items)
                or part['recovery_plan_sha256'] != sha_file(parallel / 'plan.json')):
            raise ValueError('Recovery slot does not cover its exact assigned files')
        slots.append(part)
        for item in items:
            receipt = read_json(gpu / 'score-receipts' / (digest(item['remote']) + '.json'))
            if (receipt['gpu_rank'] != slot or receipt['original_logical_rank'] != 0
                    or receipt['recovery_plan_sha256'] != sha_file(parallel / 'plan.json')
                    or receipt['recovery_runner_sha256'] != recovery['runner_sha256']):
                raise ValueError('A recovered score is not bound to its actual slot and implementation')
    network_packet = {'investigation': inv, 'prior_workers_exited': prior,
        'network_preflight': read_json(gpu / 'network-resume-preflight-20261105.json'),
        'launch': net_launch, 'completion': net_complete, 'current_workers_exited': final,
        'all_history_hashes_verified': True, 'failed_attempt_hidden': False,
        'scope': 'Initial network recovery was later deliberately interrupted for eight-GPU scheduling; exit -15 is retained.',
        'new_training_updates': 0}
    parallel_packet = {'plan': recovery, 'switch_intent': intent, 'launch': launch,
        'partition_exits': exits, 'slot_completions': slots, 'completion': completed,
        'numerical_controls': controls, 'all_history_hashes_verified': True,
        'scope': 'Original logical rank-0 files were physically repartitioned, preserving the original per-file computation and all successful outputs.'}
    return network_packet, parallel_packet
