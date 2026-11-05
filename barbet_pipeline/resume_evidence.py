"""Validate the retained failure and real process completion before publication."""
from pathlib import Path

from .common import read_json, sha_file


def collect_network_resume_evidence(gpu):
    gpu = Path(gpu)
    directory = gpu / 'resume-network-20261105'
    exited = read_json(gpu / 'workers-exited.json')
    if not directory.exists():
        if 'resume_completion_sha256' in exited:
            raise ValueError('Resume completion points to missing history')
        return None
    plan = read_json(gpu / 'full-score-plan.json')
    investigation = read_json(directory / 'investigation.json')
    prior = read_json(directory / 'prior-workers-exited.json')
    launch = read_json(directory / 'launch.json')
    completion = read_json(directory / 'completion.json')
    preflight = read_json(gpu / 'network-resume-preflight-20261105.json')
    required_hashes = [
        (investigation['prior_exit_sha256'], directory / 'prior-workers-exited.json'),
        (investigation['original_worker_log_sha256'], gpu / 'worker-0.log'),
        (investigation['recovered_input_receipt_sha256'], gpu / 'network-resume-preflight-20261105.json'),
        (completion['investigation_sha256'], directory / 'investigation.json'),
        (completion['prior_workers_exited_sha256'], directory / 'prior-workers-exited.json'),
        (completion['resumed_worker_log_sha256'], directory / 'worker-0.log'),
        (exited['resume_completion_sha256'], directory / 'completion.json'),
    ]
    if any(expected != sha_file(path) for expected, path in required_hashes):
        raise ValueError('Network resume history hash differs')
    if prior != {'exit_codes': {str(i): int(i == 0) for i in range(8)}, 'all_success': False}:
        raise ValueError('Original failed worker receipt is not preserved')
    if (not exited['all_success'] or exited['exit_codes'] != {str(i): 0 for i in range(8)}
            or not exited.get('prior_failure_preserved')):
        raise ValueError('Resumed workload did not complete successfully')
    if (completion['rank'] != 0 or launch['rank'] != 0 or completion['exit_code'] != 0
            or completion['pid'] != launch['pid'] or completion['elapsed_seconds'] < 0):
        raise ValueError('Resume is not bound to its real successful process')
    if (investigation['plan_sha256'] != sha_file(gpu / 'full-score-plan.json')
            or launch['plan_sha256'] != investigation['plan_sha256']
            or investigation['runner_sha256'] != plan['runner_sha256']
            or investigation['scorer_sha256'] != plan['scorer_sha256']
            or investigation['changed_model_or_scorer_or_assignments']
            or investigation['training_updates'] != 0 or launch['training_updates'] != 0):
        raise ValueError('Resume changed the frozen measurement contract')
    if (investigation['existing_scored_rows'] + investigation['missing_rows'] != plan['rows']
            or investigation['existing_scored_files'] + investigation['missing_files'] != plan['files']
            or not investigation['successful_scores_hash_verified_before_resume']):
        raise ValueError('Resumed work does not account for the complete original plan')
    inputs = {i['remote']: i for i in plan['assignments'][0]}
    first = inputs.get(preflight['input_path'])
    if (first is None or preflight['sha256'] != first['sha256']
            or preflight['expected_sha256'] != first['sha256']
            or preflight['bytes'] != first['bytes'] or preflight['input_revision'] != first['revision']
            or preflight['download_implementation_sha256'] != plan['runner_sha256']):
        raise ValueError('Recovered network input differs from the frozen source')
    return {'scope': 'One network-failed worker resumed with the unchanged original scoring implementation.',
        'investigation': investigation, 'prior_workers_exited': prior, 'network_preflight': preflight,
        'launch': launch, 'completion': completion, 'current_workers_exited': exited,
        'all_history_hashes_verified': True, 'failed_attempt_hidden': False,
        'new_training_updates': 0}
