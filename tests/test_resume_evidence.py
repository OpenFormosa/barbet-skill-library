from pathlib import Path
import pytest

from barbet_pipeline.common import save_new, read_json, sha_file
from barbet_pipeline.resume_evidence import collect_network_resume_evidence


def fixture(gpu):
    directory = gpu / 'resume-network-20261105'
    item = {'remote': 'full-v4/verified/tokenized/train-test.parquet', 'sha256': 'a' * 64,
        'revision': 'b' * 40, 'bytes': 12, 'rows': 1}
    plan = {'rows': 9, 'files': 9, 'runner_sha256': 'c' * 64, 'scorer_sha256': 'd' * 64,
        'assignments': [[item]] + [[] for _ in range(7)]}
    save_new(gpu / 'full-score-plan.json', plan)
    prior = {'exit_codes': {str(i): int(i == 0) for i in range(8)}, 'all_success': False}
    save_new(directory / 'prior-workers-exited.json', prior)
    (gpu / 'worker-0.log').write_text('explicit test fixture: original network failure\n')
    (directory / 'worker-0.log').write_text('explicit test fixture: resumed process\n')
    preflight = {'input_path': item['remote'], 'input_revision': item['revision'], 'bytes': item['bytes'],
        'sha256': item['sha256'], 'expected_sha256': item['sha256'],
        'download_implementation_sha256': plan['runner_sha256']}
    save_new(gpu / 'network-resume-preflight-20261105.json', preflight)
    investigation = {'rank': 0, 'plan_sha256': sha_file(gpu / 'full-score-plan.json'),
        'runner_sha256': plan['runner_sha256'], 'scorer_sha256': plan['scorer_sha256'],
        'changed_model_or_scorer_or_assignments': False, 'training_updates': 0,
        'existing_scored_rows': 8, 'missing_rows': 1, 'existing_scored_files': 8, 'missing_files': 1,
        'successful_scores_hash_verified_before_resume': True,
        'prior_exit_sha256': sha_file(directory / 'prior-workers-exited.json'),
        'original_worker_log_sha256': sha_file(gpu / 'worker-0.log'),
        'recovered_input_receipt_sha256': sha_file(gpu / 'network-resume-preflight-20261105.json')}
    save_new(directory / 'investigation.json', investigation)
    save_new(directory / 'launch.json', {'rank': 0, 'pid': 123, 'training_updates': 0,
        'plan_sha256': investigation['plan_sha256']})
    completion = {'rank': 0, 'pid': 123, 'exit_code': 0, 'elapsed_seconds': 1.,
        'investigation_sha256': sha_file(directory / 'investigation.json'),
        'prior_workers_exited_sha256': sha_file(directory / 'prior-workers-exited.json'),
        'resumed_worker_log_sha256': sha_file(directory / 'worker-0.log')}
    save_new(directory / 'completion.json', completion)
    save_new(gpu / 'workers-exited.json', {'exit_codes': {str(i): 0 for i in range(8)},
        'all_success': True, 'prior_failure_preserved': True,
        'resume_completion_sha256': sha_file(directory / 'completion.json')})
    return directory


def test_actual_history_chain_is_required(tmp_path):
    directory = fixture(tmp_path)
    result = collect_network_resume_evidence(tmp_path)
    assert result['all_history_hashes_verified'] and not result['failed_attempt_hidden']
    (directory / 'worker-0.log').write_text('changed')
    with pytest.raises(ValueError, match='history hash'):
        collect_network_resume_evidence(tmp_path)


def test_missing_resume_completion_does_not_pass(tmp_path):
    directory = fixture(tmp_path)
    (directory / 'completion.json').unlink()
    with pytest.raises(FileNotFoundError):
        collect_network_resume_evidence(tmp_path)


def test_success_flag_cannot_hide_nonzero_exit(tmp_path):
    fixture(tmp_path)
    p = tmp_path / 'workers-exited.json'
    value = read_json(p); value['exit_codes']['0'] = 1
    p.unlink(); save_new(p, value)
    with pytest.raises(ValueError, match='did not complete'):
        collect_network_resume_evidence(tmp_path)


def test_different_process_cannot_supply_completion(tmp_path):
    directory = fixture(tmp_path)
    p = directory / 'launch.json'
    value = read_json(p); value['pid'] += 1
    p.unlink(); save_new(p, value)
    with pytest.raises(ValueError, match='real successful process'):
        collect_network_resume_evidence(tmp_path)
