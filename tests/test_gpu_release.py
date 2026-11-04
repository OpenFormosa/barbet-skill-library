import copy
import math
from pathlib import Path
import pytest
from barbet_pipeline.common import sha_file, digest
from barbet_pipeline.gpu_release import FAMILIES, ScoreTotals, validate_measurement, validate_score_receipt, measured_curriculum_policy
from scripts.aggregate_full_v4_gpu_quality import merge_totals


def row(nll=2., targets=2):
    return {'record_id': 'a' * 64, 'input_tokens': 100, 'supervised_tokens': targets,
        'nll': nll, 'nll_sum': nll * targets, 'teacher_forced_top1_correct': 1,
        'readout_gradient_sketch_norm': 1., 'semantic_family_cosines': [0.1] * len(FAMILIES)}


@pytest.mark.parametrize('field,value', [
    ('nll', float('nan')), ('nll_sum', 17.), ('input_tokens', 131073),
    ('supervised_tokens', 100), ('teacher_forced_top1_correct', 3),
    ('readout_gradient_sketch_norm', -1.), ('record_id', 'a'),
    ('semantic_family_cosines', [1.1] * 7), ('semantic_family_cosines', [0.1] * 6)])
def test_invalid_measurements_rejected(field, value):
    r = row(); r[field] = value
    with pytest.raises(ValueError):
        validate_measurement(r)


def test_global_loss_is_token_weighted_not_file_average():
    a, b = ScoreTotals(), ScoreTotals()
    a.add(row(2., 2)); b.add(row(4., 8))
    merged = merge_totals([a.result(), b.result()])
    assert merged['token_weighted_nll'] == 3.6
    assert merged['example_mean_nll'] == 3.
    assert merged['teacher_forced_top1_fraction'] == .2
    assert merged['rows'] == 2 and merged['input_tokens'] == 200
    assert sum(merged['nll_histogram_upper_bounds'].values()) == 2


def test_receipts_bind_actual_input_and_score_bytes(tmp_path):
    item = {'remote': 'full-v4/verified/tokenized/train-example.parquet',
        'sha256': 'b' * 64, 'revision': 'c' * 40, 'rows': 1}
    output = tmp_path / 'scores' / (digest(item['remote']) + '.parquet')
    output.parent.mkdir(); output.write_bytes(b'explicit fixture, not a model score')
    plan = {'scorer_sha256': 'd' * 64, 'anchor_sha256': 'e' * 64}
    receipt = {'input_path': item['remote'], 'input_sha256': item['sha256'],
        'input_revision': item['revision'], 'rows': 1, **plan,
        'semantic_family_order': FAMILIES, 'training_updates': 0, 'no_truncation': True,
        'output_path': str(output), 'output_bytes': output.stat().st_size, 'output_sha256': sha_file(output)}
    assert validate_score_receipt(item, receipt, plan, tmp_path) == output
    wrong = {**receipt, 'input_revision': 'f' * 40}
    with pytest.raises(ValueError, match='identity'):
        validate_score_receipt(item, wrong, plan, tmp_path)
    output.write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        validate_score_receipt(item, receipt, plan, tmp_path)


def test_policy_does_not_erase_hard_or_value_aligned_data():
    rows = [dict(family='comparison', roots=287, discriminating_first_action_pairs=287,
        sum_logprob_pairs_correct=0, mean_logprob_pairs_correct=0),
        dict(family='operation_state', roots=285, discriminating_first_action_pairs=0,
        sum_logprob_pairs_correct=0, mean_logprob_pairs_correct=0)]
    policy = measured_curriculum_policy({'families': rows})
    assert policy['high_nll_is_not_automatic_rejection']
    assert policy['value_aligned_records_are_not_removed_for_student_disagreement']
    assert not policy['reference_only_automatic_admission']
    assert policy['semantic_family_policies'][0]['candidate_pair_failure_fraction'] == 1.
    assert policy['semantic_family_policies'][1]['candidate_pair_failure_fraction'] is None
    assert not policy['proposed_exposure']['validated']
