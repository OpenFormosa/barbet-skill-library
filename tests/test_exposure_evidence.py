import pytest
from barbet_pipeline.exposure_evidence import summarize_training_exposure


def counts(rows, supervised):
    return {'rows': rows, 'input_tokens': 100 * rows, 'supervised_tokens': supervised}


def test_exposure_uses_actual_supervised_tokens_and_train_only():
    data = {'all_exported_rows_measured': True, 'exposure_groups': {
        'verified|Wikidata5M/original-inductive-train|reading|train': counts(900, 90),
        'verified|qa-source|reading|train': counts(10, 5),
        'source_reference|natural-instructions|reading|train': counts(10, 5),
        'reference_only|coding-source|coding|train': counts(1, 900),
        'reference_only|heldout|coding|validation': counts(100000, 10000000)}}
    result = summarize_training_exposure(data)
    assert result['totals']['rows'] == 921
    assert result['totals']['supervised_tokens'] == 1000
    assert result['reference_only_supervised_fraction'] == .9
    assert result['finite_graph_supervised_fraction_all_train'] == .09
    assert result['finite_graph_supervised_fraction_of_verified_plus_source_reference'] == .9
    assert result['non_graph_with_verification_or_source_reference']['supervised_tokens'] == 10
    assert 'reference_only|heldout' not in result['by_source_and_scope']
    assert not result['curriculum_constraints']['reference_only_automatic_training_admission']


def test_partial_statistics_cannot_be_presented_as_whole_corpus():
    with pytest.raises(ValueError, match='Complete corpus'):
        summarize_training_exposure({'all_exported_rows_measured': False, 'exposure_groups': {}})


def test_unknown_quality_scope_is_not_promoted():
    with pytest.raises(ValueError, match='Unknown quality'):
        summarize_training_exposure({'all_exported_rows_measured': True, 'exposure_groups': {
            'model_approved|source|coding|train': counts(1, 10)}})
