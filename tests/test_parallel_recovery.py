import copy
from pathlib import Path
import pytest
from barbet_pipeline.parallel_recovery import partition_missing, validate_partition, writer_core


def workload():
    items = [{'remote': f'file-{i}', 'bytes': 100 - i, 'rows': i + 1} for i in range(12)]
    original = {'assignments': [items] + [[] for _ in range(7)], 'files': 12,
        'rows': sum(i['rows'] for i in items)}
    recovery = {'missing_inputs': items[2:], 'retained_inputs': items[:2],
        'assignments': partition_missing(items[2:])}
    return original, recovery


def test_all_unfinished_files_are_disjoint_and_covered():
    original, recovery = workload()
    validate_partition(original, recovery)
    assert all(recovery['assignments'])


def test_duplicate_slot_cannot_rescore_the_same_input():
    original, recovery = workload()
    recovery['assignments'][1].append(recovery['assignments'][0][0])
    with pytest.raises(ValueError, match='multiple slots'):
        validate_partition(original, recovery)


def test_omitted_or_changed_input_cannot_pass():
    original, recovery = workload()
    recovery = copy.deepcopy(recovery)
    recovery['assignments'][0].pop()
    with pytest.raises(ValueError, match='omit or change'):
        validate_partition(original, recovery)


def test_a_retained_file_cannot_be_silently_dropped():
    original, recovery = workload()
    recovery['retained_inputs'].pop()
    with pytest.raises(ValueError, match='exactly once'):
        validate_partition(original, recovery)


def test_recovery_cannot_change_a_frozen_input():
    original, recovery = workload()
    recovery = copy.deepcopy(recovery)
    altered = recovery['missing_inputs'][0]
    altered['bytes'] += 1
    for group in recovery['assignments']:
        for item in group:
            if item['remote'] == altered['remote']:
                item['bytes'] = altered['bytes']
    with pytest.raises(ValueError, match='original logical assignment'):
        validate_partition(original, recovery)


def test_the_actual_scoring_loop_is_unchanged():
    root = Path(__file__).resolve().parents[1] / 'scripts'
    assert writer_core(root / 'run_full_v4_gpu_quality.py') == writer_core(root / 'run_full_v4_gpu_parallel_recovery.py')
