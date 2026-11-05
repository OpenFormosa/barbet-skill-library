"""Explicit, disjoint scheduling of unfinished files without changing scoring."""
import ast
from pathlib import Path

from .common import read_json, sha_file


def writer_core(path):
    tree = ast.parse(Path(path).read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == 'ParquetWriter' for item in node.items)]
    if len(matches) != 1:
        raise ValueError('Exactly one canonical Parquet scoring loop is required')
    return ast.dump(matches[0], include_attributes=False)


def partition_missing(items, slots=8):
    if not 1 <= slots <= 8 or len({i['remote'] for i in items}) != len(items):
        raise ValueError('Invalid or duplicate unfinished workload')
    assignments = [[] for _ in range(slots)]
    loads = [0] * slots
    for item in sorted(items, key=lambda i: (-i['bytes'], i['remote'])):
        slot = min(range(slots), key=lambda i: loads[i])
        assignments[slot].append(item)
        loads[slot] += item['bytes']
    return assignments


def validate_partition(original, recovery):
    assigned = [item for slot in recovery['assignments'] for item in slot]
    pending = recovery['missing_inputs']
    if len(recovery['assignments']) != 8:
        raise ValueError('Recovery must use exactly the declared eight local slots')
    if len({i['remote'] for i in assigned}) != len(assigned):
        raise ValueError('A recovery input appears in multiple slots')
    if sorted(assigned, key=lambda i: i['remote']) != sorted(pending, key=lambda i: i['remote']):
        raise ValueError('Recovery partitions omit or change an unfinished input')
    planned = {i['remote']: i for i in original['assignments'][0]}
    if any(planned.get(i['remote']) != i for i in pending):
        raise ValueError('Recovery input differs from its original logical assignment')
    retained = recovery['retained_inputs']
    all_inputs = [i for slot in original['assignments'] for i in slot]
    if (len(retained) + len(pending) != original['files']
            or sum(i['rows'] for i in retained) + sum(i['rows'] for i in pending) != original['rows']
            or {i['remote'] for i in retained}.intersection(i['remote'] for i in pending)
            or len({i['remote'] for i in retained}) != len(retained)
            or {i['remote'] for i in retained + pending} != {i['remote'] for i in all_inputs}):
        raise ValueError('Retained and unfinished files do not cover the original plan exactly once')
    originals = {i['remote']: i for i in all_inputs}
    if any(originals[i['remote']] != i for i in retained):
        raise ValueError('Retained input identity changed')


def verify_recovery_contract(code, gpu, recovery):
    code, gpu = Path(code), Path(gpu)
    original = read_json(gpu / 'full-score-plan.json')
    if recovery['original_plan_sha256'] != sha_file(gpu / 'full-score-plan.json'):
        raise ValueError('Original full scoring plan changed')
    if (sha_file(code / 'barbet_pipeline/gpu_quality.py') != original['scorer_sha256']
            or sha_file(code / 'scripts/run_full_v4_gpu_quality.py') != original['runner_sha256']
            or sha_file(code / 'scripts/run_full_v4_gpu_parallel_recovery.py') != recovery['runner_sha256']):
        raise ValueError('Frozen scorer or bound recovery implementation changed')
    if writer_core(code / 'scripts/run_full_v4_gpu_quality.py') != writer_core(code / 'scripts/run_full_v4_gpu_parallel_recovery.py'):
        raise ValueError('The core scoring loop differs from the original frozen worker')
    validate_partition(original, recovery)
    return original
