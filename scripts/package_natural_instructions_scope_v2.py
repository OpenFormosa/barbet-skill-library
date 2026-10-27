"""Prospective final NI scope after detecting jointly required output items.

Retain all first reviews and exact compiled bytes. A new public package applies
the narrower source contract without relabeling rejected records or rescoring
unchanged retained conditions. The superseded partial package is never published.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts import release_natural_instructions_v26_v1 as release

ORIGINAL = release.STATE
STATE = release.SOURCE / 'selected-scope-v2'
PUBLIC = base.WORK / 'exports/memory-harness-v2-natural-instructions-public-scope-v2'
SOURCE = release.SOURCE
CONFIG = base.REPO / 'configs/natural_instructions_reference_semantics_v1.json'
ORIGINAL_PACKAGE_INIT = release.package_init


def scope_hold(ref, policy):
    name = Path(ref['file']).name
    match = re.match(r'task(\d+)_', name)
    if not match: return 'unknown_ni_task_identity'
    task = int(match[1])
    if any(marker in name for marker in policy['benchmark_source_name_markers']):
        return 'source_benchmark_without_verified_upstream_train_provenance'
    if task in policy['joint_or_unresolved_reference_task_ids']:
        return 'joint_or_unresolved_reference_output_semantics'
    if task in policy['bare_fact_task_ids'] or name in policy['bare_fact_files']:
        return 'target_facts_not_supplied'
    if task in policy['intentional_wrong_target_task_ids']:
        return 'intentionally_wrong_answer_task_outside_scope'
    if ref['reference_alternatives'] > 1 and task not in policy['complete_alternative_task_ids'] and name not in policy['additional_complete_alternative_files']:
        return 'unreviewed_multi_reference_contract'
    return None


def prepare():
    STATE.mkdir(exist_ok=True)
    path = STATE / 'final-scope-policy-v2.json'
    if path.exists():
        policy = json.loads(path.read_text())
        if policy['driver_sha256'] != base.file_sha(Path(__file__)) or policy['scope_config_sha256'] != base.file_sha(CONFIG):
            raise ValueError('final NI scope changed after freezing')
        return
    if not (ORIGINAL / 'SUPERSEDED.json').exists(): raise ValueError('original package must be marked superseded')
    if not (ORIGINAL / 'semantic-audit-v1/summary.json').exists(): raise ValueError('first winner diagnosis incomplete')
    for name in ['compiled', 'semantic-audit-v1', 'normalized']:
        # Read-only links to immutable original compiled and review artifacts.
        (STATE / name).symlink_to(ORIGINAL / name, target_is_directory=True)
    for name in ['dedup-summary.json', 'compile-summary.json', 'ppl-score-binding.json', 'release-scope.json',
                 'additional-quality-holds-v1.json', 'winner-diagnosis-policy-v1.json']:
        shutil.copyfile(ORIGINAL / name, STATE / name)
    policy = json.loads(CONFIG.read_text())
    counts = Counter(); per_task = Counter(); seen = 0
    for path_in in sorted((ORIGINAL / 'selected').glob('*.jsonl')):
        for line in path_in.open():
            row = json.loads(line); seen += 1; reason = scope_hold(row['source_refs'], policy)
            if reason:
                counts[reason] += 1; per_task[Path(row['source_refs']['file']).name] += 1
    base.save(path, {'driver_sha256': base.file_sha(Path(__file__)), 'scope_config_sha256': base.file_sha(CONFIG),
        'original_first_winner_audit_sha256': base.file_sha(ORIGINAL / 'semantic-audit-v1/summary.json'),
        'original_compilation_summary_sha256': base.file_sha(ORIGINAL / 'compile-summary.json'),
        'reference_semantics_inventory_sha256': base.file_sha(SOURCE / 'reference-output-semantics-inventory-v1.json'),
        'selected_records_examined': seen, 'selected_scope_holds': dict(counts), 'held_selected_rows_by_task': dict(per_task),
        'first_semantic_holds_also_apply': True, 'earlier_candidate_scores_unchanged': True,
        'retained_answers_and_conditions_unchanged': True, 'old_package_never_published': True,
        'weights_trained': False})
    print(json.dumps({'selected_records_examined': seen, 'scope_holds': dict(counts)}), flush=True)


def package_init():
    ORIGINAL_PACKAGE_INIT()
    policy = json.loads(CONFIG.read_text())
    release.package.EXTRA_ROW_GATE = lambda row: scope_hold(json.loads(row['source_refs_json']), policy)


def main(workers):
    prepare()
    release.STATE = STATE; release.PUBLIC = PUBLIC; release.package_init = package_init
    release.package_all(workers)
    # Independently inspect all exported record metadata, rather than assume the
    # callback actually applied the scope to every shard.
    policy = json.loads(CONFIG.read_text()); counts = Counter()
    for path in sorted((PUBLIC / 'data/wave-0004/records').glob('*.parquet')):
        for row in pq.read_table(path, columns=['source_refs_json']).to_pylist():
            ref = json.loads(row['source_refs_json'])
            if scope_hold(ref, policy): raise ValueError('held source contract entered public package')
            counts[Path(ref['file']).name] += 1
    summary = json.loads((STATE / 'public-package-summary.json').read_text())
    if sum(counts.values()) != summary['counts']['accepted']: raise ValueError('scope readback coverage mismatch')
    base.save(STATE / 'final-scope-readback-v2.json', {'all_public_record_scope_gates_reread': True,
        'accepted': sum(counts.values()), 'by_task': dict(counts), 'scope_config_sha256': base.file_sha(CONFIG),
        'policy_sha256': base.file_sha(STATE / 'final-scope-policy-v2.json'),
        'original_first_source_and_winner_reviews_preserved': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=16); args = p.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('NI package worker bound')
    main(args.workers)
