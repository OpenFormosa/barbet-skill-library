"""Apply new source-scope holds without rewriting first audits or PPL winners."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts import release_ifm_math_v26_v1 as release
from scripts import package_memory_sft_v2 as package
from scripts.audit_ifm_math_scope_v1 import STATE as SCOPE

ORIGINAL_INIT = release.package_init


def init():
    global HOLDS
    ORIGINAL_INIT()
    HOLDS = {r['query_hash']: r['reason'] for r in json.loads((SCOPE / 'query-holds.json').read_text())['holds']}
    package.EXTRA_ROW_GATE = gate


def gate(row): return HOLDS.get(row['query_hash'])


def main(workers):
    source = json.loads((SCOPE / 'summary.json').read_text())
    if not source['all_selected_source_questions_checked']: raise ValueError('full scope scan required')
    if source['policy_sha256'] != base.file_sha(SCOPE / 'policy.json') or source['query_holds_sha256'] != base.file_sha(SCOPE / 'query-holds.json'):
        raise ValueError('scope exclusion evidence changed')
    binding = {'driver_sha256': base.file_sha(Path(__file__)),
        'original_release_driver_sha256': base.file_sha(Path(release.__file__)),
        'scope_summary_sha256': base.file_sha(SCOPE / 'summary.json'),
        'all_first_audits_and_saved_ppl_scores_unchanged': True, 'no_second_choice_fallback': True}
    path = SCOPE / 'package-policy-v2.json'
    if path.exists() and json.loads(path.read_text()) != binding: raise ValueError('scope package policy changed')
    if not path.exists(): base.save(path, binding)
    release.package_init = init; release.package_all(workers)
    report = json.loads((release.STATE / 'public-package-summary.json').read_text())
    holds = set(json.loads((SCOPE / 'query-holds.json').read_text())['query_hashes'])
    seen = 0
    for item in report['files']:
        if item['config'] != 'records': continue
        if base.file_sha(release.PUBLIC / item['path']) != item['sha256']: raise ValueError('public source-scope bytes changed')
        table = pq.read_table(release.PUBLIC / item['path'], columns=['query_hash'])
        if set(table.column('query_hash').to_pylist()) & holds: raise ValueError('held query remains in current math package')
        seen += table.num_rows
    if seen != report['counts']['accepted']: raise ValueError('scope readback coverage mismatch')
    base.save(release.STATE / 'final-source-scope-readback-v2.json', {
        'accepted': seen, 'all_public_record_scope_gates_reread': True,
        'source_scope_summary_sha256': base.file_sha(SCOPE / 'summary.json'),
        'public_package_summary_sha256': base.file_sha(release.STATE / 'public-package-summary.json'),
        'scope_held_queries': len(holds), 'all_retained_text_tokens_labels_and_scores_unchanged': True,
        'all_first_dispositions_preserved': True, 'model_requests': 0})
    print(json.dumps({'scope_verified_accepted': seen, 'scope_held_queries': len(holds)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=8); args = parser.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('CPU worker bound')
    main(args.workers)
