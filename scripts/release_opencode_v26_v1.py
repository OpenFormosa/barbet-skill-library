"""First OpenCode winner diagnosis followed by exact shared SFT export gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from barbet_inference.memory import canonical
from scripts import audit_memory_sft_v2 as reviewer
from scripts import build_memory_sft_v2 as base
from scripts import release_natural_instructions_v26_v1 as shared
from scripts.finalize_opencode_v26_v1 import STATE, OUT, SOURCE
from scripts.publish_natural_instructions_v26_v1 import STATE as PRIOR

PUBLIC = base.WORK / 'exports/memory-harness-v2-opencode-public-v1'
WAVE = 'wave-0005'


def prepare_audit():
    binding_path = STATE / 'winner-diagnosis-policy-v1.json'
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        if binding['driver_sha256'] != base.file_sha(Path(__file__)):
            raise ValueError('frozen OpenCode winner driver changed')
        if binding['plan_sha256'] != base.file_sha(STATE / 'semantic-audit-v1/plan.json'):
            raise ValueError('frozen winner plan changed')
        return
    if not (STATE / 'release-scope.json').exists(): raise ValueError('OpenCode selection incomplete')
    first = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    reviewed = {r['record_id'] for r in first['results']}; pools = defaultdict(list)
    for path in sorted((STATE / 'selected').glob('*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            if row['record_id'] in reviewed: continue
            ref = row['source_refs']
            key = (ref['upstream_domain'], ref['generation_algorithm'], ref['answer_selection']['method'])
            pool = pools[key]; pool.append((base.sha('opencode-first-winner-diagnosis:' + row['record_id']), row))
            if len(pool) > 8:
                pool.sort(key=lambda x: x[0]); del pool[4:]
    selected = [row for key in sorted(pools) for _, row in sorted(pools[key], key=lambda x: x[0])[:4]]
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-selected-winners.jsonl').open('w') as handle:
        for row in selected: handle.write(canonical(row) + '\n')
    original_plan = json.loads((SOURCE / 'semantic-audit-v1/plan.json').read_text())
    plan = {key: original_plan[key] for key in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
    plan.update(jobs=[{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'coding',
                      'messages': r['messages'], 'tools': [], 'observation_origin': r['observation_origin'],
                      'source_domain': r['source_refs']['upstream_domain'],
                      'generation_algorithm': r['source_refs']['generation_algorithm'],
                      'selection_method': r['source_refs']['answer_selection']['method']} for r in selected],
                scope='Up to four distinct new winners per source domain, generation method and PPL/singleton group. First source-reviewed records excluded. Diagnostic only, not corpus accuracy or all-row approval.',
                first_non_accept_is_final_for_that_record=True)
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', plan)
    base.save(binding_path, {'driver_sha256': base.file_sha(Path(__file__)),
        'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'source_audit_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'shared_package_driver_sha256': base.file_sha(base.REPO / 'scripts/release_natural_instructions_v26_v1.py'),
        'samples': len(selected), 'first_source_reviews_not_replayed': len(reviewed)})
    print(json.dumps({'winner_samples': len(selected), 'source_reviews_not_replayed': len(reviewed)}), flush=True)


def package_init():
    package = shared.package
    package.STATE = STATE; package.OUT = OUT; package.PUBLIC = PUBLIC; package.WAVE = WAVE
    package.EXTRA_ROW_GATE = None; package.init()
    first = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    policy = json.loads((SOURCE / 'full-normalization-policy-v1.json').read_text())
    package.HOLD_IDS.update(first['excluded_record_ids']); package.HOLD_QUERIES.update(policy['held_query_hashes'])


def package_all(workers):
    prepare_audit()
    policy = json.loads((STATE / 'winner-diagnosis-policy-v1.json').read_text())
    if policy['shared_package_driver_sha256'] != base.file_sha(base.REPO / 'scripts/release_natural_instructions_v26_v1.py'):
        raise ValueError('shared exact-publication gate changed')
    # The shared exporter checks complete stored values, literal renderings,
    # all budgets, cumulative question/family identity and proof coverage.
    shared.STATE = STATE; shared.OUT = OUT; shared.PUBLIC = PUBLIC; shared.SOURCE = SOURCE
    shared.WAVE = WAVE; shared.PRIOR = PRIOR; shared.prepare_audit = prepare_audit; shared.package_init = package_init
    shared.package_all(workers)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare-audit', 'audit', 'package'])
    p.add_argument('--workers', type=int, default=16); args = p.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('OpenCode public package worker bound')
    if args.phase == 'prepare-audit': prepare_audit()
    elif args.phase == 'audit':
        prepare_audit(); reviewer.STATE = STATE; reviewer.main()
    else: package_all(args.workers)
