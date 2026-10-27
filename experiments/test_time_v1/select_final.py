#!/usr/bin/env python3
"""Apply the registered validation gate in candidate registration order."""
from __future__ import annotations
import copy
import datetime as dt
import json
from pathlib import Path
from compare import paired

HERE = Path(__file__).resolve().parent
PRIMARY = ['drcd', 'gsm8k', 'mbpp', 'tools']


def main():
    if (HERE / 'FINAL_SELECTION.json').exists():
        raise RuntimeError('never select a different policy after final sealing')
    policies = json.loads((HERE / 'policies.json').read_text())['policies']
    if len(policies) != 18:
        raise RuntimeError('declared search budget must be exhausted before this selection')
    base = HERE / 'work/scores/confirm/validation'
    order = ['bare', 'routed', 'literal', 'focus', 'knowledge']
    reports = {n: json.loads((base / f'mini-{n}.json').read_text()) for n in order}
    for name, r in reports.items():
        if not r['complete'] or r['summary']['unique_questions'] != 259 or r['summary']['seeds'] != [20261027, 20261028, 20261029]:
            raise RuntimeError('three-seed confirmation is incomplete: ' + name)
    incumbent, decisions = 'bare', []
    for name in order[1:]:
        comparison = paired(reports[incumbent], reports[name])
        domains = comparison['domains']
        passed = (comparison['primary_macro']['delta_pp'] >= 1 - 1e-9
            and all(domains[d]['delta_pp'] >= -3 - 1e-9 for d in PRIMARY)
            and domains['tmmluplus']['delta_pp'] >= -3 - 1e-9)
        decisions.append({'incumbent_before': incumbent, 'candidate': name,
                          'promoted': passed, 'comparison': comparison})
        if passed:
            incumbent = name
    result = {'selected_policy': incumbent, 'candidate_count': len(policies),
        'selection_data': '259 validation questions, three fixed seeds; no final-test access',
        'candidate_order': order, 'decisions': decisions,
        'stopping_reason': 'Declared 18-policy search budget exhausted; local budget stopping, not global optimality.',
        'final_primary_claim': 'Selection is frozen before final generation; no policy swap based on final test.',
        'selected_config': policies[incumbent]}
    (HERE / 'selection_decision.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    selected = copy.deepcopy(policies[incumbent])
    if selected.get('program'):
        raise RuntimeError('derived controls require explicit handling for program-only output')
    def control(**changes):
        value = copy.deepcopy(selected)
        value.update(changes)
        # An ablation applies to the same field in all contract overrides.
        for override in value.get('contract_overrides', {}).values():
            for key, val in changes.items():
                if key in override:
                    override[key] = val
        return value
    final = {n: policies[n] for n in dict.fromkeys(['bare', 'long', incumbent])}
    final['no_card'] = control(skill=False)
    final['no_rag'] = control(retrieval=False, knowledge_retrieval=False)
    final['no_feedback'] = control(execute=False, repair=0)
    final['self_review'] = control(execute=False, repair=0, self_revise=2)
    pairs = [['mini', name] for name in final] + [['qwen', n] for n in dict.fromkeys(['bare', 'long', incumbent])]
    (HERE / 'final_policies.json').write_text(json.dumps({'policies': final,
        'comparison_pairs': pairs,
        'notice': 'Frozen evaluation-only removals/control; not additional searched candidates and never selected using final test.',
        'no_card_scope': 'Remove optional cards; retain the selected public output-format contract, e.g. literal answer copying.',
        'no_rag_scope': 'Use original source-packet order at the same input cap; disable optional global knowledge retrieval.',
        'self_review_scope': 'Same maximum three calls, unsupported review instead of external execution feedback; report actual cost differences.'}, ensure_ascii=False, indent=2) + '\n')
    (HERE / '../../testtime/recommended_policy.json').resolve().write_text(json.dumps({
        'status': 'selected on validation; final evidence is reported separately',
        'name': incumbent, 'config': selected,
        'scope': 'MiniCPM5-1B pinned revision; public output-contract routing; not a universal best policy',
        'source': 'experiments/test_time_v1/selection_decision.json'}, ensure_ascii=False, indent=2) + '\n')
    ledger = json.loads((HERE / 'decision_ledger.json').read_text())
    ledger.append({'round': 3, 'stage': 'three-seed confirmation completed',
        'selected_policy': incumbent, 'stopping_reason': result['stopping_reason'],
        'promotion_results': [{'candidate': d['candidate'], 'promoted': d['promoted'],
            'macro_delta_pp': d['comparison']['primary_macro']['delta_pp']} for d in decisions]})
    (HERE / 'decision_ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected': incumbent, 'pairs': pairs, 'search_stopped': True}))


if __name__ == '__main__':
    main()
