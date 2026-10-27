#!/usr/bin/env python3
"""Final comparisons and descriptive failure accounting; never updates a policy."""
from __future__ import annotations
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path

from compare import paired

HERE = Path(__file__).resolve().parent
PRIMARY = ['drcd', 'gsm8k', 'mbpp', 'tools']


def main():
    seal = json.loads((HERE / 'FINAL_SELECTION.json').read_text())
    chosen = json.loads((HERE / 'selection_decision.json').read_text())['selected_policy']
    base = HERE / 'work/scores/final/test'
    reports = {(family, policy): json.loads((base / f'{family}-{policy}.json').read_text())
               for family, policy in seal['comparisons']}
    if any(not r['complete'] for r in reports.values()):
        raise RuntimeError('all sealed final arms must be complete before the final analysis')
    student = reports['mini', chosen]
    comparisons = {
        'selected_minus_bare': paired(reports['mini', 'bare'], student),
        'selected_minus_long': paired(reports['mini', 'long'], student),
        'qwen_bare_minus_selected': paired(student, reports['qwen', 'bare']),
        'qwen_long_minus_selected': paired(student, reports['qwen', 'long']),
        'qwen_same_minus_selected': paired(student, reports['qwen', chosen])}
    for policy in ['no_card', 'no_rag', 'no_feedback', 'self_review']:
        comparisons['selected_minus_' + policy] = paired(reports['mini', policy], student)
    gates = {}
    gates['measured_improvement'] = comparisons['selected_minus_bare']['primary_macro']['paired_95ci_pp'][0] > 0
    for key in ['qwen_bare_minus_selected', 'qwen_long_minus_selected', 'qwen_same_minus_selected']:
        value = comparisons[key]
        gates[key] = {'primary_near_9b': value['primary_macro']['paired_95ci_pp'][1] <= 3
            and all(value['domains'][d]['delta_pp'] <= 5 for d in PRIMARY),
            'macro_gap_upper_ci_pp': value['primary_macro']['paired_95ci_pp'][1],
            'max_primary_domain_point_gap_pp': max(value['domains'][d]['delta_pp'] for d in PRIMARY),
            'knowledge_gap_pp': value['domains']['tmmluplus']['delta_pp'],
            'knowledge_gap_95ci_pp': value['domains']['tmmluplus']['paired_95ci_pp']}
    gates['broad_general_parameter_equivalence'] = 'Not established by this finite task suite; broad knowledge and same-harness comparison remain mandatory evidence.'
    comparisons['gates'] = gates
    comparisons['selected_policy'] = chosen
    comparisons['analysis_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    (HERE / 'final_comparisons.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + '\n')

    scores = student['scores']
    domains = defaultdict(list)
    for row in scores:
        domains[row['domain']].append(row)
    diagnostics = {'unit_notice': 'Attempt counts include three seeds; unique-question statistics explicitly average/group seeds. No hidden label or final failure was fed back into policy selection.',
        'reading': {'attempts': len(domains['drcd']),
            'support_visible': sum(r['support_visible'] for r in domains['drcd']),
            'wrong_despite_visible_support': sum(r['support_visible'] and not r['correct'] for r in domains['drcd']),
            'character_f1': student['summary']['secondary']['qa_character_f1'],
            'limitation': 'Strict normalized EM penalizes aliases, omitted units and extra spans as well as semantic errors. F1 is partial overlap, not a human correctness judgment.'},
        'math': {'attempts': len(domains['gsm8k']),
            'unparseable_final': sum(not r['numeric_parseable'] for r in domains['gsm8k']),
            'parseable_but_wrong': sum(r['numeric_parseable'] and not r['correct'] for r in domains['gsm8k']),
            'limitation': 'A parseable number does not validate the word-problem model.'},
        'code': {'attempts': len(domains['mbpp']),
            'public_pass_but_hidden_failure': sum(r['public_pass_before_scoring'] and not r['correct'] for r in domains['mbpp']),
            'failure_statuses': dict(Counter(r['execution_status'] for r in domains['mbpp'] if not r['correct'])),
            'limitation': 'Restricted reference-executable MBPP subset; actual public feedback cannot expose hidden edge cases.'},
        'tools': {'attempts': len(domains['tools']),
            'schema_valid': sum(r['schema_valid'] for r in domains['tools']),
            'valid_but_semantically_wrong': sum(r['schema_valid'] and not r['correct'] for r in domains['tools']),
            'wrong_fields': dict(Counter(k for r in domains['tools'] for k in r['wrong_arguments'])),
            'limitation': 'Single-call synthetic family holdout, not full multi-step agent reliability.'}}
    mc = defaultdict(list)
    for row in domains['tmmluplus']:
        mc[row['id']].append(row['correct'])
    diagnostics['knowledge'] = {'unique_questions': len(mc),
        'attempts': len(domains['tmmluplus']),
        'unparseable_final': sum(not r['letter_parseable'] for r in domains['tmmluplus']),
        'valid_letter_but_wrong': sum(r['letter_parseable'] and not r['correct'] for r in domains['tmmluplus']),
        'all_three_wrong': sum(not any(v) for v in mc.values()),
        'some_but_not_all_correct': sum(any(v) and not all(v) for v in mc.values()),
        'all_three_correct': sum(all(v) for v in mc.values()),
        'limitation': 'Any-correct across seeds is only a diagnostic oracle, not a deployable selection rule.'}
    diagnostics['serving'] = {'unclosed_thinking_calls': sum(r['unclosed_thinking_calls'] for r in scores),
        'length_stopped_calls': sum(r['length_stopped_calls'] for r in scores),
        'batch_size': seal['serving']['batch_size'],
        'cost_unit': 'Prompt/output tokens across every returned model call; batch seconds are amortized throughput, not per-user latency.'}
    call_accounting = []
    for (family, policy), report in reports.items():
        calls, inp, out = 0, 0, 0
        for path in sorted((HERE / 'work/call_logs/final/test' / family / policy).glob('*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                calls += 1; inp += row['prompt_tokens']; out += row['output_tokens']
        retained = sum(r['calls'] for r in report['scores'])
        call_accounting.append({'model': family, 'policy': policy,
            'all_logged_calls': calls, 'calls_in_completed_trajectories': retained,
            'extra_returned_calls_from_interrupted_attempts': calls-retained,
            'total_prompt_tokens': inp, 'total_output_tokens': out})
        if calls < retained:
            raise RuntimeError('model-call log is incomplete')
    diagnostics['all_model_call_accounting'] = call_accounting
    schedule = json.loads((HERE / 'work/final_job_schedule.json').read_text())
    diagnostics['job_chain'] = [{k:v for k,v in job.items() if k not in {'command'}} for job in schedule['jobs']]
    (HERE / 'final_diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected': chosen, 'gates': gates}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
