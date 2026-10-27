#!/usr/bin/env python3
"""Post-hoc format diagnosis. The sealed primary scorer remains unchanged.

Only an already-emitted final channel may be read. No number is extracted from
unfinished thinking, a question, a gold answer, or an ambiguous prose line.
"""
import json
import copy
from pathlib import Path
import re
from score import number, summarize
from compare import paired

HERE = Path(__file__).resolve().parent


def conservative_readout(final):
    parsed = number(final)
    if parsed is not None:
        return parsed
    lines = [x.strip() for x in final.splitlines() if x.strip()]
    while lines and (re.fullmatch('#{1,6}', lines[-1]) or lines[-1] == '```'):
        lines.pop()
    if not lines:
        return None
    last = lines[-1].strip('*` ')
    last = re.sub(r'\s*#{1,6}\s*$', '', last).strip()
    return number(last)


def main():
    work = HERE / 'work'
    labels = {r['id']: r for r in map(json.loads, (work / 'test.labels.jsonl').read_text().splitlines())}
    seal = json.loads((HERE / 'FINAL_SELECTION.json').read_text())
    table = []
    diagnostic_reports = {}
    for family, policy in seal['comparisons']:
        report = json.loads((work / 'scores/final/test' / f'{family}-{policy}.json').read_text())
        if not report['complete']:
            raise RuntimeError('all comparisons must complete before diagnostic reporting')
        scores = {(r['id'], r['seed']): r for r in report['scores'] if r['domain'] == 'gsm8k'}
        diagnostic = copy.deepcopy(report)
        diagnostic_scores = {(r['id'], r['seed']): r for r in diagnostic['scores']}
        count = {'model': family, 'policy': policy, 'attempts': len(scores),
            'strict_correct': sum(r['correct'] for r in scores.values()),
            'strict_unparseable': sum(not r['numeric_parseable'] for r in scores.values()),
            'recovered_correct_from_explicit_final': 0, 'recovered_wrong_from_explicit_final': 0,
            'unclosed_thinking': 0}
        for path in sorted((work / 'runs/final/test' / family / policy).glob('*.jsonl')):
            for row in map(json.loads, path.read_text().splitlines()):
                key = (row['id'], row['seed'])
                if key not in scores:
                    continue
                trace = row['trace'][-1]
                count['unclosed_thinking'] += int(trace['unclosed_thinking'])
                if scores[key]['numeric_parseable'] or not trace['final']:
                    continue
                readout = conservative_readout(trace['final'])
                if readout is not None:
                    correct = abs(readout - number(labels[row['id']]['gold'])) <= 1e-9
                    count['recovered_correct_from_explicit_final' if correct else 'recovered_wrong_from_explicit_final'] += 1
                    diagnostic_scores[key]['correct'] = bool(correct)
        count['strict_accuracy'] = count['strict_correct'] / count['attempts']
        count['diagnostic_accuracy_after_readout'] = (count['strict_correct'] + count['recovered_correct_from_explicit_final']) / count['attempts']
        count['diagnostic_macro_with_other_domains_unchanged'] = summarize(diagnostic['scores'])['primary_macro']
        diagnostic_reports[family, policy] = diagnostic
        table.append(count)
    result = {'status': 'POST-HOC DIAGNOSTIC ONLY; not a preregistered metric or a selected deployment-policy change',
        'reason': 'Observed reversed #### markers and standalone final numbers; separate interface adherence from numerical content.',
        'method': 'If strict parse fails, take only a standalone numeric last final-channel line after removing trailing heading/fence/emphasis markers. Never mine an unfinished thought or use gold to choose among numbers. Apply identically to all models and arms.',
        'primary_results_unchanged': True, 'table': table,
        'diagnostic_selected_minus_bare': paired(diagnostic_reports['mini', 'bare'], diagnostic_reports['mini', 'literal']),
        'diagnostic_qwen_same_minus_selected': paired(diagnostic_reports['mini', 'literal'], diagnostic_reports['qwen', 'literal']),
        'interpretation': 'The strict mathematics gain must not be called improved numerical reasoning. Explicit-final readout gives MiniCPM bare 72.92% and selected 71.09%; the primary result includes interface adherence. These post-hoc sensitivity estimates do not promote a new deployment adapter or replace sealed metrics.'}
    (HERE / 'numeric_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
