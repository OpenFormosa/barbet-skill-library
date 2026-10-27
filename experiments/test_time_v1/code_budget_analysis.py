#!/usr/bin/env python3
"""Evaluation-only replay of frozen code trajectories at 1/2/3-call ceilings.

No additional model calls or policy selection. Prefix replay shares the exact
same first attempts; it is not a fresh serving benchmark with a new batch mix.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from testtime.harness import python_body
from compare import paired
from score import grade, summarize


def main():
    chosen = json.loads((HERE / 'selection_decision.json').read_text())['selected_policy']
    source = HERE / 'work'
    final = json.loads((source / 'scores/final/test' / f'mini-{chosen}.json').read_text())
    if not final['complete']:
        raise RuntimeError('selected final arm is incomplete')
    labels = {r['id']: r for r in map(json.loads, (source / 'test.labels.jsonl').read_text().splitlines())}
    requests = {r['id']: r for r in map(json.loads, (source / 'test.inputs.jsonl').read_text().splitlines())}
    rows = [r for path in sorted((source / 'runs/final/test/mini' / chosen).glob('*.jsonl'))
            for r in map(json.loads, path.read_text().splitlines()) if labels[r['id']]['domain'] == 'mbpp']
    final_scores = {(r['id'], r['seed']): r for r in final['scores'] if r['domain'] == 'mbpp'}
    cache = {(r['id'], r['seed'], r['prediction']): final_scores[r['id'], r['seed']] for r in rows}
    def cached_grade(row):
        key = (row['id'], row['seed'], row['prediction'])
        if key not in cache:
            cache[key] = grade(row, labels[row['id']], requests[row['id']])
        scored = dict(cache[key])
        scored.update(policy=row['policy'], calls=row['calls'], prompt_tokens=row['prompt_tokens'],
            output_tokens=row['output_tokens'], generation_seconds_amortized=row['amortized_generation_seconds'],
            execution_seconds=row['execution_seconds'], retrieval_seconds=row['retrieval_seconds'],
            unclosed_thinking_calls=sum(t['unclosed_thinking'] for t in row['trace']),
            length_stopped_calls=sum(t['finish_reason']=='length' for t in row['trace']),
            public_pass_before_scoring=any(t.get('execution', {}).get('passed') for t in row['trace']))
        return scored
    reports = {}
    for ceiling in [1, 2, 3]:
        prefix_rows = []
        for original in rows:
            row = copy.deepcopy(original)
            row['trace'] = row['trace'][:ceiling]
            row['prediction'] = python_body(row['trace'][-1]['final'])
            row['policy'] = f'code_prefix_{ceiling}'
            row['calls'] = len(row['trace'])
            for key in ['prompt_tokens', 'output_tokens', 'amortized_generation_seconds']:
                row[key] = sum(t[key] for t in row['trace'])
            row['execution_seconds'] = sum(t.get('execution', {}).get('seconds', 0) for t in row['trace'])
            prefix_rows.append(row)
        with ThreadPoolExecutor(max_workers=12) as pool:
            scores = list(pool.map(cached_grade, prefix_rows))
        reports[ceiling] = {'model': 'mini', 'policy': f'code_prefix_{ceiling}',
            'scores': scores, 'summary': summarize(scores)}
    result = {'method': 'Replay prefixes of the final selected code trajectories; same exact initial generations. Reuse recorded final grades for identical question/seed/code bytes; grade each additional artifact once. Actual serving at a smaller call ceiling may differ through BF16 batch effects.',
        'does_not_select_a_new_policy': True,
        'unit': '128 held-out code questions, three seeds; public checks only during generation, hidden checks only here',
        'table': [{'call_ceiling': k, **r['summary']} for k, r in reports.items()],
        'two_minus_one': paired(reports[1], reports[2]),
        'three_minus_two': paired(reports[2], reports[3]),
        'three_minus_one': paired(reports[1], reports[3])}
    (HERE / 'code_budget_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'code_accuracy_by_ceiling': {k:r['summary']['domains']['mbpp']['accuracy'] for k,r in reports.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
