#!/usr/bin/env python3
"""Summarize preconstructed source/position controls with question-level rows."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    base = HERE / 'work/context_probe'
    audit = json.loads((base / 'audit.json').read_text())
    table = []
    for path in sorted((base / 'scores/main/dev').glob('*-context_probe.json')):
        report = json.loads(path.read_text())
        groups = defaultdict(list)
        for row in report['scores']:
            groups[row['id'].split('/')[0]].append(row)
        for condition, values in sorted(groups.items()):
            scores = np.array([v['correct'] for v in values], dtype=float)
            rng = np.random.default_rng(20261027)
            boot = scores[rng.integers(len(scores), size=(10000, len(scores)))].mean(axis=1)
            table.append({'model': report['model'], 'condition': condition, 'questions': len(values),
                'accuracy': float(scores.mean()), 'accuracy_95ci': [float(x) for x in np.quantile(boot, [.025, .975])],
                'character_f1': sum(v['character_f1'] for v in values) / len(values),
                'mean_prompt_tokens': sum(v['prompt_tokens'] for v in values) / len(values),
                'mean_output_tokens': sum(v['output_tokens'] for v in values) / len(values),
                'unclosed_calls': sum(v['unclosed_thinking_calls'] for v in values)})
    result = {'stage': 'development diagnostic, one generation seed; not used as final-test evidence',
        'task_composition': audit['question_count'], 'excluded': audit['excluded'],
        'randomness_notice': audit['randomness_notice'], 'source_control': audit['source_control'],
        'table': table}
    (HERE / 'context_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'conditions_scored': len(table), 'models': sorted({v['model'] for v in table})}))


if __name__ == '__main__':
    main()
