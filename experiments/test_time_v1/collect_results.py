#!/usr/bin/env python3
"""Collect completed evidence for an offline dashboard, without inventing runs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def maybe(name, default=None):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else default


def main():
    result = {'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'final_evaluation' if (HERE / 'FINAL_SELECTION.json').exists() else 'development_validation',
        'test_status': 'selection sealed' if (HERE / 'FINAL_SELECTION.json').exists() else 'unopened',
        'decisions': maybe('decision_ledger.json', []),
        'final_selection': maybe('FINAL_SELECTION.json'),
        'comparisons': maybe('final_comparisons.json'),
        'diagnostics': maybe('final_diagnostics.json'),
        'engineering': maybe('engineering_results.json'),
        'context': maybe('context_results.json'),
        'code_budget': maybe('code_budget_results.json'),
        'stopping': maybe('selection_decision.json'),
        'job_chain': maybe('work/final_job_schedule.json', {}).get('jobs', [])}
    for key, tag, split in [('dev', 'main', 'dev'), ('validation', 'main', 'validation'),
                             ('validation_confirm', 'confirm', 'validation'), ('test', 'final', 'test')]:
        path = HERE / 'work/scores' / tag / f'{split}-summary.json'
        result[key] = json.loads(path.read_text()) if path.exists() else []
    if (result['comparisons'] and result['final_selection']
            and len(result['test']) == len(result['final_selection']['comparisons'])
            and all(r['complete'] for r in result['test'])):
        result['status'] = 'completed'
        result['test_status'] = 'all sealed comparisons completed; no post-test policy selection'
    # Stable preselected development gallery, never hand-picked after seeing a win.
    probes = maybe('probe_selection.json', {'ids': {}})['ids']
    ids = {i for group in probes.values() for i in group}
    inputs = {r['id']: r for r in map(json.loads, (HERE / 'work/dev.inputs.jsonl').read_text().splitlines())}
    labels = {r['id']: r for r in map(json.loads, (HERE / 'work/dev.labels.jsonl').read_text().splitlines())}
    gallery = {ident: {'id': ident, 'domain': labels[ident]['domain'],
        'public_task_excerpt': inputs[ident]['prompt'][:180],
        'reference': '[hidden code checks omitted]' if labels[ident]['domain'] == 'mbpp' else labels[ident]['gold'],
        'trajectories': []} for ident in sorted(ids)}
    selected = (result['stopping'] or {}).get('selected_policy', 'routed')
    for policy in dict.fromkeys(['bare', 'routed', selected]):
        p = HERE / 'work/runs/main/dev/mini' / policy / '20261027-s0.jsonl'
        if not p.exists():
            continue
        for row in map(json.loads, p.read_text().splitlines()):
            if row['id'] in gallery:
                gallery[row['id']]['trajectories'].append({'policy': policy,
                    'prediction': row['prediction'], 'calls': row['calls'],
                    'prompt_tokens': row['prompt_tokens'], 'output_tokens': row['output_tokens'],
                    # Do not redistribute the retrieved article corpus/prompt.
                    'trace': [{k: v for k, v in t.items() if k not in {'prompt', 'sampling'}} for t in row['trace']]})
    result['fixed_dev_gallery'] = list(gallery.values())
    dest = ROOT / 'docs/results/test-time'
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted((HERE / 'work/scores/final/test').glob('*.json')):
        report = json.loads(path.read_text())
        # Scores/provenance only: generated prompts and hidden checks stay local.
        (dest / path.name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'collected_final_arms': len(result['test']), 'status': result['status']}))


if __name__ == '__main__':
    main()
