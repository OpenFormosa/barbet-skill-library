"""First, immutable diagnosis across NI task categories; never rewrites answers."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

from scripts import audit_memory_sft_v2 as reviewer
from scripts.build_natural_instructions_memory_v1 import STATE
from scripts import build_memory_sft_v2 as base


def prepare():
    path = STATE / 'semantic-audit-v1/plan.json'
    if path.exists(): return json.loads(path.read_text())
    pools = defaultdict(list)
    for p in sorted((STATE / 'normalized').glob('pilot-*.jsonl')):
        if p.name.endswith('.excluded.jsonl'): continue
        for line in p.open():
            row = json.loads(line)
            for category in row['source_refs']['categories']:
                pool = pools[category]
                pool.append((base.sha('ni-category-audit:' + row['record_id']), row))
                if len(pool) > 8:
                    pool.sort(key=lambda x: x[0]); del pool[2:]
    selected = {}; categories = defaultdict(list)
    for category, values in sorted(pools.items()):
        # At most one answer per input in this source diagnosis. All versions of
        # a first non-accept query will remain held for the later PPL pipeline.
        used = set()
        for _, row in sorted(values, key=lambda x: x[0]):
            if row['source_task_key'] in used: continue
            used.add(row['source_task_key']); rid = row['record_id']
            categories[rid].append(category); selected[rid] = row
            if len(used) == 2: break
    jobs = [{'id': 'sample-' + rid, 'record_id': rid, 'domain': row['domain'],
             'messages': row['messages'], 'tools': row['tools'],
             'observation_origin': row['observation_origin'], 'categories': categories[rid]}
            for rid, row in sorted(selected.items())]
    controls = [
        {'id': 'control-wrong-number', 'domain': 'math', 'expected': 'not_accept', 'messages': [
            {'role': 'user', 'content': 'What is 2 + 3?'}, {'role': 'assistant', 'content': '6', 'loss': True}]},
        {'id': 'control-unsupported-success', 'domain': 'tool_use', 'expected': 'not_accept', 'messages': [
            {'role': 'user', 'content': 'Look up the order status. No tool is available and no result is provided.'},
            {'role': 'assistant', 'content': 'I checked the database; your order has shipped.', 'loss': True}]},
        {'id': 'control-positive-number', 'domain': 'math', 'expected': 'accept', 'messages': [
            {'role': 'user', 'content': 'What is 2 + 3?'}, {'role': 'assistant', 'content': '5', 'loss': True}]},
        {'id': 'control-positive-evidence', 'domain': 'reading', 'expected': 'accept', 'messages': [
            {'role': 'user', 'content': 'Document: The blue tower has 21 floors, not 31. How many floors does the blue tower have?'},
            {'role': 'assistant', 'content': '21', 'loss': True}]},
    ]
    plan = {'schema_sha256': reviewer.digest(reviewer.SCHEMA), 'prompt_sha256': reviewer.digest(reviewer.PROMPT),
        'model': reviewer.MODEL, 'revision': reviewer.REV, 'controls': controls, 'jobs': jobs,
        'scope': 'Deterministic source-category pilot, up to two different task inputs per category. Not representative corpus accuracy, all-row approval or a review of the later minimum-PPL winners.',
        'first_non_accept_is_final_for_that_record': True}
    reviewer.dump(path, plan); return plan


def main():
    if not (STATE / 'pilot-normalize-summary.json').exists(): raise ValueError('NI pilot incomplete')
    reviewer.STATE = STATE; plan = prepare()
    with ThreadPoolExecutor(2) as pool: controls = list(pool.map(reviewer.review, plan['controls']))
    passed = all((r['verdict'] == 'ACCEPT') == (c['expected'] == 'accept') for r, c in zip(controls, plan['controls']))
    reviewer.dump(STATE / 'semantic-audit-v1/controls.json', {'passed': passed, 'results': controls})
    if not passed: raise ValueError('NI first-audit controls failed; do not start samples')
    with ThreadPoolExecutor(2) as pool: results = list(pool.map(reviewer.review, plan['jobs']))
    reviewer.dump(STATE / 'semantic-audit-v1/summary.json', {'controls': controls, 'results': results,
        'counts': dict(Counter(r['verdict'] for r in results)),
        'excluded_record_ids': [r['record_id'] for r in results if r['verdict'] != 'ACCEPT'],
        'scope': plan['scope']})


if __name__ == '__main__': main()
