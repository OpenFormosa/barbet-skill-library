#!/usr/bin/env python3
"""Offline scoring: the only evaluation entry point that reads answer files."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from testtime.contracts import validate_call
from testtime.execution import run_python

PRIMARY = ['drcd', 'gsm8k', 'mbpp', 'tools']


def sha(value):
    return hashlib.sha256(value).hexdigest()


def normalize_qa(text):
    # Declared Unicode punctuation/whitespace normalized EM, not an assertion
    # that this implementation is the official DRCD leaderboard evaluator.
    return ''.join(c for c in unicodedata.normalize('NFKC', str(text)).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def qa_f1(pred, gold):
    p, g = Counter(normalize_qa(pred)), Counter(normalize_qa(gold))
    overlap = sum((p & g).values())
    return 2 * overlap / (sum(p.values()) + sum(g.values())) if p or g else 1.


def number(text):
    if text is None:
        return None
    text = str(text).strip()
    if '####' in text:
        text = text.rsplit('####', 1)[-1].strip()
    text = re.sub(r'^\\boxed\{([^{}]+)\}$', r'\1', text).strip().strip('$').replace(',', '')
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?', text):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def typed_equal(a, b):
    # Numeric schema accepts int/float equivalence but never bool/integer.
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def grade(row, label, request):
    pred, gold, domain = row['prediction'], label['gold'], label['domain']
    detail = {}
    if domain == 'drcd':
        correct = any(normalize_qa(pred or '') == normalize_qa(g) for g in gold)
        detail['character_f1'] = max(qa_f1(pred or '', g) for g in gold)
        support = label['sufficient_document_id']
        detail['support_retrieved'] = support in {d['id'] for d in row.get('retrieved', [])}
        detail['support_visible'] = support in {d['id'] for d in row.get('selected_sources', [])}
        detail['span_verified'] = any('source_span_verified' in t for t in row['trace'])
    elif domain == 'gsm8k':
        p, g = number(pred), number(gold)
        correct = p is not None and g is not None and abs(p - g) <= Fraction(1, 10**9)
        detail['numeric_parseable'] = p is not None
        detail['program_executed'] = any(t.get('execution', {}).get('passed') for t in row['trace'])
    elif domain == 'mbpp':
        receipt = run_python(pred or '', request.get('public_tests', []) + gold['tests'], gold['setup'])
        correct = receipt['passed']
        detail.update(execution_status=receipt['status'], execution_error=receipt.get('error'),
                      scoring_seconds=receipt.get('seconds', 0))
        detail['public_pass_before_scoring'] = any(t.get('execution', {}).get('passed') for t in row['trace'])
    elif domain == 'tools':
        valid = not validate_call(pred, request['tools'])
        correct = valid and typed_equal(pred, gold)
        detail['schema_valid'] = valid
        detail['wrong_name'] = not isinstance(pred, dict) or pred.get('name') != gold['name']
        detail['wrong_arguments'] = ([] if correct else [k for k, v in gold['arguments'].items()
            if not isinstance(pred, dict) or not isinstance(pred.get('arguments'), dict)
            or not typed_equal(pred['arguments'].get(k), v)])
    elif domain == 'tmmluplus':
        answer = str(pred or '').strip()
        correct = answer == gold
        detail['letter_parseable'] = bool(re.fullmatch('[ABCD]', answer))
    else:
        raise ValueError('unknown scoring domain')
    return {'id': row['id'], 'domain': domain, 'group': label['group'], 'model': row['model'],
        'policy': row['policy'], 'seed': row['seed'], 'correct': bool(correct), **detail,
        'calls': row['calls'], 'prompt_tokens': row['prompt_tokens'], 'output_tokens': row['output_tokens'],
        'generation_seconds_amortized': row['amortized_generation_seconds'],
        'retrieval_seconds': row['retrieval_seconds'], 'execution_seconds': row['execution_seconds'],
        'unclosed_thinking_calls': sum(t['unclosed_thinking'] for t in row['trace']),
        'length_stopped_calls': sum(t['finish_reason'] == 'length' for t in row['trace'])}


def summarize(scores):
    grouped = defaultdict(list)
    for row in scores:
        grouped[row['domain']].append(row)
    domains = {d: {'n': len(v), 'correct': sum(r['correct'] for r in v),
                  'accuracy': sum(r['correct'] for r in v) / len(v)} for d, v in grouped.items()}
    n = len(scores)
    return {'items': n, 'domains': domains,
        'primary_macro': sum(domains[d]['accuracy'] for d in PRIMARY) / len(PRIMARY) if all(d in domains for d in PRIMARY) else None,
        'cost': {k: sum(r[k] for r in scores) / max(1, n) for k in ['calls', 'prompt_tokens', 'output_tokens',
                  'generation_seconds_amortized', 'retrieval_seconds', 'execution_seconds']},
        'unclosed_thinking_calls': sum(r['unclosed_thinking_calls'] for r in scores),
        'length_stopped_calls': sum(r['length_stopped_calls'] for r in scores),
        'qa_support_visible_rate': sum(r.get('support_visible', False) for r in scores) / max(1, len(grouped['drcd'])),
        'secondary': {
            'qa_character_f1': sum(r.get('character_f1', 0) for r in scores) / max(1, len(grouped['drcd'])),
            'tool_schema_valid_rate': sum(r.get('schema_valid', False) for r in scores) / max(1, len(grouped['tools'])),
            'math_numeric_parseable_rate': sum(r.get('numeric_parseable', False) for r in scores) / max(1, len(grouped['gsm8k'])),
            'code_public_pass_before_final_rate': sum(r.get('public_pass_before_scoring', False) for r in scores) / max(1, len(grouped['mbpp']))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev', 'validation', 'test'], required=True)
    p.add_argument('--run-tag', default='main')
    p.add_argument('--work', type=Path, default=HERE / 'work')
    p.add_argument('--models', nargs='*')
    p.add_argument('--policies', nargs='*')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    if args.split == 'test' and not (HERE / 'FINAL_SELECTION.json').exists():
        raise RuntimeError('test scoring requires final selection seal')
    label_path = args.work / f'{args.split}.labels.jsonl'
    labels = {r['id']: r for r in map(json.loads, label_path.read_text().splitlines())}
    inputs = {r['id']: r for r in map(json.loads, (args.work / f'{args.split}.inputs.jsonl').read_text().splitlines())}
    base = args.work / 'runs' / args.run_tag / args.split
    runs = defaultdict(list)
    for path in sorted(base.glob('*/*/*.jsonl')):
        family, policy = path.parent.parent.name, path.parent.name
        if args.models and family not in args.models or args.policies and policy not in args.policies:
            continue
        runs[(family, policy)].append(path)
    reports = []
    for (family, policy), paths in runs.items():
        rows, provenance = [], []
        for path in paths:
            part = [json.loads(x) for x in path.read_text().splitlines()]
            metadata = json.loads(path.with_suffix('.runtime.json').read_text())
            if any(r['run_fingerprint'] != metadata['run_fingerprint'] for r in part):
                raise RuntimeError('raw trajectory provenance mismatch')
            completed_path = path.with_suffix('.completed.json')
            if completed_path.exists() and json.loads(completed_path.read_text())['raw_sha256'] != sha(path.read_bytes()):
                raise RuntimeError('completed trajectory file changed')
            rows += part
            provenance.append({'raw': str(path.relative_to(args.work)), 'sha256': sha(path.read_bytes()),
                'runtime': metadata, 'complete': path.with_suffix('.completed.json').exists()})
        keys = [(r['id'], r['seed']) for r in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError('duplicate id/seed across shards')
        expected = {(i, meta['runtime']['seed']) for meta in provenance for i in meta['runtime']['request_ids']}
        complete = set(keys) == expected and all(x['complete'] for x in provenance)
        if not complete and not args.allow_partial:
            print(json.dumps({'event': 'skip_incomplete', 'model': family, 'policy': policy}), flush=True)
            continue
        # Changes to scorer or raw rows invalidate the cache explicitly.
        score_fingerprint = sha(json.dumps({'scorer': sha(Path(__file__).read_bytes()),
            'worker': sha((ROOT / 'testtime/_worker.py').read_bytes()),
            'executor': sha((ROOT / 'testtime/execution.py').read_bytes()),
            'contracts': sha((ROOT / 'testtime/contracts.py').read_bytes()),
            'labels': sha(label_path.read_bytes()), 'raw': [x['sha256'] for x in provenance]}, sort_keys=True).encode())
        out = args.work / 'scores' / args.run_tag / args.split / f'{family}-{policy}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and json.loads(out.read_text()).get('score_fingerprint') == score_fingerprint:
            result = json.loads(out.read_text())
        else:
            with ThreadPoolExecutor(max_workers=12) as pool:
                scores = list(pool.map(lambda r: grade(r, labels[r['id']], inputs[r['id']]), rows))
            result = {'model': family, 'policy': policy, 'split': args.split,
                'complete': complete, 'score_fingerprint': score_fingerprint,
                'summary': summarize(scores), 'scores': scores, 'provenance': provenance}
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        reports.append({'model': family, 'policy': policy, 'complete': complete, **result['summary']})
    summary_path = args.work / 'scores' / args.run_tag / f'{args.split}-summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
