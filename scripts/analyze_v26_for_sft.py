"""Read frozen v26 receipts and emit aggregate SFT design evidence.

No inference, training, dataset rewriting, or example text export. IFEval decisions
are read from the historical per-item scoring ledger, not recomputed here.
Optional Parquet inspection needs pyarrow; the default path is stdlib-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def keyed(values):
    result = {}
    for value in values:
        if value['id'] in result:
            raise ValueError('duplicate receipt id')
        result[value['id']] = value
    return result


def numeric(text):
    text = text.replace(',', '')
    if '####' in text:
        values = re.findall(r'[-+]?\d+(?:\.\d+)?', text.rsplit('####', 1)[-1])
        value = values[0] if values else None
    else:
        boxes = re.findall(r'\\boxed\{([^{}]+)\}', text)
        values = re.findall(r'[-+]?\d+(?:\.\d+)?', boxes[-1] if boxes else text)
        value = values[-1] if values else None
    return str(Decimal(value)) if value is not None else 'None'


def typed_equal(a, b):
    if isinstance(b, bool):
        return isinstance(a, bool) and a == b
    if isinstance(b, (int, float)):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def unique_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('duplicate key')
        out[k] = v
    return out


def decode_call(text):
    text = text.strip()
    for start, end in [('<|tool_call_start|>', '<|tool_call_end|>'), ('<tool_call>', '</tool_call>')]:
        if start in text:
            text = text.split(start, 1)[1].split(end, 1)[0].strip()
            break
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    obj = json.loads(text, object_pairs_hook=unique_pairs)
    if isinstance(obj, list):
        if len(obj) != 1:
            raise ValueError('not one call')
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError('not an object')
    if 'tool_calls' in obj:
        obj = obj['tool_calls']
        if not isinstance(obj, list) or len(obj) != 1:
            raise ValueError('not one call')
        obj = obj[0]
    if 'function' in obj:
        obj = obj['function']
    args = obj['arguments']
    if isinstance(args, str):
        args = json.loads(args, object_pairs_hook=unique_pairs)
    if not isinstance(args, dict):
        raise ValueError('not arguments')
    return {'name': obj['name'], 'arguments': args}


def candidate_audit(path, original):
    import pyarrow.parquet as pq

    counts, skills, sources, modes, role_patterns = (Counter() for _ in range(5))
    roots, groups, record_ids = set(), set(), set()
    skill_roots = defaultdict(set)
    skill_tokens = Counter()
    total, supervised = [], []
    columns = ['record_id', 'root_task_id', 'source_group', 'split', 'reasoning_mode',
               'messages', 'assistant_loss', 'skill_names', 'upstream_id',
               'total_tokens', 'supervised_tokens']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            if row['record_id'] in record_ids or row['split'] != 'train':
                raise ValueError('duplicate id or unexpected split in candidate snapshot')
            record_ids.add(row['record_id'])
            roots.add(row['root_task_id'])
            groups.add(row['source_group'])
            sources[row['upstream_id']] += 1
            modes[row['reasoning_mode']] += 1
            total.append(row['total_tokens'])
            supervised.append(row['supervised_tokens'])
            for skill in row['skill_names']:
                skills[skill] += 1
                skill_roots[skill].add(row['root_task_id'])
                skill_tokens[skill] += row['supervised_tokens']
            messages, keep = row['messages'], row['assistant_loss']
            if len(messages) != len(keep):
                raise ValueError('turn-mask length mismatch')
            counts['rows_with_tool_observation'] += any(m['role'] == 'tool' for m in messages)
            counts['rows_with_multiple_approved_assistant_turns'] += sum(
                bool(k) and m['role'] == 'assistant' for m, k in zip(messages, keep)) > 1
            for i, (message, loss) in enumerate(zip(messages, keep)):
                if not loss:
                    continue
                counts['loss_on_non_assistant_turns'] += message['role'] != 'assistant'
                if message['role'] != 'assistant':
                    continue
                counts['supervised_assistant_points'] += 1
                counts['has_tool_calls'] += bool(message.get('tool_calls'))
                content = message.get('content') or ''
                counts['content_dash_only'] += content.strip() == '---'
                previous = messages[i - 1] if i else {}
                counts['immediately_preceded_by_skill_system'] += (
                    previous.get('role') == 'system'
                    and '[BARBET_SKILL_CONTEXT_V1]' in (previous.get('content') or ''))
                role_patterns['>'.join(m['role'] for m in messages[max(0, i - 2):i + 1])] += 1
    if len(record_ids) != original['static_pass_train_rows'] or dict(skills) != original['skill_row_counts']:
        raise ValueError('candidate snapshot differs from the frozen audit')
    return {
        'source': {'repo': original['repo'], 'revision': original['revision'], 'parquet_sha256': digest(path)},
        'rows': len(record_ids), 'distinct_root_task_ids': len(roots), 'distinct_source_groups': len(groups),
        'counts': dict(counts), 'reasoning_modes': dict(modes), 'source_row_counts': dict(sources),
        'preceding_role_patterns': dict(role_patterns),
        'stored_token_statistics': {
            'total_tokens': sum(total), 'supervised_tokens': sum(supervised),
            'supervised_fraction': sum(supervised) / sum(total),
            'median_total_tokens': statistics.median(total),
            'median_supervised_tokens': statistics.median(supervised),
            'scope': 'Stored row counts; this analysis does not retokenize every row or verify trainer consumption.'},
        'skill_coverage': {k: {'rows': skills[k], 'distinct_root_task_ids': len(skill_roots[k]),
                             'associated_row_supervised_tokens': skill_tokens[k]} for k in sorted(skills)},
        'coverage_note': 'Skill tags overlap. Associated row tokens are not tokens exclusively attributable to that skill.',
        'limit': original['limit'],
    }


def analyze(root, parquet=None):
    inputs = {}

    def read(name, lines=False):
        path = root / name
        inputs[name] = digest(path)
        return rows(path) if lines else json.loads(path.read_text())

    base = 'diagnostics/failure_20261024/'
    native = base + 'native_rerun/'
    combined = native + 'combined/'
    questions = keyed(read(native + 'questions.jsonl', lines=True))
    raw = keyed([row for p in sorted((root / native).glob('barbet-v26-s[0-9].jsonl'))
                 for row in read(str(p.relative_to(root)), lines=True)])
    scores = keyed([x for x in read(combined + 'primary-scored_items.jsonl', lines=True)
                    if x['model'] == 'barbet-v26'])
    if set(questions) != set(raw) or set(raw) != set(scores) or len(raw) != 514:
        raise ValueError('missing, duplicate, or unexpected evaluation items')
    original_raw = [r for p in sorted((root / 'runs/20261024-v26').glob('barbet-v26-s[0-9].jsonl'))
                    for r in read(str(p.relative_to(root)), lines=True)]
    if set(keyed(original_raw)) != set(questions):
        raise ValueError('original/native identity mismatch')
    primary = read(combined + 'primary-scores.json')
    protocol = read(native + 'protocol.json')
    if inputs[native + 'questions.jsonl'] != protocol['questions_sha256']:
        raise ValueError('question hash differs from frozen protocol')
    model = read(native + 'barbet-v26.json')
    execution = read(native + 'execution_parameters.json')
    read(combined + 'combination_audit.json')
    read(combined + 'native_parser_provenance.json')
    old_training = read(base + 'training_snapshot_audit.json')
    old_errors = read(base + 'native_error_analysis.json')
    probes = read(base + 'probe_scores.json')
    inputs['score.py'] = digest(root / 'score.py')
    inputs['prepare_data.py'] = digest(root / 'prepare_data.py')
    tasks, tools, args = defaultdict(Counter), Counter(), Counter()
    tool_families, instruction_checks = defaultdict(Counter), defaultdict(Counter)
    choices, math = Counter(), Counter()
    for id, s in scores.items():
        r, q = raw[id], questions[id]
        if r['task'] != q['task'] or s['task'] != q['task'] or r['output_tokens'] != s['output_tokens']:
            raise ValueError('receipt and score mismatch')
        t = tasks[q['task']]
        t['n'] += 1
        t['correct'] += bool(s['correct'])
        t['length_capped'] += r['finish_reason'] == 'length'
        if 'format_ok' in s:
            t['format_valid'] += s['format_ok']
        text = r['answer_text'].strip()
        if q['task'] == 'gsm8k':
            prediction = numeric(text)
            if prediction != s['prediction'] or (prediction == numeric('#### ' + q['gold'])) != s['correct']:
                raise ValueError('numeric rescore disagrees with ledger')
            math['numeric_parseable'] += prediction != 'None'
            math['hash_and_whitespace_only'] += bool(re.fullmatch(r'[#\s]+', text))
            math['literal_2'] += text == '2'
        if q['task'] == 'tmmluplus':
            choices[str(s['prediction'])] += 1
            t['parseable_option'] += s['prediction'] is not None
            t['correct_with_strict_letter_format'] += bool(s['correct'] and s['format_ok'])
        if q['task'] == 'ifeval':
            ids = q['gold']['instruction_id_list']
            if len(ids) != len(s['instruction_strict']):
                raise ValueError('instruction-check length mismatch')
            for name, good in zip(ids, s['instruction_strict']):
                instruction_checks[name]['n'] += 1
                instruction_checks[name]['correct'] += good
        if q['task'] != 'fresh_tools':
            continue
        family = tool_families[q['family']]
        family['n'] += 1
        family['correct'] += s['correct']
        try:
            call = decode_call(text)
        except (ValueError, TypeError, KeyError):
            tools['no_single_parseable_call'] += 1
            if s['format_ok'] or s['correct']:
                raise ValueError('call parse disagrees with score')
            continue
        tools['one_parseable_call'] += 1
        if typed_equal(call, q['gold']) != s['correct']:
            raise ValueError('typed call rescore disagrees with ledger')
        if call['name'] != q['gold']['name']:
            tools['wrong_tool_name'] += 1
            continue
        tools['right_tool_name'] += 1
        actual, expected = call['arguments'], q['gold']['arguments']
        if typed_equal(actual, expected):
            tools['fully_correct'] += 1
            continue
        tools['right_tool_wrong_arguments'] += 1
        args['missing_expected_fields'] += not set(expected).issubset(actual)
        args['extra_fields'] += bool(set(actual) - set(expected))
        for key, value in expected.items():
            if key not in actual or typed_equal(actual[key], value):
                continue
            kind = ('boolean' if isinstance(value, bool) else 'numeric'
                    if isinstance(value, (int, float)) else 'array' if isinstance(value, list) else 'string_or_enum')
            args['wrong_' + kind + '_value_or_type'] += 1
    for task, counts in tasks.items():
        saved = primary['models']['barbet-v26']['tasks'][task]
        for key in ['n', 'correct', 'length_capped']:
            if counts[key] != saved[key]:
                raise ValueError('aggregate differs from frozen summary')
    for k, v in tools.items():
        if v != old_errors['tool_counts'][k]:
            raise ValueError('tool decomposition differs from frozen diagnostic')
    result = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'READ_ONLY_REANALYSIS_NO_NEW_MODEL_RUN',
        'model': {k: model[k] for k in ['repo', 'requested_revision', 'revision']},
        'scope': '514 fixed diagnostic items. Numeric and tool results independently reconciled against raw outputs; IFEval and MC judgments use the frozen item ledger. Not a new hidden test or full official benchmark.',
        'native_task_metrics': {k: dict(v) for k, v in tasks.items()},
        'frozen_model_comparison': primary['models'],
        'interface_dash_only': {'original': sum(r['answer_text'].strip() == '---' for r in original_raw),
                                'native': sum(r['answer_text'].strip() == '---' for r in raw.values())},
        'math_diagnostics': dict(math), 'choice_predictions': dict(choices),
        'tool_counts': dict(tools), 'argument_errors_among_right_tool_wrong_arguments_overlap': dict(args),
        'tool_families': {k: dict(v) for k, v in tool_families.items()},
        'native_ifeval_by_instruction': {k: dict(v) for k, v in instruction_checks.items()},
        'small_control': old_errors['single_skill_control'],
        'toy_native_probes': {k: probes[k] for k in ['training_native', 'training_native_tools']},
        'execution_protocol': execution,
        'candidate_snapshot_historical_audit': {k: old_training[k] for k in
            ['repo', 'revision', 'static_pass_train_rows', 'counts', 'unique_skills', 'skill_row_counts', 'limit']},
        'limitations': [
            'The candidate dataset is not proof of the data actually consumed by the private v26 training job.',
            'OpenFormosa/barbet-long-context-sft is a separate dataset with a separate 1M-token contract.',
            'No predecessor checkpoint, optimizer history, actual trainer masks or causal-shift implementation were audited.',
            'Existing v26 and MiniCPM test items must be excluded from new training and model selection; derive failure families, not their answers.',
            'Skill-tag counts and syntactic checks do not certify semantic correctness or independent task diversity.',
        ],
    }
    if parquet:
        result['candidate_snapshot_reinspection'] = candidate_audit(parquet, old_training)
    result['source_sha256'] = inputs
    result['analysis_script_sha256'] = digest(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--candidate-train-parquet', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.benchmark_root, args.candidate_train_parquet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'questions': 514, 'status': result['status']}))


if __name__ == '__main__':
    main()
