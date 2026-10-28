"""Inspect fixed stratified full records and exact training/inference boundaries."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import audit_barbet_continuation_v1 as audit
from scripts import publish_barbet_only_v1 as release
from barbet_inference.sft import normalize_messages, prepare_tools, pangolin_render, tokenize_spans
from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.memory import canonical


def inspect_file(job):
    path, pointers = job
    indices = [p['row'] for p in pointers]
    records = pq.read_table(release.parent.DEST / path).take(pa.array(indices)).to_pylist()
    tokens = pq.read_table(release.parent.DEST / path.replace('/records/', '/tokenized/')).take(pa.array(indices)).to_pylist()
    results = []; counts = Counter(); packets = []
    for pointer, row, token in zip(pointers, records, tokens):
        assert row['record_id'] == token['record_id'] == pointer['record_id']
        messages = json.loads(row['messages_json']); tools = json.loads(row['tools_json'])
        refs = json.loads(row['source_refs_json']); spans = json.loads(row['target_spans_json'])
        checked_tools, validators = prepare_tools(tools)
        normalized = normalize_messages(messages, checked_tools, validators)
        assert checked_tools == tools and normalized == messages
        text, actual_spans = pangolin_render(messages, tools)
        assert text == row['text'] and [list(s) for s in actual_spans] == spans
        ids, labels = tokenize_spans(audit.TOKENIZER, text, actual_spans)
        assert ids == token['input_ids'] and labels == token['labels']
        native = audit.NATIVE.render(messages, tools)
        result = {'record_id': row['record_id'], 'stratum': pointer['stratum'],
                  'all_role_and_typed_tool_contracts_valid': True,
                  'complete_training_text_tokens_labels_exact': True,
                  'native_template_equals_exported_training': native['text_sha256'] == audit.digest(text),
                  'targets_checked': len(spans), 'source': row['source'], 'domain': row['domain']}
        counts['records'] += 1; counts['source:' + row['source']] += 1
        counts['domain:' + row['domain']] += 1
        counts['native_template_matches' if result['native_template_equals_exported_training'] else 'portable_template_differs_from_native'] += 1
        for left, right, turn in spans:
            request = prepare_dataset_request(row, model='audit-only', mode='pangolin_completion', target_turn=turn)
            assert request['prompt'] == text[:left] and not request['add_special_tokens']
            chat = prepare_dataset_request(row, model='audit-only', target_turn=turn)
            expected_roles = [m['role'] for m in messages[:turn]]
            assert [m['role'] for m in chat['messages']] == expected_roles
            assert len(chat['messages']) == turn
            counts['target_prefixes_checked'] += 1
        selection = refs['answer_selection']
        assert canonical(selection) not in text
        counts['selection_metadata_not_in_prompt'] += 1
        if row['source'].startswith(('Wikidata5M/', 'ConceptNet/')):
            memory = restore_relational_memory(row)
            expected = memory.evaluate(refs['query'])['target']
            final = next(m for m in reversed(messages) if m['role'] == 'assistant' and m['loss'])
            assert json.loads(final['content']) == expected
            pending = {}
            for message in messages:
                for call in message.get('tool_calls', []):
                    args = call['function']['arguments']; assert call['function']['name'] == 'relation_lookup'
                    pending[call['id']] = memory.lookup(**args)
                if message['role'] == 'tool':
                    assert json.loads(message['content']) == pending.pop(message['tool_call_id'])
                    counts['actual_relation_observations_restored'] += 1
            assert not pending
            counts['exact_relation_targets_checked'] += 1
        # Full packets remain private; some contain long quoted source documents.
        packets.append({'record_id': row['record_id'], 'source': row['source'], 'domain': row['domain'],
                        'split': row['split'], 'messages': messages, 'tools': tools,
                        'answer_selection': selection, 'memory_placement': row['memory_placement']})
        results.append(result)
    return {'counts': dict(counts), 'results': results, 'packets': packets}


if __name__ == '__main__':
    if (audit.STATE / 'content-sample-summary.json').exists():
        raise ValueError('First stratified content inspection already complete')
    pointers = release.read(audit.STATE / 'sample-pointers.json')
    grouped = defaultdict(list)
    for pointer in pointers: grouped[pointer['path']].append(pointer)
    release.base.save(audit.STATE / 'content-policy.json', {
        'driver_sha256': release.base.file_sha(Path(__file__)),
        'pointer_sha256': release.base.file_sha(audit.STATE / 'sample-pointers.json'),
        'structural_and_exact_query_checks_not_general_semantic_accuracy': True,
        'complete_sample_packets_retained_locally': True, 'teacher_requests': 0,
        'first_holds_and_model_weights_unchanged': True, 'workers': 6})
    counts = Counter(); results = []; packets = []
    with ProcessPoolExecutor(max_workers=6, initializer=audit.init) as pool:
        for n, part in enumerate(pool.map(inspect_file, sorted(grouped.items())), 1):
            counts.update(part['counts']); results.extend(part['results']); packets.extend(part['packets'])
            print(json.dumps({'sample_files': n, 'total_files': len(grouped), 'samples_checked': counts['records']}), flush=True)
    release.base.save(audit.STATE / 'content-sample-summary.json', {'counts': dict(counts), 'results': results})
    release.base.save(audit.STATE / 'content-sample-packets.private.json', packets)
    print(json.dumps({'samples_checked': counts['records'], 'template_mismatches': counts['portable_template_differs_from_native']}), flush=True)
