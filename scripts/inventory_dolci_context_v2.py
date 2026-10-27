"""Inventory Dolci's documented call text and environment roles as inert data.

The v1 inventory assumed JSON calls and consequently could not project 208,000
rows. That is an adapter-scope failure, not a source-quality verdict. This new
version retains every v1 identity and adds source-visible call-text trajectories.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts import inventory_dolci_context_v1 as previous
from barbet_inference.memory import canonical

STATE = previous.STATE.parent / 'inventory-v2'


def field_present(value):
    if value is None: return False
    if not isinstance(value, str): raise ValueError('non-string source tool field')
    return value.strip() not in {'', '[]', '{}', 'null'}


def non_json_call(value):
    if not field_present(value): return False
    try: json.loads(value)
    except (ValueError, TypeError): return True
    return False


def job(item):
    number, source = item; folder = STATE / 'parts'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{number:03d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('pinned source changed')
    records = []; counts = Counter(); source_counts = {}; offset = 0
    for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            index = offset; offset += 1; counts['raw_rows'] += 1
            name = row.get('source_dataset', '<missing>'); source_counts.setdefault(name, Counter())['raw_rows'] += 1
            try:
                messages, question = previous.project(row)
                has_tools = any(field_present(m.get('functions')) for m in messages)
                has_calls = any(field_present(m.get('function_calls')) for m in messages)
                record = {'row_index': index, 'query_hash': base.sha(base.norm(question)),
                    'literal_first_query_hash': base.sha(question),
                    'source_final_input_hash': base.sha(canonical(messages[:-1])),
                    'source_final_answer_hash': base.sha(canonical(messages[-1])),
                    'source_dataset': row['source_dataset'], 'domain': row['domain'], 'source_id': row['id']}
            except (KeyError, ValueError, TypeError):
                counts['unprojectable'] += 1; source_counts[name]['unprojectable'] += 1; continue
            records.append(record); counts['projected_rows'] += 1; source_counts[name]['projected_rows'] += 1
            fields = {'multiple_user_turns': sum(m.get('role') == 'user' for m in messages) > 1,
                'multiple_assistant_turns': sum(m.get('role') == 'assistant' for m in messages) > 1,
                'declared_function_metadata': has_tools, 'recorded_function_calls': has_calls,
                'non_json_function_call_text': any(non_json_call(m.get('function_calls')) for m in messages),
                'recorded_tool_or_environment_observations': any(m.get('role') in {'tool', 'function', 'observation', 'environment'} for m in messages),
                'inline_thinking_markers': any('<think>' in (m.get('content') or '') or '</think>' in (m.get('content') or '') for m in messages),
                'reference_solution_marker_in_input': '<solution>' in canonical(messages[:-1])}
            for key, value in fields.items(): counts[key] += value; source_counts[name][key] += value
    old_receipt = json.loads((previous.STATE / 'parts' / f'{number:03d}.json').read_text())
    if base.file_sha(old_receipt['part']) != old_receipt['sha256']: raise ValueError('v1 inventory evidence changed')
    current = {r['row_index']: r for r in records}; old_rows = pq.read_table(old_receipt['part']).to_pylist()
    if any(current.get(r['row_index']) != r for r in old_rows): raise ValueError('v2 changed or lost a v1 projected identity')
    schema = pa.schema([('row_index', pa.int64()), *[(k, pa.string()) for k in ['query_hash', 'literal_first_query_hash',
        'source_final_input_hash', 'source_final_answer_hash', 'source_dataset', 'domain', 'source_id']]])
    part = receipt.with_suffix('.parquet'); pq.write_table(pa.Table.from_pylist(records, schema=schema), part, compression='zstd')
    result = {'source': source, 'counts': dict(counts), 'source_counts': {k: dict(v) for k, v in source_counts.items()},
        'part': str(part), 'sha256': base.file_sha(part), 'all_v1_identities_unchanged': True,
        'v1_projected_rows_retained': len(old_rows), 'v1_part_sha256': old_receipt['sha256']}
    base.save(receipt, result); return result


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'summary.json').exists(): raise ValueError('Dolci v2 inventory already complete')
    sources = json.loads((previous.RAW / 'lock.json').read_text())
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'source_lock_sha256': base.file_sha(previous.RAW / 'lock.json'),
        'v1_driver_sha256': base.file_sha(Path(previous.__file__)), 'v1_summary_sha256': base.file_sha(previous.STATE / 'summary.json'),
        'source_calls_kept_as_inert_literal_text': True, 'source_environment_role_counted': True,
        'accepted_training_rows': 0, 'model_requests': 0, 'source_programs_executed': False,
        'scope': 'Source identity and structure inventory only. No source registry inferred, no call executed, no v26 ranking or content-license admission.',
        'v1_unprojectable_not_a_source_error_rate': True}
    path = STATE / 'policy.json'
    if path.exists() and json.loads(path.read_text()) != policy: raise ValueError('inventory policy changed')
    if not path.exists(): base.save(path, policy)
    counts = Counter(); source_counts = {}; parts = []; retained = 0
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, part in enumerate(pool.map(job, enumerate(sources)), 1):
            parts.append(part); counts.update(part['counts']); retained += part['v1_projected_rows_retained']
            for name, value in part['source_counts'].items(): source_counts.setdefault(name, Counter()).update(value)
            print(json.dumps({'files': n, 'total': len(sources), 'raw_rows': counts['raw_rows']}), flush=True)
    path = STATE / 'identities.sqlite'
    if path.exists(): raise ValueError('partial v2 global inventory; inspect instead of overwriting')
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE identities(query_hash TEXT,input_hash TEXT,answer_hash TEXT,source_dataset TEXT,domain TEXT,PRIMARY KEY(query_hash,input_hash,answer_hash,source_dataset,domain)) WITHOUT ROWID')
    for part in parts:
        if base.file_sha(part['part']) != part['sha256']: raise ValueError('inventory part changed')
        rows = pq.read_table(part['part'], columns=['query_hash', 'source_final_input_hash', 'source_final_answer_hash', 'source_dataset', 'domain']).to_pylist()
        db.executemany('INSERT OR IGNORE INTO identities VALUES (?,?,?,?,?)',
            [(r['query_hash'], r['source_final_input_hash'], r['source_final_answer_hash'], r['source_dataset'], r['domain']) for r in rows]); db.commit()
    unique = {name: db.execute(f'SELECT COUNT(DISTINCT {column}) FROM identities').fetchone()[0]
              for name, column in [('normalized_first_queries', 'query_hash'), ('source_final_conditions', 'input_hash')]}
    per_source = {name: {'unique_first_queries': q, 'unique_final_conditions': inp, 'distinct_source_answer_pairs': n}
        for name, q, inp, n in db.execute('SELECT source_dataset,COUNT(DISTINCT query_hash),COUNT(DISTINCT input_hash),COUNT(*) FROM identities GROUP BY source_dataset')}
    db.close()
    result = {'counts': dict(counts), 'unique_counts': unique, 'source_counts': {k: dict(v) for k, v in source_counts.items()},
        'source_unique_counts': per_source, 'parts': parts, 'accepted_training_rows': 0,
        'v1_projected_rows_retained_exactly': retained, 'prior_release_overlap_not_yet_subtracted': True,
        'source_content_license_admission_not_completed': True, 'model_weights_trained': False}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['parts', 'source_counts']}), flush=True)


if __name__ == '__main__': main()
