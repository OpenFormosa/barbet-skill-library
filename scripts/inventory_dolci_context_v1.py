"""Count actual Dolci source questions and conditions before any admission.

Original per-row source labels are retained. String tool fields are only parsed
as JSON for inventory; no registry is inferred and no source program executes.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts.fetch_dolci_inventory_v1 import ROOT as RAW
from barbet_inference.memory import canonical

STATE = base.WORK / 'reports/memory-harness-v2-dolci-v1/inventory-v1'


def json_field_present(value):
    if value in (None, ''): return False
    if not isinstance(value, str): raise ValueError('source tool field is not serialized JSON')
    return bool(json.loads(value))


def project(row):
    messages = row['messages']
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError('invalid source message array')
    queries = [m['content'] for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    if not queries or not queries[0].strip() or messages[-1].get('role') != 'assistant':
        raise ValueError('missing initial user or final assistant')
    if not isinstance(row['source_dataset'], str) or not row['source_dataset']:
        raise ValueError('missing per-row source label')
    return messages, queries[0]


def job(item):
    number, source = item; folder = STATE / 'parts'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{number:03d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('pinned source changed')
    records = []; counts = Counter(); source_counts = {}; offset = 0
    for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            index = offset; offset += 1; counts['raw_rows'] += 1
            source_counts.setdefault(row.get('source_dataset', '<missing>'), Counter())['raw_rows'] += 1
            try:
                messages, question = project(row)
                has_tools = any(json_field_present(m.get('functions')) for m in messages)
                has_calls = any(json_field_present(m.get('function_calls')) for m in messages)
                record = {'row_index': index, 'query_hash': base.sha(base.norm(question)),
                    'literal_first_query_hash': base.sha(question),
                    'source_final_input_hash': base.sha(canonical(messages[:-1])),
                    'source_final_answer_hash': base.sha(canonical(messages[-1])),
                    'source_dataset': row['source_dataset'], 'domain': row['domain'],
                    'source_id': row['id']}
            except (KeyError, ValueError, TypeError): counts['unprojectable'] += 1; continue
            records.append(record); counts['projected_rows'] += 1
            fields = {'multiple_user_turns': sum(m.get('role') == 'user' for m in messages) > 1,
                'multiple_assistant_turns': sum(m.get('role') == 'assistant' for m in messages) > 1,
                'declared_function_metadata': has_tools, 'recorded_function_calls': has_calls,
                'recorded_tool_or_function_observations': any(m.get('role') in {'tool', 'function', 'observation'} for m in messages),
                'inline_thinking_markers': any('<think>' in (m.get('content') or '') or '</think>' in (m.get('content') or '') for m in messages),
                'reference_solution_marker_in_input': '<solution>' in canonical(messages[:-1])}
            for key, value in fields.items():
                counts[key] += value; source_counts[row['source_dataset']][key] += value
    schema = pa.schema([('row_index', pa.int64()), *[(k, pa.string()) for k in ['query_hash', 'literal_first_query_hash',
        'source_final_input_hash', 'source_final_answer_hash', 'source_dataset', 'domain', 'source_id']]])
    part = receipt.with_suffix('.parquet'); pq.write_table(pa.Table.from_pylist(records, schema=schema), part, compression='zstd')
    result = {'source': source, 'counts': dict(counts), 'source_counts': {k: dict(v) for k, v in source_counts.items()},
              'part': str(part), 'sha256': base.file_sha(part)}
    base.save(receipt, result); return result


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'summary.json').exists(): raise ValueError('Dolci inventory already complete')
    sources = json.loads((RAW / 'lock.json').read_text())
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'source_lock_sha256': base.file_sha(RAW / 'lock.json'),
        'accepted_training_rows': 0, 'model_requests': 0, 'source_programs_executed': False,
        'scope': 'Source-level query and complete source-condition inventory. Not canonical v26 groups, per-source license approval, semantic verification or accepted task counts.',
        'database_license_not_assumed_to_cover_individual_content': True}
    path = STATE / 'policy.json'
    if path.exists() and json.loads(path.read_text()) != policy: raise ValueError('inventory policy changed')
    if not path.exists(): base.save(path, policy)
    counts = Counter(); source_counts = {}; parts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, part in enumerate(pool.map(job, enumerate(sources)), 1):
            parts.append(part); counts.update(part['counts'])
            for name, value in part['source_counts'].items(): source_counts.setdefault(name, Counter()).update(value)
            print(json.dumps({'files': n, 'total': len(sources), 'raw_rows': counts['raw_rows']}), flush=True)
    path = STATE / 'identities.sqlite'
    if path.exists(): raise ValueError('partial global inventory aggregation; inspect instead of overwriting')
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
    result = {'counts': dict(counts), 'unique_counts': unique,
        'source_counts': {k: dict(v) for k, v in source_counts.items()}, 'source_unique_counts': per_source,
        'parts': parts, 'accepted_training_rows': 0, 'prior_release_overlap_not_yet_subtracted': True,
        'source_content_license_admission_not_completed': True, 'model_weights_trained': False}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'parts'}), flush=True)


if __name__ == '__main__': main()
