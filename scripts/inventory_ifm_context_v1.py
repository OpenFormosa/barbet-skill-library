"""Inventory distinct public final-answer inputs; never import think fields.

This source inventory accepts no training examples and is not a substitute for
tool validation, grounding, source-split checks or native v26 answer selection.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base

RAW = base.RAW / 'scale10m-ifm'
STATE = base.WORK / 'reports/memory-harness-v2-ifm-v1/inventory-v1'
VISIBLE = {'role', 'content', 'tools', 'tool_calls', 'tool_responses', 'tool_call_id', 'name'}


def project(raw):
    messages = json.loads(raw['messages'])
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError('invalid message array')
    projected = [{k: v for k, v in m.items() if k in VISIBLE} for m in messages]
    users = [m['content'] for m in projected if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    if not users or not users[0].strip() or projected[-1].get('role') != 'assistant':
        raise ValueError('missing first query or final target')
    # Source hashes only: this does not assert that a tool observation or answer
    # is valid, and does not normalize different literal conditions together.
    return projected, users[0]


def job(item):
    number, source = item; folder = STATE / 'parts'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{number:03d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('IFM source changed')
    category = source['path'].split('/')[0]; effort = Path(source['path']).name.split('-')[0]
    rows = []; counts = Counter(); offset = 0
    for batch in pq.ParquetFile(source['local']).iter_batches(columns=['messages'], batch_size=1024):
        for raw in batch.to_pylist():
            index = offset; offset += 1; counts['raw_rows'] += 1
            try:
                messages, query = project(raw)
                body = canonical(messages)
                row = {'row_index': index, 'query_hash': base.sha(base.norm(query)),
                       'literal_first_query_hash': base.sha(query),
                       'source_final_input_hash': base.sha(canonical(messages[:-1])),
                       'source_final_answer_hash': base.sha(canonical(messages[-1])),
                       'source_trajectory_hash': base.sha(body)}
            except (TypeError, ValueError, KeyError): counts['unprojectable'] += 1; continue
            counts['projected_rows'] += 1
            counts['multi_user_turn_rows'] += sum(m.get('role') == 'user' for m in messages) > 1
            counts['multi_assistant_turn_rows'] += sum(m.get('role') == 'assistant' for m in messages) > 1
            counts['tool_observation_rows'] += any(m.get('role') == 'tool' for m in messages)
            counts['declared_tool_schema_rows'] += any(m.get('tools') for m in messages)
            counts['final_tool_call_rows'] += bool(messages[-1].get('tool_calls'))
            counts['reference_solution_marker_rows'] += '<solution>' in canonical(messages[:-1])
            rows.append(row)
    schema = pa.schema([('row_index', pa.int64()), *[(name, pa.string()) for name in [
        'query_hash', 'literal_first_query_hash', 'source_final_input_hash',
        'source_final_answer_hash', 'source_trajectory_hash']]])
    part = receipt.with_suffix('.parquet'); pq.write_table(pa.Table.from_pylist(rows, schema=schema), part, compression='zstd')
    result = {'source': source, 'category': category, 'effort': effort, 'counts': dict(counts),
              'part': str(part), 'sha256': base.file_sha(part)}
    base.save(receipt, result); return result


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'summary.json').exists(): raise ValueError('source inventory already frozen')
    sources = json.loads((RAW / 'lock.json').read_text()); parts = []; counts = Counter(); categories = {}
    policy = {'source_lock_sha256': base.file_sha(RAW / 'lock.json'), 'driver_sha256': base.file_sha(Path(__file__)),
        'excluded_fields': ['think', 'reasoning', 'analysis', 'all message keys outside explicit visible allowlist'],
        'hash_scope': 'Exact source final input/final target and full visible trajectory; not the normalized native scoring protocol.',
        'quality_or_license_admission': False, 'accepted_training_rows': 0, 'model_requests': 0,
        'source_programs_executed': False, 'weights_trained': False}
    path = STATE / 'policy.json'
    if path.exists() and json.loads(path.read_text()) != policy: raise ValueError('inventory policy changed')
    if not path.exists(): base.save(path, policy)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n, part in enumerate(pool.map(job, enumerate(sources)), 1):
            parts.append(part); counts.update(part['counts'])
            categories.setdefault(part['category'], Counter()).update(part['counts'])
            if n % 10 == 0 or n == len(sources): print(json.dumps({'files': n, 'total': len(sources), 'rows': counts['raw_rows']}), flush=True)
    db_path = STATE / 'identities.sqlite'
    if db_path.exists(): raise ValueError('partial global identity aggregation; inspect before resuming')
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE identities(query_hash TEXT,input_hash TEXT,answer_hash TEXT,category TEXT,PRIMARY KEY(query_hash,input_hash,answer_hash,category)) WITHOUT ROWID')
    for n, part in enumerate(parts, 1):
        if base.file_sha(part['part']) != part['sha256']: raise ValueError('inventory part changed')
        rows = pq.read_table(part['part'], columns=['query_hash', 'source_final_input_hash', 'source_final_answer_hash']).to_pylist()
        db.executemany('INSERT OR IGNORE INTO identities VALUES (?,?,?,?)',
            [(r['query_hash'], r['source_final_input_hash'], r['source_final_answer_hash'], part['category']) for r in rows]); db.commit()
        if n % 10 == 0 or n == len(parts): print(json.dumps({'parts_aggregated': n, 'total': len(parts)}), flush=True)
    unique_questions = db.execute('SELECT COUNT(DISTINCT query_hash) FROM identities').fetchone()[0]
    unique_inputs = db.execute('SELECT COUNT(DISTINCT input_hash) FROM identities').fetchone()[0]
    per_category = {cat: {'distinct_first_queries': q, 'distinct_final_inputs': inp, 'distinct_source_answer_pairs': n}
        for cat, q, inp, n in db.execute('SELECT category,COUNT(DISTINCT query_hash),COUNT(DISTINCT input_hash),COUNT(*) FROM identities GROUP BY category')}
    result = {'counts': dict(counts), 'category_counts': {k: dict(v) for k, v in categories.items()},
        'distinct_normalized_first_queries': unique_questions, 'distinct_complete_source_final_inputs': unique_inputs,
        'category_unique_counts': per_category, 'parts': parts, 'accepted_training_rows': 0,
        'scope': 'Downloaded eight-category source only, excluding safety and self-identity. Source effort variants and multi-turn prefixes are not independently accepted episodes. Before prior-source dedup, provenance/grounding/schema checks, first diagnosis, native v26 PPL and export gates.'}
    db.close(); base.save(STATE / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'parts'}), flush=True)


if __name__ == '__main__': main()
