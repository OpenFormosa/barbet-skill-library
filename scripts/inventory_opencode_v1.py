"""Full raw-question inventory and first-test gates, without executing code."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
from barbet_inference.sft import Rejected, RESERVED
from scripts import build_memory_sft_v2 as base
from scripts.opencode_memory_adapter_v1 import source_quality

STATE = base.WORK / 'reports/memory-harness-v2-opencode-v1'
RAW = base.RAW / 'scale10m-opencode'


def job(spec):
    number, source = spec; folder = STATE / 'inventory-v1/parts'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{number:03d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('source bytes changed')
    rows = []; counts = Counter(); offset = 0
    for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=4096):
        for raw in batch.to_pylist():
            counts['raw_rows'] += 1; reason = None
            try:
                source_quality(raw)
                if not isinstance(raw.get('input'), str) or not raw['input'].strip() or not isinstance(raw.get('output'), str) or not raw['output'].strip():
                    raise Rejected('missing_question_or_answer')
                if RESERVED.search(raw['input'] + raw['output']): raise Rejected('literal_reserved_delimiter')
            except Rejected as exc: reason = str(exc)
            counts['held:' + reason if reason else 'source_gate_eligible_answers'] += 1
            rows.append({'query_hash': base.sha(base.norm(raw['input'])), 'literal_question_hash': base.sha(raw['input']),
                         'answer_hash': base.sha(raw['output']), 'row_index': offset,
                         'eligible': reason is None, 'hold_reason': reason or '',
                         'cohort': str(raw['domain']) + '/' + str(raw['generation_algorithm'])})
            offset += 1
    part = receipt.with_suffix('.parquet'); pq.write_table(pa.Table.from_pylist(rows), part, compression='zstd')
    result = {'source': source, 'counts': dict(counts), 'part': str(part), 'sha256': base.file_sha(part)}
    base.save(receipt, result); return result


def main():
    STATE.mkdir(exist_ok=True); folder = STATE / 'inventory-v1'; folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists(): raise ValueError('source inventory already frozen')
    sources = json.loads((RAW / 'lock.json').read_text()); counts = Counter(); parts = []
    base.save(folder / 'policy.json', {'source_lock_sha256': base.file_sha(RAW / 'lock.json'),
        'adapter_sha256': base.file_sha(base.REPO / 'scripts/opencode_memory_adapter_v1.py'),
        'driver_sha256': base.file_sha(Path(__file__)), 'source_test_failures_remain_held': True,
        'source_programs_executed': False, 'accepted_training_rows': 0})
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, part in enumerate(pool.map(job, enumerate(sources)), 1):
            counts.update(part['counts']); parts.append(part)
            if n % 10 == 0 or n == len(sources): print(json.dumps({'files': n, 'total': len(sources), 'counts': dict(counts)}), flush=True)
    db = sqlite3.connect(folder / 'questions.sqlite')
    db.execute('CREATE TABLE questions(query_hash TEXT PRIMARY KEY,raw_answers INTEGER,eligible_answers INTEGER)')
    db.execute('CREATE TABLE eligible_answers(query_hash TEXT,literal_question_hash TEXT,answer_hash TEXT,PRIMARY KEY(query_hash,literal_question_hash,answer_hash))')
    for part in parts:
        if base.file_sha(part['part']) != part['sha256']: raise ValueError('source projection changed')
        rows = pq.read_table(part['part']).to_pylist(); totals = {}
        for r in rows:
            total = totals.setdefault(r['query_hash'], [0, 0]); total[0] += 1; total[1] += int(r['eligible'])
        db.executemany('INSERT INTO questions VALUES (?,?,?) ON CONFLICT(query_hash) DO UPDATE SET raw_answers=raw_answers+excluded.raw_answers,eligible_answers=eligible_answers+excluded.eligible_answers',
                       [(q, n, e) for q, (n, e) in totals.items()])
        db.executemany('INSERT OR IGNORE INTO eligible_answers VALUES (?,?,?)',
                       [(r['query_hash'], r['literal_question_hash'], r['answer_hash']) for r in rows if r['eligible']]); db.commit()
    summary = {'counts': dict(counts), 'raw_distinct_normalized_questions': db.execute('SELECT COUNT(*) FROM questions').fetchone()[0],
        'questions_with_a_source_gate_eligible_answer': db.execute('SELECT COUNT(*) FROM questions WHERE eligible_answers>0').fetchone()[0],
        'distinct_source_gate_eligible_answers': db.execute('SELECT COUNT(*) FROM eligible_answers').fetchone()[0],
        'question_aliases_with_unequal_literal_inputs': db.execute('SELECT COUNT(*) FROM (SELECT query_hash FROM eligible_answers GROUP BY query_hash HAVING COUNT(DISTINCT literal_question_hash)>1)').fetchone()[0],
        'scope': 'Full main source; before legacy/evaluation/cross-source exclusion, semantic diagnosis, v26 ranking, splits or token budgets. Upstream test execution not rerun or independently confirmed.',
        'accepted_training_rows': 0, 'source_programs_executed': False, 'parts': parts}
    db.close(); base.save(folder / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'parts'}), flush=True)


if __name__ == '__main__': main()
