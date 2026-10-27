"""Count actual normalized problems, without counting solution multiplicity."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts.diagnose_openmath_memory_v1 import STATE, RAW


def job(item):
    ident, source = item; folder = STATE / 'problem-inventory-v1/parts'; folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'{ident:03d}.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('raw source bytes changed')
    counts = Counter(); totals = Counter()
    for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=8192, columns=['problem', 'problem_source']):
        for row in batch.to_pylist():
            counts[(base.sha(base.norm(row['problem'])), row['problem_source'])] += 1
            totals[row['problem_source']] += 1
    rows = [{'query_hash': q, 'problem_source': category, 'solutions': n} for (q, category), n in sorted(counts.items())]
    path = receipt.with_suffix('.parquet'); pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    result = {'source': source['path'], 'source_sha256': source['sha256'], 'counts': dict(totals),
              'part': str(path), 'part_sha256': base.file_sha(path)}
    base.save(receipt, result); return result


def main():
    folder = STATE / 'problem-inventory-v1'; folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists(): raise ValueError('inventory already frozen')
    files = json.loads((RAW / 'lock.json').read_text()); parts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, part in enumerate(pool.map(job, enumerate(files)), 1):
            parts.append(part); print(json.dumps({'files_inventoried': n, 'total': len(files)}), flush=True)
    db = sqlite3.connect(folder / 'problems.sqlite')
    db.execute('CREATE TABLE problems(query_hash TEXT,problem_source TEXT,solutions INTEGER,PRIMARY KEY(query_hash,problem_source))')
    for part in parts:
        if base.file_sha(part['part']) != part['part_sha256']: raise ValueError('inventory bytes changed')
        rows = pq.read_table(part['part']).to_pylist()
        db.executemany('INSERT INTO problems VALUES (?,?,?) ON CONFLICT(query_hash,problem_source) DO UPDATE SET solutions=solutions+excluded.solutions',
                       [(r['query_hash'], r['problem_source'], r['solutions']) for r in rows]); db.commit()
    result = {'raw_problem_solution_pairs': db.execute('SELECT SUM(solutions) FROM problems').fetchone()[0],
        'distinct_normalized_problems': db.execute('SELECT COUNT(DISTINCT query_hash) FROM problems').fetchone()[0],
        'by_category': {s: {'distinct_problems': n, 'raw_solutions': c} for s, n, c in db.execute('SELECT problem_source,COUNT(*),SUM(solutions) FROM problems GROUP BY problem_source')},
        'scope': 'Complete main train split only. Exact NFKC/casefold/whitespace question identity, not semantic or cross-source deduplication. Before any quality or contamination filters.',
        'accepted_training_rows': 0, 'parts': parts}
    db.close(); base.save(folder / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'parts'}), flush=True)


if __name__ == '__main__': main()
