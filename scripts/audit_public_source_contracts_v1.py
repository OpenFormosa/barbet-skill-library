"""Read-only scan of every currently published record for exact schema holds."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pyarrow.parquet as pq
from barbet_inference.source_contract_holds import held_tool_schemas
from scripts import build_memory_sft_v2 as base
from scripts.publish_opencode_v26_v1 import STATE as PUBLICATION, DEST

STATE = base.WORK / 'reports/memory-harness-v2-public-source-contracts-v1'
POLICY = base.REPO / 'configs/source_tool_contract_holds_v1.json'


def job(item):
    source = DEST / item['path']
    if base.file_sha(source) != item['sha256']: raise ValueError('published local file binding changed')
    policy = json.loads(POLICY.read_text()); counts = Counter(); found = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=2048, columns=['record_id', 'query_hash', 'source', 'tools_json']):
        for row in batch.to_pylist():
            counts['records_scanned'] += 1
            if row['tools_json'] == '[]': continue
            counts['records_with_tools'] += 1
            matches = held_tool_schemas(json.loads(row['tools_json']), policy)
            if matches:
                found.append({'record_id': row['record_id'], 'query_hash': row['query_hash'],
                              'source': row['source'], 'path': item['path'], 'holds': matches})
    return {'path': item['path'], 'sha256': item['sha256'], 'counts': dict(counts), 'matches': found}


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'summary.json').exists(): raise ValueError('public source-contract scan already frozen')
    proof = json.loads((PUBLICATION / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']: raise ValueError('current publication not verified')
    plan = json.loads((PUBLICATION / 'publication-plan.json').read_text())
    files = [f for f in plan['manifest'] if f['path'].startswith('data/') and '/records/' in f['path'] and f['path'].endswith('.parquet')]
    counts = Counter(); matches = []; reports = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n, result in enumerate(pool.map(job, files), 1):
            counts.update(result['counts']); matches += result['matches']; reports.append({k: v for k, v in result.items() if k != 'matches'})
            if n % 25 == 0 or n == len(files): print(json.dumps({'files': n, 'total': len(files), 'matches': len(matches)}), flush=True)
    if counts['records_scanned'] != proof['active_new_episodes']: raise ValueError('published row coverage mismatch')
    report = {'dataset': proof['dataset'], 'data_commit': proof['data_commit'], 'verified_public_head': proof['final_head'],
        'policy_sha256': base.file_sha(POLICY), 'driver_sha256': base.file_sha(Path(__file__)),
        'counts': dict(counts), 'matched_records': len(matches), 'matched_query_hashes': sorted({r['query_hash'] for r in matches}),
        'matches': matches, 'parts': reports, 'read_only': True, 'remote_mutations': 0,
        'model_requests': 0, 'source_programs_executed': False,
        'scope': 'Exact known inconsistent source metadata fingerprint across all current record rows. Not a generic semantic tool audit.'}
    base.save(STATE / 'summary.json', report)
    print(json.dumps({'rows_scanned': counts['records_scanned'], 'matched_records': len(matches), 'remote_mutations': 0}), flush=True)


if __name__ == '__main__': main()
