"""Scan new source-bound defects; preserve inputs/targets and hold whole queries."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from scripts import audit_barbet_continuation_v1 as audit
from scripts import publish_barbet_only_v1 as release
from barbet_inference.readiness_quality import continuation_content_holds

TASK = 'tasks/task574_air_dialogue_sentence_generation.json'


def scan(part_path):
    part = release.read(part_path); counts = part['counts']
    needed = any(counts.get('source:' + name) for name in
                 ['allenai/natural-instructions', 'voidful/agent-sft', 'nvidia/Nemotron-SFT-Agentic-v2'])
    if not needed: return []
    result = []
    columns = ['record_id', 'query_hash', 'source', 'domain', 'messages_json', 'tools_json', 'source_refs_json']
    for batch in pq.ParquetFile(release.parent.DEST / part['path']).iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            source = row['source']; issues = []
            if source == 'allenai/natural-instructions' and TASK in row['source_refs_json']:
                refs = json.loads(row['source_refs_json'])
                if refs['file'] == TASK:
                    issues.append({'reason': 'task574_generative_gap_requires_unsupplied_booking_facts'})
            if source in ['voidful/agent-sft', 'nvidia/Nemotron-SFT-Agentic-v2'] and row['domain'] == 'tool_use':
                issues.extend(continuation_content_holds(source, row['domain'], json.loads(row['messages_json']),
                                                         has_tools=bool(json.loads(row['tools_json']))))
            if issues:
                result.append({'record_id': row['record_id'], 'query_hash': row['query_hash'],
                               'source': source, 'path': part['path'], 'issues': issues})
    return result


if __name__ == '__main__':
    path = audit.STATE / 'new-query-holds-v1.json'
    if path.exists(): raise ValueError('First quality holds already frozen')
    manual = release.read(audit.STATE / 'manual-first-review-v1.json')
    bad = [r for r in manual['results'] if r['verdict'] != 'ACCEPT']
    diagnosis = release.read(audit.STATE / 'special-token-source-diagnosis.json')
    pads = [r for r in diagnosis['records'] if 'pad' in r['tags']]
    rows = []
    for row in pads:
        rows.append({k: row[k] for k in ['record_id', 'query_hash', 'source']} |
                    {'issues': [{'reason': 'literal_native_PAD_inside_real_content'}]})
    db = sqlite3.connect(f'file:{release.parent.STATE / "public-index.sqlite"}?mode=ro', uri=True)
    for row in bad:
        query = db.execute('SELECT query_hash FROM records WHERE record_id=?', (row['record_id'],)).fetchone()[0]
        rows.append({'record_id': row['record_id'], 'query_hash': query, 'source': row['source'],
                     'issues': [{'reason': 'first_manual_nonaccept', 'detail': row['reason']}]})
    db.close()
    release.base.save(audit.STATE / 'source-hold-scan-policy.json', {
        'driver_sha256': release.base.file_sha(Path(__file__)),
        'guard_sha256': release.base.file_sha(release.base.REPO / 'barbet_inference/readiness_quality.py'),
        'manual_review_sha256': release.base.file_sha(audit.STATE / 'manual-first-review-v1.json'),
        'special_token_diagnosis_sha256': release.base.file_sha(audit.STATE / 'special-token-source-diagnosis.json'),
        'task574_scope': TASK, 'all_first_new_nonaccept_queries_held': True,
        'source_code_executed': False, 'score_or_answer_replacement': False, 'workers': 12})
    parts = sorted((audit.STATE / 'files').glob('*.json'))
    with ProcessPoolExecutor(max_workers=12) as pool:
        for n, added in enumerate(pool.map(scan, parts), 1):
            rows.extend(added)
            if n % 100 == 0 or n == len(parts):
                print(json.dumps({'files_scanned': n, 'total': len(parts), 'hold_findings': len(rows)}), flush=True)
    by_record = {}
    for row in rows:
        key = row['record_id']
        if key not in by_record: by_record[key] = {**row, 'issues': list(row['issues'])}
        else: by_record[key]['issues'].extend(row['issues'])
    rows = sorted(by_record.values(), key=lambda x: x['record_id'])
    reasons = Counter(i['reason'] for r in rows for i in r['issues'])
    report = {'query_hashes': sorted({r['query_hash'] for r in rows}), 'holds': rows,
              'reason_findings': dict(reasons), 'task574_scope': TASK,
              'all_queries_held_without_runner_up': True,
              'no_corpus_semantic_accuracy_claim': True,
              'literal_UNK_occurrences': 287, 'UNK_rows': 75,
              'all_UNK_offsets_equal_literal_unk': all(c['literal'] == '<unk>' for r in diagnosis['records']
                                                      for c in r['contexts'] if c['token'] == 114688)}
    release.base.save(path, report)
    print(json.dumps({'held_record_findings': len(rows), 'held_queries': len(report['query_hashes']),
                      'reasons': dict(reasons)}), flush=True)
