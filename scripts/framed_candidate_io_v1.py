"""Prospective storage adapters for the unchanged candidate/family algorithms.

New cohort bindings must name this file and candidate_archive.py explicitly.
The old JSONL index, guard and canonicalizer files are never edited.
"""
from collections import Counter
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity


def checked_records(spec):
    if spec.get('format') != archive.FORMAT:
        raise ValueError('explicit framed candidate format required')
    if base.file_sha(spec['path']) != spec['sha256']:
        raise ValueError('candidate archive bytes changed')
    for offset, length, raw in archive.iter_records(spec['path']):
        yield offset, length, json.loads(raw)


def index_file(spec):
    dest = candidates.STATE / 'candidate-index-parts' / f"{spec['file_id']:05d}.parquet"
    receipt = dest.with_suffix('.json')
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if (result['storage_adapter_sha256'] != base.file_sha(Path(__file__))
                or result['source_archive_sha256'] != spec['sha256']
                or result['path'] and base.file_sha(result['path']) != result['sha256']):
            raise ValueError('framed index receipt differs')
        return result
    counts = Counter(); rows = []
    for offset, length, row in checked_records(spec):
        counts['seen'] += 1; original = row['record_id']
        try:
            candidates.eligible(row)
        except Rejected as exc:
            counts['excluded:' + str(exc)] += 1
            continue
        row = candidates.canonicalize(row)
        rows.append({'candidate_identity': row['candidate_identity'], 'record_id': row['record_id'],
            'original_record_id': original, 'input_hash': row['ppl_input_hash'],
            'bucket': int(row['ppl_input_hash'][:8], 16) % 8, 'file_id': spec['file_id'], 'offset': offset, 'length': length,
            'domain': row['domain'], 'query_hash': row['query_hash'], 'source_family_id': row['source_family_id'],
            'template_family_id': row.get('template_family_id', row['source_family_id']), 'source': row['source']})
        counts['eligible'] += 1
    dest.parent.mkdir(exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), dest, compression='zstd')
    result = {'file_id': spec['file_id'], 'counts': dict(counts), 'path': str(dest) if rows else None,
        'sha256': base.file_sha(dest) if rows else None, 'storage_adapter_sha256': base.file_sha(Path(__file__)),
        'source_archive_sha256': spec['sha256'], 'archive_format': archive.FORMAT}
    base.save(receipt, result)
    return result


def guard_file(spec):
    receipt = identity.STATE / 'parts' / f"{spec['file_id']:05d}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['storage_adapter_sha256'] != base.file_sha(Path(__file__)) or result['source_archive_sha256'] != spec['sha256']:
            raise ValueError('framed identity receipt differs')
        for key in ['cases', 'prefixes']:
            if result[key] and base.file_sha(result[key]) != result[key + '_sha256']:
                raise ValueError('framed identity part bytes differ')
        return result
    counts = Counter(); rows = []; prefixes = set()
    for unused_offset, unused_length, row in checked_records(spec):
        try:
            candidates.eligible(row)
        except Rejected:
            continue
        row = candidates.canonicalize(row); task, previous = identity.identities(row)
        rows.append({'candidate_identity': row['candidate_identity'], 'input_hash': row['ppl_input_hash'], 'base_input_hash': task})
        prefixes.update(previous); counts['eligible_rows'] += 1
    receipt.parent.mkdir(exist_ok=True)
    case_path = receipt.with_suffix('.cases.parquet'); prefix_path = receipt.with_suffix('.prefixes.parquet')
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), case_path, compression='zstd')
    if prefixes:
        pq.write_table(pa.table({'candidate_identity': sorted(prefixes)}), prefix_path, compression='zstd')
    result = {'file_id': spec['file_id'], 'counts': dict(counts),
        'cases': str(case_path) if rows else None, 'cases_sha256': base.file_sha(case_path) if rows else None,
        'prefixes': str(prefix_path) if prefixes else None, 'prefixes_sha256': base.file_sha(prefix_path) if prefixes else None,
        'storage_adapter_sha256': base.file_sha(Path(__file__)), 'source_archive_sha256': spec['sha256'],
        'archive_format': archive.FORMAT}
    base.save(receipt, result)
    return result
