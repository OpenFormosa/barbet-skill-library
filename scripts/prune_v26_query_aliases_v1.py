"""Hold single-turn question aliases before publication; never compare unequal inputs.

The original compact release is retained as superseded evidence. Every retained
record, label and budget proof is value-identical to that fully checked release.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base
from scripts.build_v26_complete_release import STATE as SELECTED, COMPACT as ORIGINAL, DEST as SOURCE
from scripts import compact_memory_release_v2 as exact

STATE = SELECTED / 'query-alias-pruned-v1'
DEST = base.WORK / 'exports/memory-harness-v2-v26-complete-public-v2'
COHORTS = {'Open-Orca/OpenOrca', 'microsoft/orca-math-word-problems-200k'}
WAVE = 'wave-0003'


def prepare():
    STATE.mkdir(exist_ok=True)
    path = STATE / 'policy.json'
    if path.exists(): return json.loads(path.read_text())
    if (ORIGINAL / 'publication-data-commit.json').exists(): raise ValueError('original payload already published')
    prior = sqlite3.connect(f'file:{SELECTED / "public-index.sqlite"}?mode=ro', uri=True)
    queries = [{'source': source, 'query_hash': query, 'selected_views': count}
        for source, query, count in prior.execute('SELECT source,query_hash,COUNT(*) FROM records GROUP BY source,query_hash HAVING COUNT(*)>1')
        if source in COHORTS]
    prior.close()
    policy = {'version': 'same-normalized-single-turn-question-v1', 'cohorts': sorted(COHORTS),
        'held_source_queries': queries, 'held_selected_views': sum(r['selected_views'] for r in queries),
        'rule': 'Hold all selected views when the same normalized question in an admitted single-turn text source has multiple complete conditioning inputs. Do not select the first source or compare PPL from different input bytes.',
        'future_recovery': 'These are unresolved input-canonicalization holds, not negative semantic verdicts. Recover only under a prospective meaning-preserving common-input policy and new measured likelihoods; all existing semantic holds remain final.',
        'original_compact_summary_sha256': base.file_sha(ORIGINAL / 'summary.json'),
        'prior_manifest_sha256': base.file_sha(ORIGINAL / 'publication-plan.json'),
        'driver_sha256': base.file_sha(Path(__file__)), 'model_weights_trained': False}
    base.save(path, policy)
    base.save(ORIGINAL / 'SUPERSEDED.json', {'reason': 'Additional normalized single-turn query aliases found before publication.',
        'replacement_state': str(STATE), 'policy_sha256': base.file_sha(path), 'original_files_preserved': True})
    print(json.dumps({'held_source_queries': len(queries), 'held_selected_views': policy['held_selected_views']}), flush=True)
    return policy


def init():
    global HOLDS
    policy = json.loads((STATE / 'policy.json').read_text())
    if policy['driver_sha256'] != base.file_sha(Path(__file__)): raise ValueError('prospective alias policy changed')
    HOLDS = {(r['source'], r['query_hash']) for r in policy['held_source_queries']}
    exact.DEST = DEST; exact.STATE = STATE


def keep_indices(rows, holds):
    return [n for n, row in enumerate(rows) if (row['source'], row['query_hash']) not in holds]


def job(spec):
    split, name, expected = spec
    receipt = STATE / 'parts' / (split + '-' + name.removesuffix('.parquet') + '.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    tables = {}
    for config in ['records', 'tokenized', 'minicpm5_tokenized']:
        path = SOURCE / 'data' / WAVE / config / name
        if base.file_sha(path) != expected[str(path.relative_to(SOURCE))]: raise ValueError('original compact bytes changed')
        tables[config] = pq.read_table(path)
    ids = tables['records'].column('record_id').to_pylist()
    if any(t.column('record_id').to_pylist() != ids for t in tables.values()): raise ValueError('unaligned input views')
    rows = tables['records'].to_pylist(); keep = keep_indices(rows, HOLDS); counts = Counter(); index = []
    for n in keep:
        row = rows[n]; counts['accepted'] += 1; counts['split:' + split] += 1
        for key in ['pangolin_tokens', 'pangolin_supervised_tokens', 'minicpm_tokens', 'minicpm_supervised_tokens',
                    'assistant_turns', 'tool_observations']:
            counts[key] += row[key]
        for key in ['domain', 'source', 'upstream_source', 'license', 'verification_tier', 'memory_placement', 'observation_origin']:
            counts[key + ':' + str(row[key])] += 1
        for key in ['actual_retrieval', 'recorded_error_followup']: counts[key] += int(row[key])
        counts['selection:' + json.loads(row['source_refs_json'])['answer_selection']['method']] += 1
        length = max(row['pangolin_tokens'], row['minicpm_tokens'])
        counts['length_bucket:' + str(next(b for b in [4096, 8192, 16384, 32768, 65536, 131072] if length <= b))] += 1
        index.append({k: row[k] for k in ['record_id', 'family_id', 'query_hash', 'split', 'domain', 'source', 'upstream_source']})
    counts['excluded:ambiguous_normalized_single_turn_question'] += len(rows) - len(keep)
    files = []
    for config, table in tables.items():
        selected = table.take(pa.array(keep, type=pa.int64()))
        if len(selected):
            files.append(exact.write_checked(selected, DEST / 'data' / WAVE / config / name,
                                             wave=WAVE, config=config, split=split))
    receipt.parent.mkdir(exist_ok=True)
    index_path = receipt.with_suffix('.index.parquet')
    pq.write_table(pa.Table.from_pylist(index), index_path, compression='zstd')
    result = {'counts': dict(counts), 'files': files, 'index': str(index_path), 'index_sha256': base.file_sha(index_path),
              'all_retained_values_equal': True}
    base.save(receipt, result); return result


def build(workers):
    policy = prepare(); init()
    if (STATE / 'summary.json').exists(): raise ValueError('alias-pruned release already complete')
    original = json.loads((ORIGINAL / 'summary.json').read_text())
    if base.file_sha(ORIGINAL / 'summary.json') != policy['original_compact_summary_sha256']: raise ValueError('original summary changed')
    expected = {f['path']: f['sha256'] for f in original['files']}; specs = []
    for path in sorted((SOURCE / 'data' / WAVE / 'records').glob('*.parquet')):
        specs.append((path.name.split('-')[0], path.name, expected))
    parts = []; counts = Counter(); files = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, part in enumerate(pool.map(job, specs), 1):
            parts.append(part); counts.update(part['counts']); files.extend(part['files'])
            if n % 10 == 0 or n == len(specs): print(json.dumps({'shards': n, 'total': len(specs), 'accepted': counts['accepted']}), flush=True)
    if counts['excluded:ambiguous_normalized_single_turn_question'] != policy['held_selected_views']:
        raise ValueError('alias removal coverage differs')
    if counts['accepted'] + policy['held_selected_views'] != original['counts']['accepted']:
        raise ValueError('unexpected rows added or lost')
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)')
    for part in parts:
        if base.file_sha(part['index']) != part['index_sha256']: raise ValueError('retained index changed')
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)', [tuple(r.values()) for r in pq.read_table(part['index']).to_pylist()]); db.commit()
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0] for key in ['record_id', 'query_hash', 'family_id']}
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('retained fold leakage')
    for source in COHORTS:
        if db.execute('SELECT 1 FROM records WHERE source=? GROUP BY query_hash HAVING COUNT(*)>1 LIMIT 1', (source,)).fetchone():
            raise ValueError('ambiguous single-turn query revived')
    ids = {r[0] for r in db.execute('SELECT record_id FROM records')}; db.close(); proofs = set()
    for p in sorted((SOURCE / 'proofs' / WAVE).glob('*.parquet')):
        if base.file_sha(p) != expected[str(p.relative_to(SOURCE))]: raise ValueError('original proof changed')
        table = pq.read_table(p); keys = table.column('record_id').to_pylist(); keep = [n for n, key in enumerate(keys) if key in ids]
        selected = table.take(pa.array(keep, type=pa.int64())); kept = selected.column('record_id').to_pylist()
        if set(kept) & proofs or len(set(kept)) != len(kept): raise ValueError('duplicate proof')
        proofs.update(kept)
        if len(selected): files.append(exact.write_checked(selected, DEST / 'proofs' / WAVE / p.name, wave=WAVE, config='budget-proof', split=None))
    if ids != proofs: raise ValueError('retained proof coverage differs')
    for key, value in original['counts'].items():
        if key.startswith('excluded:'): counts[key] = value
    counts['seen'] = original['counts']['seen']
    summary = {**{k: v for k, v in original.items() if k not in ['files', 'counts', 'unique_counts']},
        'counts': dict(counts), 'unique_counts': unique, 'files': files,
        'all_retained_records_and_token_arrays_equal': True, 'all_budget_proofs_preserved': True,
        'all_single_turn_query_alias_holds_absent': True, 'alias_policy_sha256': base.file_sha(STATE / 'policy.json'),
        'original_verified_payload_preserved': True, 'status': 'LOCAL_VERIFIED_NOT_YET_PUBLISHED'}
    base.save(STATE / 'summary.json', summary)
    print(json.dumps({'accepted': counts['accepted'], 'unique_counts': unique, 'files': len(files)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=16); args = p.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('worker bound')
    build(args.workers)
