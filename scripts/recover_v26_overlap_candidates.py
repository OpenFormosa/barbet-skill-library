"""Inspect source rows previously skipped only because their query was seen.

Quality, legacy, evaluation and first-negative holds remain in force. The old
normalizer artifacts and the completed score run are not modified. New answer
variants are staged separately; they are not accepted training rows yet.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import copy
import json
import sqlite3

import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts import build_memory_sft_wave2 as wave
from scripts import prepare_v26_answer_candidates as candidates
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected

FIRST = base.WORK / 'reports/memory-harness-v2'
STATE = candidates.STATE / 'raw-overlap-recovery-v1'


def init():
    global EXISTING
    wave.init()
    original = json.loads((FIRST / 'exclusions.private.json').read_text())
    holds = json.loads((candidates.STATE / 'candidate-holds.private.json').read_text())
    base.EX['old_query_hashes'] = (set(original['old_query_hashes'])
                                 | set(holds['query_hashes'])
                                 | set(holds['supplemental_prior_queries']))
    candidates.init()
    EXISTING = sqlite3.connect(f'file:{candidates.STATE / "candidates.sqlite"}?mode=ro', uri=True)


def job(j):
    receipt = STATE / 'normalized' / (j['job_id'] + '.report.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    exclusions = wave.STATE / 'normalized' / (j['job_id'] + '.excluded.jsonl')
    wanted = {r['row_index'] for line in exclusions.open()
              if (r := json.loads(line))['reason'] == 'prior_wave_or_held_query'}
    counts = Counter(); rows = []; seen = set()
    if wanted:
        file = pq.ParquetFile(j['read_local'])
        table = file.read() if j['row_group'] is None else file.read_row_group(j['row_group'])
        indices = sorted(wanted)
        table = table.take([i - j['start'] for i in indices])
        for index, raw in zip(indices, table.to_pylist()):
            counts['seen_query_overlap'] += 1
            try:
                row = wave.normalize_row(raw, j['source'], index)
                candidates.eligible(row)
                normalized = candidates.canonicalize(copy.deepcopy(row))
            except Rejected as exc:
                counts['held:' + str(exc)] += 1
                continue
            identity = normalized['candidate_identity']
            if identity in seen or EXISTING.execute(
                    'SELECT 1 FROM candidates WHERE candidate_identity=?', (identity,)).fetchone():
                counts['already_indexed_same_answer'] += 1
                continue
            seen.add(identity); rows.append(row); counts['new_eligible_answer_variant'] += 1
    path = receipt.with_name(j['job_id'] + '.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(canonical(row) + '\n')
    result = {'job_id': j['job_id'], 'counts': dict(counts), 'path': str(path),
              'sha256': base.file_sha(path), 'source': j['source']['repo']}
    base.save(receipt, result)
    return result


def main(workers):
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'summary.json').exists():
        raise ValueError('recovery already complete')
    jobs = json.loads((wave.STATE / 'normalize-jobs.json').read_text())
    # Check all original source bytes before processing any new candidate.
    checked = set()
    for j in jobs:
        if j['read_local'] in checked:
            continue
        if base.file_sha(j['read_local']) != j['read_sha256']:
            raise ValueError('frozen raw source changed')
        checked.add(j['read_local'])
    counts = Counter(); files = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(job, jobs), 1):
            counts.update(result['counts'])
            if result['counts'].get('new_eligible_answer_variant'):
                files.append({'path': result['path'], 'sha256': result['sha256']})
            if n % 100 == 0 or n == len(jobs):
                print(canonical({'jobs': n, 'total': len(jobs), 'counts': dict(counts)}), flush=True)
    base.save(STATE / 'summary.json', {'counts': dict(counts), 'files': files,
        'status': 'STAGED_NOT_SCORED_OR_ACCEPTED', 'source_programs_executed': False,
        'original_quality_legacy_and_first_negative_holds_preserved': True,
        'prior_query_dedup_only_is_not_an_answer_quality_gate': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('CPU worker bound')
    main(args.workers)
