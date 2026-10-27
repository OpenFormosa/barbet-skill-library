"""Add raw query-overlap alternatives without discarding completed v26 scores.

The old run remains immutable. Scores are reusable only for exactly the same
candidate pointer, source bytes and numerical protocol. New groups and previously
unscored singletons receive actual inference; candidate counts are never scores.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from barbet_inference.memory import canonical

ORIGINAL = candidates.STATE
RECOVERY = ORIGINAL / 'raw-overlap-recovery-v1'
STATE = base.WORK / 'reports/memory-harness-v2-v26-complete-selection-v1'


def configure():
    candidates.STATE = STATE


def prepare():
    if (STATE / 'candidate-input-files.json').exists():
        raise ValueError('extension preparation already frozen')
    STATE.mkdir(parents=True, exist_ok=True)
    recovery = json.loads((RECOVERY / 'summary.json').read_text())
    files = json.loads((ORIGINAL / 'candidate-input-files.json').read_text())
    start = len(files)
    for item in recovery['files']:
        files.append({**item, 'file_id': len(files), 'wave': 'recovered-prior-query-overlap'})
    for name in ['candidate-holds.private.json', 'canonical-single-v3-validation.json',
                 'live-calibration-diagnostic-v3.json', 'remote-v26-tag-check.json']:
        shutil.copyfile(ORIGINAL / name, STATE / name)
    base.save(STATE / 'candidate-input-files.json', files)
    base.save(STATE / 'candidate-policy-binding.json', {
        'original_policy_binding_sha256': base.file_sha(ORIGINAL / 'candidate-policy-binding.json'),
        'recovery_summary_sha256': base.file_sha(RECOVERY / 'summary.json'),
        'extension_script_sha256': base.file_sha(Path(__file__)),
        'new_file_id_start': start, 'source_order_not_used_for_answer_selection': True,
        'old_first_holds_unchanged': True, 'raw_quality_failures_not_revived': True})
    # Existing identity results are exact, already verified projections of the
    # unchanged manifest prefix. Reuse receipts, retaining original file paths.
    dest = STATE / 'episode-identity-v1/parts'; dest.mkdir(parents=True)
    for path in (ORIGINAL / 'episode-identity-v1/parts').glob('*.json'):
        shutil.copyfile(path, dest / path.name)
    print(canonical({'original_files': start, 'recovered_files': len(files) - start,
                     'raw_recovery_counts': recovery['counts']}), flush=True)


def index(workers):
    configure()
    if (STATE / 'candidate-index-summary.json').exists():
        raise ValueError('extended index already complete')
    binding = json.loads((STATE / 'candidate-policy-binding.json').read_text())
    if binding['extension_script_sha256'] != base.file_sha(Path(__file__)):
        raise ValueError('extension code changed after preparation')
    files = json.loads((STATE / 'candidate-input-files.json').read_text())
    prefix = json.loads((ORIGINAL / 'candidate-input-files.json').read_text())
    if files[:len(prefix)] != prefix:
        raise ValueError('existing score input pointers changed')
    parts = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=candidates.init) as pool:
        for part in pool.map(candidates.index_file, files[len(prefix):]):
            parts.append(part); counts.update(part['counts'])
    path = STATE / 'candidates.sqlite'
    if path.exists():
        raise ValueError('partial extended index exists; inspect before recovery')
    source = sqlite3.connect(f'file:{ORIGINAL / "candidates.sqlite"}?mode=ro', uri=True)
    db = sqlite3.connect(path); source.backup(db); source.close()
    before = db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
    for part in parts:
        if not part['path']:
            continue
        if base.file_sha(part['path']) != part['sha256']:
            raise ValueError('new candidate index bytes changed')
        rows = pq.read_table(part['path']).to_pylist()
        db.executemany('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       [tuple(r.values()) for r in rows]); db.commit()
    db.execute('DROP TABLE groups')
    db.execute('CREATE TABLE groups AS SELECT input_hash,MIN(bucket) AS bucket,COUNT(*) AS candidates FROM candidates GROUP BY input_hash')
    db.execute('CREATE UNIQUE INDEX group_identity ON groups(input_hash)')
    db.execute('CREATE INDEX group_buckets ON groups(bucket,candidates)'); db.commit()
    old = json.loads((ORIGINAL / 'candidate-index-summary.json').read_text())
    total_counts = Counter(old['counts']); total_counts.update(counts)
    result = {'counts': dict(total_counts),
        'distinct_answers': db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
        'input_groups': db.execute('SELECT COUNT(*) FROM groups').fetchone()[0],
        'competing_groups': db.execute('SELECT COUNT(*) FROM groups WHERE candidates>1').fetchone()[0],
        'answers_to_score': db.execute('SELECT SUM(candidates) FROM groups WHERE candidates>1').fetchone()[0],
        'additional_distinct_answers': db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] - before,
        'status': 'INDEXED_NOT_YET_PPL_SELECTED', 'all_score_inputs_are_eligible_only': True,
        'no_model_training': True, 'raw_overlap_recovery': json.loads((RECOVERY / 'summary.json').read_text())['counts'],
        'extension_binding_sha256': base.file_sha(STATE / 'candidate-policy-binding.json'),
        'numerical_scores_can_be_reused_only_for_unchanged_original_candidates': True}
    db.close(); base.save(STATE / 'candidate-index-summary.json', result)
    print(canonical(result), flush=True)


def score_binding(bucket):
    return {'policy_sha256': base.file_sha(base.REPO / 'configs/memory_sft_v2_v26_selection.json'),
        'candidate_manifest_sha256': base.file_sha(STATE / 'candidate-input-files.json'),
        'candidate_summary_sha256': base.file_sha(STATE / 'candidate-index-summary.json'),
        'bucket': bucket, 'limit': 0,
        'scorer_sha256': base.file_sha(base.REPO / 'barbet_inference/answer_selection.py'),
        'runner_sha256': base.file_sha(base.REPO / 'scripts/score_v26_answer_candidates.py'),
        'adapter_sha256': base.file_sha(base.REPO / 'scripts/prepare_v26_answer_candidates.py'),
        'canonical_batch_size': 1, 'model_revision': '67131067ff1825179d5e38e91d499159669e9c66'}


def protocol_hash(binding):
    return hashlib.sha256(canonical({k: v for k, v in binding.items() if k not in ['bucket', 'limit']}).encode()).hexdigest()


def rebind_score(score, old_binding, new_binding, original_db_sha256):
    for key in ['policy_sha256', 'scorer_sha256', 'runner_sha256', 'adapter_sha256',
                'canonical_batch_size', 'model_revision']:
        if old_binding[key] != new_binding[key]:
            raise ValueError('numerical reuse protocol changed: ' + key)
    if score['protocol_sha256'] != protocol_hash(old_binding):
        raise ValueError('cached score does not match its original protocol')
    # This is a new metadata object, not an edit to the immutable old score.
    return {**score, 'protocol_sha256': protocol_hash(new_binding),
        'reused_from': {'protocol_sha256': score['protocol_sha256'],
                        'score_database_sha256': original_db_sha256,
                        'numerical_values_unchanged': True}}


def seed():
    folder = STATE / 'scores/canonical-v2'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'reuse-summary.json').exists():
        raise ValueError('score reuse already complete')
    # Bind every original normalized file again before reusing any likelihood.
    original_files = json.loads((ORIGINAL / 'candidate-input-files.json').read_text())
    extended_files = json.loads((STATE / 'candidate-input-files.json').read_text())
    if extended_files[:len(original_files)] != original_files:
        raise ValueError('original candidate source pointers changed')
    for item in original_files:
        if base.file_sha(item['path']) != item['sha256']:
            raise ValueError('original scored source bytes changed')
    receipts = []; totals = Counter()
    for bucket in range(8):
        original = ORIGINAL / 'scores/canonical-v2'
        completed = json.loads((original / f'bucket-{bucket}.summary.json').read_text())
        if completed['status'] != 'BUCKET_COMPLETE' or completed['limited_pilot']:
            raise ValueError('cannot reuse incomplete source run')
        old_binding = json.loads((original / f'bucket-{bucket}.binding.json').read_text())
        new_binding = score_binding(bucket); new_protocol = protocol_hash(new_binding)
        old_sha = base.file_sha(original / f'bucket-{bucket}.sqlite')
        db_path = folder / f'bucket-{bucket}.sqlite'
        if db_path.exists():
            raise ValueError('partial reuse bucket exists; do not overwrite')
        db = sqlite3.connect(db_path)
        db.execute('CREATE TABLE scores(candidate_identity TEXT PRIMARY KEY,input_hash TEXT,record_id TEXT,score_json TEXT)')
        db.execute('CREATE TABLE decisions(input_hash TEXT PRIMARY KEY,status TEXT,winner TEXT,report_json TEXT)')
        db.execute('ATTACH DATABASE ? AS oldscores', (str(original / f'bucket-{bucket}.sqlite'),))
        db.execute('ATTACH DATABASE ? AS oldinputs', (str(ORIGINAL / 'candidates.sqlite'),))
        db.execute('ATTACH DATABASE ? AS inputs', (str(STATE / 'candidates.sqlite'),))
        # Appending new answers must leave each old score's exact source pointer
        # unchanged, including literal provider IDs in the native rendered text.
        mismatch = db.execute('''SELECT COUNT(*) FROM oldscores.scores s
            LEFT JOIN oldinputs.candidates o USING(candidate_identity)
            LEFT JOIN inputs.candidates n USING(candidate_identity)
            WHERE o.candidate_identity IS NULL OR n.candidate_identity IS NULL
              OR o.file_id != n.file_id OR o.offset != n.offset OR o.length != n.length
              OR s.input_hash != n.input_hash OR s.record_id != n.record_id''').fetchone()[0]
        if mismatch:
            raise ValueError('cached candidate pointer changed')
        copied = 0
        cursor = db.execute('SELECT * FROM oldscores.scores')
        while rows := cursor.fetchmany(2000):
            values = [(a, b, c, canonical(rebind_score(json.loads(d), old_binding, new_binding, old_sha)))
                      for a, b, c, d in rows]
            db.executemany('INSERT INTO scores VALUES (?,?,?,?)', values); copied += len(values)
        resumed = 0
        for ih, status, winner, raw in db.execute('''SELECT d.* FROM oldscores.decisions d
                JOIN oldinputs.groups o USING(input_hash) JOIN inputs.groups n USING(input_hash)
                WHERE o.candidates=n.candidates''').fetchall():
            report = json.loads(raw)
            report.update(protocol_sha256=new_protocol, reused_original_decision=True,
                          original_score_database_sha256=old_sha)
            db.execute('INSERT INTO decisions VALUES (?,?,?,?)', (ih, status, winner, canonical(report))); resumed += 1
        db.commit(); db.close()
        base.save(folder / f'bucket-{bucket}.binding.json', new_binding)
        receipts.append({'bucket': bucket, 'reused_scores': copied, 'unchanged_group_decisions': resumed,
                         'original_database_sha256': old_sha, 'protocol_sha256': new_protocol})
        totals.update(reused_scores=copied, unchanged_group_decisions=resumed)
        print(canonical(receipts[-1]), flush=True)
    base.save(folder / 'reuse-summary.json', {'buckets': receipts, 'counts': dict(totals),
        'all_original_source_bytes_rechecked': True, 'all_original_pointers_unchanged': True,
        'old_artifacts_unchanged': True, 'new_candidates_require_actual_inference': True})


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'index', 'guard', 'seed', 'score'])
    p.add_argument('--workers', type=int, default=16); p.add_argument('--bucket', type=int)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('CPU worker bound')
    configure()
    if args.phase == 'prepare': prepare()
    elif args.phase == 'index': index(args.workers)
    elif args.phase == 'seed': seed()
    elif args.phase == 'guard':
        from scripts import guard_v26_episode_identity as guard
        guard.STATE = STATE / 'episode-identity-v1'; guard.main()
    else:
        if args.bucket is None or not 0 <= args.bucket < 8:
            raise ValueError('scoring requires a bucket from 0 to 7')
        from scripts import score_v26_answer_candidates as scorer
        scorer.STATE = STATE; scorer.main(args.bucket, 0, 'canonical-v2')


if __name__ == '__main__':
    main()
