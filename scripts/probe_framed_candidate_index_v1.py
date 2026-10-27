"""Compare actual index/episode algorithms with only the storage reader changed."""
import json
from pathlib import Path
import sqlite3

from barbet_inference import candidate_archive as archive
from scripts import build_memory_sft_v2 as base
from scripts import prepare_v26_answer_candidates as candidates
from scripts import guard_v26_episode_identity as identity
from scripts import framed_candidate_io_v1 as framed

STATE = base.WORK / 'reports/memory-harness-v2-candidate-archive-v1/index-equivalence-v1'
SOURCE = STATE.parent / 'source-records.bca'


def values(path, query):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    result = db.execute(query).fetchall(); db.close()
    return result


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'plan.json').exists():
        raise ValueError('index equivalence probe already started')
    base.save(STATE / 'plan.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'new_io_sha256': base.file_sha(Path(framed.__file__)), 'archive_sha256': base.file_sha(Path(archive.__file__)),
        'old_index_sha256': base.file_sha(Path(candidates.__file__)), 'old_guard_sha256': base.file_sha(Path(identity.__file__)),
        'source_archive_sha256': base.file_sha(SOURCE), 'accepted_training_rows': 0,
        'all_source_samples_are_diagnostics_only': True, 'semantic_holds_not_reconsidered': True})
    plain = STATE / 'identical-source-records.jsonl'
    with plain.open('xb') as handle:
        for unused_offset, unused_length, raw in archive.iter_records(SOURCE):
            handle.write(raw)
    original_index = candidates.index_file; original_guard = identity.job
    for label, path, reader in [('jsonl', plain, original_index), ('framed', SOURCE, framed.index_file)]:
        root = STATE / label; root.mkdir()
        spec = {'file_id': 0, 'path': str(path), 'sha256': base.file_sha(path)}
        if label == 'framed':
            spec['format'] = archive.FORMAT
        base.save(root / 'candidate-input-files.json', [spec])
        base.save(root / 'candidate-holds.private.json', {'record_ids': [], 'query_hashes': [], 'search_allowed': [], 'supplemental_prior_queries': []})
        candidates.STATE = root; candidates.index_file = reader; candidates.index(1)
        identity.STATE = root / 'episode-identity-v1'; identity.job = original_guard if label == 'jsonl' else framed.guard_file
        identity.main()
    left = STATE / 'jsonl'; right = STATE / 'framed'
    checks = {}
    columns = 'candidate_identity,record_id,original_record_id,input_hash,bucket,domain,query_hash,source_family_id,template_family_id,source'
    for name, query, dbfile in [
        ('all_candidate_metadata_except_storage_positions', 'SELECT '+columns+' FROM candidates ORDER BY candidate_identity', 'candidates.sqlite'),
        ('all_input_groups', 'SELECT * FROM groups ORDER BY input_hash', 'candidates.sqlite'),
        ('all_episode_conditions', 'SELECT * FROM cases ORDER BY candidate_identity', 'episode-identity-v1/identity.sqlite'),
        ('all_prefixes', 'SELECT * FROM prefixes ORDER BY candidate_identity', 'episode-identity-v1/identity.sqlite'),
        ('all_structural_holds', 'SELECT * FROM holds ORDER BY candidate_identity', 'episode-identity-v1/identity.sqlite')]:
        a = values(left / dbfile, query); b = values(right / dbfile, query)
        if a != b:
            raise ValueError('storage changed candidate semantics: ' + name)
        checks[name] = {'equal': True, 'rows': len(a)}
    result = {'checks': checks, 'all_checks_passed': True, 'plan_sha256': base.file_sha(STATE / 'plan.json'),
        'accepted_training_rows': 0, 'model_requests': 0, 'scores_created': 0,
        'scope': 'Unchanged full index and episode-identity algorithms on the 86 prior storage-probe rows with explicit old/new byte readers. No semantic reevaluation or training admission.'}
    base.save(STATE / 'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
