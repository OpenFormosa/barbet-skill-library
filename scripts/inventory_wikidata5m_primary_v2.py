"""Finish source inventory with explicit alias gaps; admit no training rows.

The original all-labels assertion and archive bytes remain unchanged. Missing
entity/property labels are not invented: all incident facts are excluded from
the prospective usable graph. This is still source inspection, not admission.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import tarfile

import numpy as np

from scripts import build_memory_sft_v2 as base
from scripts import inventory_wikidata5m_primary_v1 as original

STATE = original.SOURCE / 'inventory-v2'
PRIOR = original.STATE


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'inventory-policy.json').exists(): raise ValueError('prospective inventory already started')
    first_log = base.WORK / 'reports/wikidata5m-primary-inventory-v1.log'
    if (PRIOR / 'summary.json').exists() or 'ValueError: incomplete alias coverage' not in first_log.read_text():
        raise ValueError('first inventory failure boundary differs')
    lock_path = original.SOURCE / 'source-lock.json'; lock = json.loads(lock_path.read_text())
    prior_policy = json.loads((PRIOR / 'inventory-policy.json').read_text())
    if prior_policy['driver_sha256'] != base.file_sha(Path(original.__file__)):
        raise ValueError('first inventory driver changed')
    prior_files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                   for p in sorted(PRIOR.iterdir()) if p.is_file()]
    base.save(STATE / 'inventory-policy.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'first_driver_sha256': base.file_sha(Path(original.__file__)),
        'first_failure_log_sha256': base.file_sha(first_log), 'first_policy_sha256': base.file_sha(PRIOR / 'inventory-policy.json'),
        'first_artifacts': prior_files, 'source_lock_sha256': base.file_sha(lock_path),
        'scope': 'Recheck every graph row against original archive bytes; retain missing-alias evidence and exclude all facts incident on unlabeled entities/properties. No source values are repaired or invented.',
        'training_rows_admitted': 0, 'model_requests': 0, 'first_failed_check_preserved': True})
    arrays = {}; entities = {}; stats = {}
    graph = next(f for f in lock['files'] if f['local'].endswith('inductive.tar.gz'))
    if base.file_sha(graph['local']) != graph['sha256']: raise ValueError('source graph changed')
    with tarfile.open(graph['local'], 'r:gz') as tar:
        for member in tar:
            split = member.name.removeprefix('wikidata5m_inductive_').removesuffix('.txt')
            array = np.load(PRIOR / (split + '-triples.npy'), mmap_mode='r'); digest = hashlib.sha256(); offset = 0
            stream = tar.extractfile(member)
            for lines in iter(lambda: list(itertools.islice(stream, 100000)), []):
                parsed = []
                for line in lines:
                    digest.update(line); h, r, t = line.rstrip(b'\r\n').split(b'\t')
                    parsed.append((original.identifier(h, b'Q'), original.identifier(r, b'P'), original.identifier(t, b'Q')))
                batch = np.asarray(parsed, dtype=np.int64)
                if not np.array_equal(batch, array[offset:offset+len(batch)]):
                    raise ValueError('cached triple array differs from original source row order')
                offset += len(batch)
            expected = next(m for m in graph['members'] if m['name'] == member.name)
            if offset != len(array) or digest.hexdigest() != expected['sha256']:
                raise ValueError('source graph row/hash coverage incomplete')
            arrays[split] = array; entities[split] = np.load(PRIOR / (split + '-entities.npy'), mmap_mode='r')
            relations, rel_counts = np.unique(array[:, 1], return_counts=True)
            stats[split] = {'triples': len(array), 'entities': len(entities[split]), 'relations': len(relations),
                            'relation_counts': {str(int(k)): int(v) for k, v in zip(relations, rel_counts)},
                            'member_sha256': digest.hexdigest(), 'all_cached_rows_equal_source': True}
            print(json.dumps({'verified_split': split, 'rows': len(array)}), flush=True)
    overlaps = {a + '_' + b: int(len(np.intersect1d(entities[a], entities[b], assume_unique=True)))
                for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
    if any(overlaps.values()): raise ValueError('inductive entity overlap')
    db = sqlite3.connect(f'file:{PRIOR / "aliases.sqlite"}?mode=ro', uri=True)
    aliases = np.fromiter((r[0] for r in db.execute('SELECT id FROM entities ORDER BY id')), dtype=np.int64)
    relation_ids = {r[0] for r in db.execute('SELECT id FROM relations')}
    coverage = {}
    for split, array in arrays.items():
        missing = np.setdiff1d(entities[split], aliases, assume_unique=True)
        missing_relations = sorted(set(map(int, stats[split]['relation_counts'])) - relation_ids)
        coverage[split] = {'missing_entity_ids': missing.tolist(), 'missing_relation_ids': missing_relations,
                           'missing_entities': len(missing), 'missing_relations': len(missing_relations)}
    db.close(); train = arrays['train']; missing = np.asarray(coverage['train']['missing_entity_ids'], dtype=np.int64)
    usable = ~(np.isin(train[:, 0], missing) | np.isin(train[:, 2], missing)
               | np.isin(train[:, 1], coverage['train']['missing_relation_ids']))
    np.save(STATE / 'train-label-complete-mask.npy', usable)
    good = train[usable]; order = np.lexsort((good[:, 2], good[:, 1], good[:, 0])); ordered = good[order]
    duplicates = int(np.all(ordered[1:] == ordered[:-1], axis=1).sum())
    if duplicates: raise ValueError('usable graph repeats literal triples')
    # Offsets refer to the original source-row array via this retained index.
    source_rows = np.flatnonzero(usable); head_order = source_rows[order]
    np.save(STATE / 'train-usable-head-order.npy', head_order)
    heads, starts, degrees = np.unique(ordered[:, 0], return_index=True, return_counts=True)
    np.save(STATE / 'train-usable-heads.npy', heads)
    np.save(STATE / 'train-usable-head-offsets.npy', np.r_[starts, len(ordered)])
    usable_entities = np.unique(np.concatenate((good[:, 0], good[:, 2])))
    np.save(STATE / 'train-usable-entities.npy', usable_entities)
    second = np.lexsort((good[:, 0], good[:, 1], good[:, 2]))
    np.save(STATE / 'train-usable-tail-order.npy', source_rows[second])
    tails, tail_starts = np.unique(good[second, 2], return_index=True)
    np.save(STATE / 'train-usable-tails.npy', tails)
    np.save(STATE / 'train-usable-tail-offsets.npy', np.r_[tail_starts, len(good)])
    base.save(STATE / 'alias-coverage-exclusions.json', coverage)
    result = {'source_splits': stats, 'inductive_entity_intersections': overlaps,
        'source_entity_alias_ids': len(aliases), 'source_relation_alias_ids': len(relation_ids),
        'usable_training_triples': len(good), 'excluded_training_triples': len(train)-len(good),
        'usable_training_entities': len(usable_entities), 'usable_training_subjects': len(heads),
        'usable_training_objects': len(tails), 'usable_duplicate_triples': duplicates,
        'out_degree_one': int((degrees == 1).sum()), 'out_degree_two_to_eight': int(((degrees >= 2) & (degrees <= 8)).sum()),
        'out_degree_nine_to_64': int(((degrees >= 9) & (degrees <= 64)).sum()), 'out_degree_above_64': int((degrees > 64).sum()),
        'coverage_exclusions_sha256': base.file_sha(STATE / 'alias-coverage-exclusions.json'),
        'policy_sha256': base.file_sha(STATE / 'inventory-policy.json'),
        'files': [{'path': str(p.relative_to(STATE)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                  for p in sorted(STATE.iterdir()) if p.is_file()],
        'source_programs_executed': False, 'generated_questions': 0, 'accepted_training_rows': 0,
        'model_requests': 0, 'historical_graph_facts_are_not_assertions_of_current_world_truth': True,
        'task_contract_and_first_diagnosis_still_required': True}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_splits', 'files']}), flush=True)


if __name__ == '__main__': main()
