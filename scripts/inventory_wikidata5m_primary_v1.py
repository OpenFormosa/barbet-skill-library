"""Inventory original inductive graph/aliases as data; admit no SFT episodes."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile

import numpy as np

from scripts import build_memory_sft_v2 as base
from scripts.fetch_wikidata5m_primary_v1 import STATE as SOURCE
from scripts.normalize_ifm_math_memory_v1 import processed_bytes

STATE = SOURCE / 'inventory-v1'
ID = re.compile(rb'([QP])([1-9][0-9]*)\Z')


def identifier(value, prefix):
    match = ID.fullmatch(value)
    if match is None or match[1] != prefix:
        raise ValueError('nonliteral source entity/property identifier')
    return int(match[2])


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'inventory-policy.json').exists():
        raise ValueError('source inventory already started; inspect first state')
    lock_path = SOURCE / 'source-lock.json'; lock = json.loads(lock_path.read_text())
    resource_path = base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json'
    resources = json.loads(resource_path.read_text())['resources']; reserve = 8 * 1024 ** 3
    used = processed_bytes(); free = shutil.disk_usage(base.WORK).free
    if used + reserve > resources['new_processed_cap_bytes'] or free < resources['minimum_free_disk_bytes'] + reserve:
        raise ValueError('source inventory storage bound')
    base.save(STATE / 'inventory-policy.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'source_lock_sha256': base.file_sha(lock_path), 'resources_sha256': base.file_sha(resource_path),
        'reserved_stage_bytes': reserve, 'processed_bytes_before': used, 'free_bytes_before': free,
        'only_inductive_training_graph_may_become_training_evidence': True,
        'validation_and_test_entities_excluded_from_future_tasks': True,
        'wikipedia_prose_not_used': True, 'no_generated_questions': True,
        'accepted_training_rows': 0, 'source_programs_executed': False, 'model_requests': 0})
    graph = next(f for f in lock['files'] if f['local'].endswith('inductive.tar.gz'))
    aliases = next(f for f in lock['files'] if f['local'].endswith('alias.tar.gz'))
    split_stats = {}; entities = {}; arrays = {}
    for source in [graph, aliases]:
        if base.file_sha(source['local']) != source['sha256']:
            raise ValueError('complete source archive differs')
    expected_counts = {'train': 20496514, 'valid': 6699, 'test': 6894}
    with tarfile.open(graph['local'], 'r:gz') as tar:
        for member in tar:
            split = member.name.removeprefix('wikidata5m_inductive_').removesuffix('.txt')
            if split not in expected_counts: raise ValueError('unexpected inductive member')
            expected = next(m for m in graph['members'] if m['name'] == member.name)
            array = np.lib.format.open_memmap(STATE / (split + '-triples.npy'), mode='w+', dtype='<i8', shape=(expected_counts[split], 3))
            digest = hashlib.sha256(); count = 0
            stream = tar.extractfile(member)
            for lines in iter(lambda: list(itertools.islice(stream, 100000)), []):
                triples = []
                for line in lines:
                    digest.update(line); cells = line.rstrip(b'\r\n').split(b'\t')
                    if len(cells) != 3: raise ValueError('source triple column mismatch')
                    triples.append((identifier(cells[0], b'Q'), identifier(cells[1], b'P'), identifier(cells[2], b'Q')))
                if count + len(triples) > len(array): raise ValueError('source exceeds declared triple count')
                array[count:count+len(triples)] = triples; count += len(triples)
            if count != len(array) or digest.hexdigest() != expected['sha256']:
                raise ValueError('source member count or complete digest differs')
            array.flush(); arrays[split] = array
            entity_ids = np.unique(np.concatenate((array[:, 0], array[:, 2])))
            np.save(STATE / (split + '-entities.npy'), entity_ids); entities[split] = entity_ids
            relations, rel_counts = np.unique(array[:, 1], return_counts=True)
            order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
            sorted_rows = array[order]
            duplicates = int(np.all(sorted_rows[1:] == sorted_rows[:-1], axis=1).sum())
            split_stats[split] = {'triples': count, 'distinct_triples': count-duplicates, 'duplicate_triples': duplicates,
                'entities': len(entity_ids), 'relations': len(relations), 'source_member_sha256': digest.hexdigest(),
                'relation_counts': {str(int(k)): int(v) for k, v in zip(relations, rel_counts)}}
            if split == 'train':
                np.save(STATE / 'train-head-order.npy', order)
                heads, starts, degrees = np.unique(sorted_rows[:, 0], return_index=True, return_counts=True)
                np.save(STATE / 'train-heads.npy', heads)
                np.save(STATE / 'train-head-offsets.npy', np.r_[starts, len(array)])
                tail_order = np.lexsort((array[:, 0], array[:, 1], array[:, 2]))
                np.save(STATE / 'train-tail-order.npy', tail_order)
                tails, tail_starts, tail_degrees = np.unique(array[tail_order, 2], return_index=True, return_counts=True)
                np.save(STATE / 'train-tails.npy', tails)
                np.save(STATE / 'train-tail-offsets.npy', np.r_[tail_starts, len(array)])
                base.save(STATE / 'train-degree-statistics.json', {
                    'subjects': len(heads), 'objects': len(tails), 'source_entities': len(entity_ids),
                    'out_degree_one': int((degrees == 1).sum()), 'out_degree_two_to_eight': int(((degrees >= 2) & (degrees <= 8)).sum()),
                    'out_degree_nine_to_64': int(((degrees >= 9) & (degrees <= 64)).sum()),
                    'out_degree_above_64': int((degrees > 64).sum()), 'maximum_out_degree': int(degrees.max()),
                    'maximum_in_degree': int(tail_degrees.max()),
                    'one_entity_is_not_a_new_reasoning_skill': True,
                    'task_path_and_template_diversity_unmeasured': True})
            print(json.dumps({'split': split, **{k:v for k,v in split_stats[split].items() if k != 'relation_counts'}}), flush=True)
            del sorted_rows, order
    overlap = {a + '_' + b: int(len(np.intersect1d(entities[a], entities[b], assume_unique=True)))
               for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
    if any(overlap.values()): raise ValueError('inductive entity sets overlap; no future admission allowed')
    db_path = STATE / 'aliases.sqlite'; db = sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
    db.execute('CREATE TABLE entities(id INTEGER PRIMARY KEY, label TEXT NOT NULL, aliases_json TEXT NOT NULL, source_line INTEGER NOT NULL)')
    db.execute('CREATE TABLE relations(id INTEGER PRIMARY KEY, label TEXT NOT NULL, aliases_json TEXT NOT NULL, source_line INTEGER NOT NULL)')
    alias_stats = {}
    with tarfile.open(aliases['local'], 'r:gz') as tar:
        for member in tar:
            kind = {'wikidata5m_entity.txt': 'entities', 'wikidata5m_relation.txt': 'relations'}[member.name]
            prefix = b'Q' if kind == 'entities' else b'P'; digest = hashlib.sha256(); count = 0; alias_count = 0
            expected = next(m for m in aliases['members'] if m['name'] == member.name)
            stream = tar.extractfile(member)
            for lines in iter(lambda: list(itertools.islice(stream, 10000)), []):
                rows = []
                for line in lines:
                    digest.update(line); cells = line.rstrip(b'\r\n').split(b'\t'); count += 1
                    if len(cells) < 2: raise ValueError('source alias column mismatch')
                    eid = identifier(cells[0], prefix); labels = [s.decode('utf-8') for s in cells[1:]]
                    if not labels[0]: raise ValueError('empty primary label')
                    alias_count += len(labels)
                    rows.append((eid, labels[0], json.dumps(labels, ensure_ascii=False, separators=(',', ':')), count))
                db.executemany('INSERT INTO ' + kind + ' VALUES(?,?,?,?)', rows)
                db.commit()
            if digest.hexdigest() != expected['sha256']: raise ValueError('alias member bytes changed')
            alias_stats[kind] = {'ids': count, 'aliases_including_first': alias_count, 'member_sha256': digest.hexdigest()}
            print(json.dumps({'aliases': kind, **alias_stats[kind]}), flush=True)
    alias_ids = np.fromiter((r[0] for r in db.execute('SELECT id FROM entities ORDER BY id')), dtype=np.int64)
    missing_aliases = {split: int(len(np.setdiff1d(ids, alias_ids, assume_unique=True))) for split, ids in entities.items()}
    relation_ids = {r[0] for r in db.execute('SELECT id FROM relations')}
    missing_relations = {split: sorted(set(map(int, value['relation_counts'])) - relation_ids) for split,value in split_stats.items()}
    db.close()
    if any(missing_aliases.values()) or any(missing_relations.values()):
        raise ValueError('incomplete alias coverage')
    files = [{'path': str(p.relative_to(STATE)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
             for p in sorted(STATE.iterdir()) if p.is_file()]
    if sum(p['bytes'] for p in files) > reserve: raise ValueError('inventory exceeds reserved stage bytes')
    result = {'splits': split_stats, 'inductive_entity_intersections': overlap,
        'aliases': alias_stats, 'missing_entity_aliases': missing_aliases, 'missing_relation_aliases': missing_relations,
        'files': files, 'all_member_hashes_verified_again': True, 'source_lock_sha256': base.file_sha(lock_path),
        'inventory_policy_sha256': base.file_sha(STATE / 'inventory-policy.json'),
        'only_training_graph_allowed_for_future_task_evidence': True, 'generated_questions': 0,
        'accepted_training_rows': 0, 'model_requests': 0, 'source_programs_executed': False,
        'wikipedia_prose_used': False, 'new_task_contract_first_diagnosis_and_exact_verifier_still_required': True}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({'inventory_complete': True, 'training_entities': split_stats['train']['entities'],
                      'training_triples': split_stats['train']['triples'], 'accepted_training_rows': 0}), flush=True)


if __name__ == '__main__': main()
