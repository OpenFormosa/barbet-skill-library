"""Measure bounded relation-join capacity before making any training examples."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from scripts import build_memory_sft_v2 as base
from scripts.inventory_wikidata5m_primary_v2 import STATE as INVENTORY, PRIOR

STATE = INVENTORY.parent / 'path-capacity-v1'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'policy.json').exists(): raise ValueError('capacity probe already started')
    summary = json.loads((INVENTORY / 'summary.json').read_text())
    for item in summary['files']:
        if base.file_sha(INVENTORY / item['path']) != item['sha256']:
            raise ValueError('completed source inventory changed')
    base.save(STATE / 'policy.json', {'driver_sha256': base.file_sha(Path(__file__)),
        'inventory_summary_sha256': base.file_sha(INVENTORY / 'summary.json'),
        'max_complete_anchor_facts': 32, 'max_complete_intermediate_facts': 24,
        'scope': 'Whole-source capacity of outgoing two-hop joins with complete bounded adjacency. Self-loop-only paths and immediate return to anchor do not count. No task, target or training row is created.',
        'generated_questions': 0, 'accepted_training_rows': 0, 'model_requests': 0})
    triples = np.load(PRIOR / 'train-triples.npy', mmap_mode='r')
    heads = np.load(INVENTORY / 'train-usable-heads.npy', mmap_mode='r')
    offsets = np.load(INVENTORY / 'train-usable-head-offsets.npy', mmap_mode='r')
    order = np.load(INVENTORY / 'train-usable-head-order.npy', mmap_mode='r')
    degrees = np.diff(offsets); good_anchors = (degrees <= 32)
    anchor_has_join = np.zeros(len(heads), dtype=bool)
    anchor_has_specific_join = np.zeros(len(heads), dtype=bool)
    route_counts = Counter(); eligible_first_edges = 0; complete_second_edges = 0
    generic = {31, 279}; batch_size = 250000
    # Per-source-entity route family counts avoid multiplying by many paths.
    family_keys = []
    for start in range(0, len(order), batch_size):
        indexes = order[start:start+batch_size]; rows = triples[indexes]
        hi = np.searchsorted(heads, rows[:, 0]); ti = np.searchsorted(heads, rows[:, 2])
        found = ti < len(heads); safe_ti = np.minimum(ti, len(heads)-1)
        found &= heads[safe_ti] == rows[:, 2]
        candidates = found & good_anchors[hi] & (degrees[safe_ti] <= 24) & (rows[:, 0] != rows[:, 2])
        for h, r, t, hp, tp in zip(rows[candidates, 0], rows[candidates, 1], rows[candidates, 2], hi[candidates], safe_ti[candidates]):
            second = triples[order[offsets[tp]:offsets[tp+1]]]
            second = second[(second[:, 2] != h) & (second[:, 2] != t)]
            if not len(second): continue
            anchor_has_join[hp] = True; eligible_first_edges += 1; complete_second_edges += len(second)
            if int(r) not in generic and np.any(~np.isin(second[:, 1], list(generic))):
                anchor_has_specific_join[hp] = True
            # Aggregate path signatures for a bounded source diagnosis, not a
            # promise to generate one row per raw path or per signature.
            for r2 in np.unique(second[:, 1]): route_counts[(int(r), int(r2))] += 1
        if start % (batch_size * 8) == 0:
            print(json.dumps({'first_edges_seen': start+len(rows), 'anchors_with_join_so_far': int(anchor_has_join.sum())}), flush=True)
    result = {'source_usable_subjects': len(heads), 'bounded_anchor_subjects': int(good_anchors.sum()),
        'subjects_with_bounded_nonreturn_two_hop_path': int(anchor_has_join.sum()),
        'subjects_with_both_relations_outside_instance_or_subclass': int(anchor_has_specific_join.sum()),
        'eligible_first_edges': eligible_first_edges, 'second_edge_visits': complete_second_edges,
        'distinct_relation_pair_signatures': len(route_counts),
        'top_relation_pairs_by_first_edge_count': [{'first_relation': 'P'+str(k[0]), 'second_relation': 'P'+str(k[1]), 'first_edges': v} for k,v in route_counts.most_common(40)],
        'policy_sha256': base.file_sha(STATE / 'policy.json'), 'generated_questions': 0,
        'accepted_training_rows': 0, 'model_requests': 0,
        'entity_and_path_counts_are_not_model_capability_measurements': True}
    np.save(STATE / 'anchor-has-bounded-join.npy', anchor_has_join)
    np.save(STATE / 'anchor-has-specific-join.npy', anchor_has_specific_join)
    base.save(STATE / 'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
