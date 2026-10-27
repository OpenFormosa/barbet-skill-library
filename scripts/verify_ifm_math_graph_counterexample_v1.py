"""Check an independently authored finite counterexample, not source code."""
import json
from pathlib import Path
from scripts import build_memory_sft_v2 as base

STATE = base.WORK / 'reports/memory-harness-v2-ifm-v1/math-full-v1/selected-v1'
RECORD = 'd81d24b0cfbb263166cb92f32086e53f6f86fb4c688ef226a76a38986d1d71b8'
QUERY = 'a097c50d7086e65da440d09d6b074e3d559d44b14a6db6a2531c496791d0034b'


def main():
    target = STATE / 'independent-graph-counterexample-v1.json'
    if target.exists():
        raise ValueError('counterexample receipt already exists')
    source = None
    for path in (STATE / 'normalized').glob('*.jsonl'):
        for line in path.open():
            row = json.loads(line)
            if row['record_id'] == RECORD:
                source = row
    if source is None or source['query_hash'] != QUERY:
        raise ValueError('original reviewed question identity differs')
    first = STATE / 'semantic-audit-v1' / ('sample-' + RECORD) / 'disposition.json'
    if json.loads(first.read_text())['verdict'] != 'REJECT':
        raise ValueError('first disposition differs; never replace it')
    # Fifty A vertices connect to the first 99-i of 99 B vertices.
    # This construction is authored here; nothing from the source is executed.
    neighbors = [set() for unused in range(149)]
    for a in range(50):
        for b in range(99-a):
            neighbors[a].add(50+b)
            neighbors[50+b].add(a)
    degrees = [len(n) for n in neighbors]
    triangle_incidence = sum(len(neighbors[a] & neighbors[b])
                             for a in range(149) for b in neighbors[a] if a < b)
    checks = {'149_vertices_less_than_claimed_199': len(neighbors) < 199,
              'every_degree_1_through_99_present': set(range(1, 100)) <= set(degrees),
              'zero_triangles': triangle_incidence == 0,
              'symmetric_simple_undirected_edges': all(a not in neighbors[a] and all(a in neighbors[b] for b in neighbors[a]) for a in range(149))}
    if not all(checks.values()):
        raise ValueError('independent finite counterexample check failed')
    base.save(target, {'record_id': RECORD, 'query_hash': QUERY,
        'first_disposition_sha256': base.file_sha(first), 'checker_sha256': base.file_sha(Path(__file__)),
        'source_question_sha256': base.sha(source['memory']['documents'][0]['text']),
        'vertices': len(neighbors), 'edges': sum(degrees) // 2, 'degree_values': sorted(set(degrees)),
        'construction': 'A_i, 0<=i<50, connects to B_j, 0<=j<99-i; no within-part edges.',
        'checks': checks, 'source_programs_executed': False, 'model_requests': 0,
        'first_disposition_unchanged': True, 'held_question_not_replaced': True,
        'scope': 'Counterexample disproves the source minimum 199. This receipt does not certify all math data or substitute a new training answer.'})
    print(json.dumps({'vertices': len(neighbors), 'edges': sum(degrees)//2, 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
