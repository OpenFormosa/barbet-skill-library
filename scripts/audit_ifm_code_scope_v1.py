"""Scan all saved coding winners for bounded static scope exclusions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from barbet_inference import code_source_scope as scope
from scripts import build_memory_sft_v2 as base
from scripts.release_ifm_code_v26_v1 import STATE as SELECTED

STATE = SELECTED / 'source-scope-v1'


def independent_cases(rows):
    word_id = '1e002ad4823e3b09292c61fd2166556c3ed0c58778f093e2cdcfba27c0c810a7'
    graph_id = '332bd067423ca24637ecf7602cfd47201e9d2a64465aabb58ed75938cab9900b'
    print_id = 'd7f4c84b65f9c032550e2b499856e4b6f674997bee4944f7c7cddfd907fcb667'
    # Independently authored arithmetic from the literal two-vertex example.
    # For t >= 19 arrival >= 20, so this finite range is exhaustive.
    possible = [(t, t + (100 + t) // (t + 1)) for t in range(19)]
    optimum = min(a for _, a in possible)
    if optimum != 19 or (0 + 100 + 1 - 1) // 1 != 100:
        raise AssertionError('independent edge counterexample changed')
    graph = rows[graph_id]; word = rows[word_id]; illustrative = rows[print_id]
    if 'nxt = (cur + b + a - 1) // a' not in graph['messages'][-1]['content']:
        raise ValueError('selected graph expression differs')
    if not scope.signals(word['messages'][-1]['content'], has_recorded_observation=False):
        raise ValueError('word-count duplicate not located')
    if scope.signals(illustrative['messages'][-1]['content'], has_recorded_observation=False):
        raise ValueError('illustrative print was misclassified as actual execution')
    return {'cases': [
        {'record_id': graph_id, 'query_hash': graph['query_hash'],
         'independent_result': {'a': 1, 'b': 100, 'minimum_correct_arrival': 19,
                                'departures_attaining_minimum': [t for t, a in possible if a == 19],
                                'selected_expression_at_departure_zero': 100},
         'status': 'incorrect transition; first provided example contradicted',
         'scope': 'One literal example and selected expression, not execution of the source program.'},
        {'record_id': word_id, 'query_hash': word['query_hash'],
         'status': 'expected example contains duplicate literal s key; intended regex yields s twice while dictionary retains value one',
         'scope': 'Static inspection of the supplied example, not a claim of executing its test harness.'},
        {'record_id': print_id, 'query_hash': illustrative['query_hash'],
         'status': 'reviewer execution-fabrication allegation not independently established',
         'scope': 'Running the script prints describes expected output; it does not explicitly state the source model actually ran it. First non-accept hold is nevertheless preserved.'}],
        'all_first_dispositions_unchanged': True, 'model_requests': 0, 'source_programs_executed': False}


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'summary.json').exists():
        raise ValueError('coding scope already inspected')
    policy = {'driver_sha256': base.file_sha(Path(__file__)),
        'static_checker_sha256': base.file_sha(Path(scope.__file__)),
        'selection_summary_sha256': base.file_sha(SELECTED / 'dedup-summary.json'),
        'ppl_score_binding_sha256': base.file_sha(SELECTED / 'ppl-score-binding.json'),
        'first_winner_audit_sha256': base.file_sha(SELECTED / 'semantic-audit-v1/summary.json'),
        'scope': 'Conservative whole-question hold for duplicate literal dict keys in explicit Python fences or a narrow past-execution claim with no recorded observation. Intentional duplicate-key demonstrations may also be held; findings are not corpus error-rate estimates.',
        'outside_scope': 'No global requirement for an English UNVERIFIED marker. Illustrative tests, expected outputs and Running the script prints are not treated as observed execution. No all-code semantic verification.',
        'no_answer_rewrite': True, 'no_second_choice_fallback': True, 'model_requests': 0,
        'source_programs_executed': False}
    path = STATE / 'policy.json'
    if path.exists() and json.loads(path.read_text()) != policy:
        raise ValueError('coding static-scope binding changed')
    if not path.exists(): base.save(path, policy)
    sample = {r['record_id']: r for r in (json.loads(line) for line in
              (SELECTED / 'normalized/pilot-selected-winners.jsonl').open())}
    base.save(STATE / 'independent-case-diagnosis.json', independent_cases(sample))
    counts = Counter(); holds = []; shards = []; seen = set()
    for path in sorted((SELECTED / 'selected').glob('*.jsonl')):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line); counts['selected_rows_seen'] += 1
                if row['query_hash'] in seen: raise ValueError('multiple selected views of one coding question')
                seen.add(row['query_hash']); findings = []; observed = False
                for turn, message in enumerate(row['messages']):
                    if message['role'] == 'tool': observed = True
                    if message['role'] != 'assistant': continue
                    findings.extend(dict(f, turn=turn) for f in scope.signals(
                        message.get('content') or '', has_recorded_observation=observed))
                if findings:
                    for reason in {f['reason'] for f in findings}: counts['held:' + reason] += 1
                    holds.append({'record_id': row['record_id'], 'query_hash': row['query_hash'],
                        'reason': 'coding_static_source_scope_hold', 'findings': findings,
                        'messages_sha256': base.sha(json.dumps(row['messages'], ensure_ascii=False, sort_keys=True))})
        shards.append({'path': str(path), 'sha256': digest.hexdigest()})
    if counts['selected_rows_seen'] != json.loads((SELECTED / 'dedup-summary.json').read_text())['selected']:
        raise ValueError('coding source scope coverage incomplete')
    base.save(STATE / 'query-holds.json', {'holds': holds, 'query_hashes': sorted(r['query_hash'] for r in holds),
        'no_second_choice_fallback': True, 'previous_first_dispositions_unchanged': True})
    result = {'counts': dict(counts), 'held_queries': len(holds), 'files': shards,
        'all_selected_source_questions_checked': True, 'policy_sha256': base.file_sha(STATE / 'policy.json'),
        'query_holds_sha256': base.file_sha(STATE / 'query-holds.json'),
        'independent_case_diagnosis_sha256': base.file_sha(STATE / 'independent-case-diagnosis.json'),
        'all_retained_answers_and_ppl_scores_unchanged': True, 'accepted_training_rows': 0,
        'model_requests': 0, 'all_records_semantically_verified': False}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__': main()
