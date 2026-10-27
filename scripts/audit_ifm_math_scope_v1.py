"""Read every selected math question for bounded arithmetic and known scope holds.

No model request, answer rewrite, second-choice selection or source execution.
The source's math label does not make a crossword a self-contained calculation.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from scripts import build_memory_sft_v2 as base
from scripts.normalize_ifm_math_memory_v1 import STATE as NORMALIZED
from barbet_inference import arithmetic_verifier as arithmetic

STATE = NORMALIZED / 'selected-v1/source-scope-v2'
SELECTED = STATE.parent
CROSSWORD = re.compile(r'\bcrossword\b', re.I)
CONTRADICTORY_QUERY = '759ed800282c8b3eb8a1788e37bb5ae67ee7dbafb3a17caf7983191232e61bbf'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'summary.json').exists(): raise ValueError('source-scope inspection already complete')
    # As written: price = 5*n - 45 = 7*n + 3, hence n = -24.
    people = Fraction(45 + 3, 5 - 7)
    if people >= 0: raise AssertionError('independent contradiction proof changed')
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'arithmetic_verifier_sha256': base.file_sha(Path(arithmetic.__file__)),
        'selection_summary_sha256': base.file_sha(SELECTED / 'dedup-summary.json'),
        'ppl_score_binding_sha256': base.file_sha(SELECTED / 'ppl-score-binding.json'),
        'independent_inconsistent_question': {'query_hash': CONTRADICTORY_QUERY,
            'source_conditions': ['5*n - price = 45', 'price - 7*n = 3', 'n is a positive number of people'],
            'derived_people': str(people), 'selected_source_answer': '12',
            'status': 'no positive solution under literal source conditions; no speculative correction'},
        'crossword_scope': 'Hold entire IFM-math crossword question cohort: external lexical/wordplay facts are not supplied as memory. This is a scope exclusion, not proof that every crossword answer is wrong.',
        'arithmetic_scope': 'Exact whole-question calculation form with a single integer target. Unsupported forms remain unverified, not automatically wrong.',
        'no_ppl_threshold': True, 'no_second_choice_fallback': True, 'source_programs_executed': False, 'model_requests': 0}
    path = STATE / 'policy.json'
    if path.exists() and json.loads(path.read_text()) != policy: raise ValueError('scope policy changed')
    if not path.exists(): base.save(path, policy)
    counts = Counter(); holds = []; proofs = []; shards = []; seen_queries = set(); found_known = False
    for path in sorted((SELECTED / 'selected').glob('*.jsonl')):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line); counts['selected_rows_seen'] += 1
                if row['query_hash'] in seen_queries: raise ValueError('multiple selected math views of one question')
                seen_queries.add(row['query_hash'])
                docs = row['memory']['documents']
                if len(docs) != 1 or docs[0]['id'] != 'initial-task': raise ValueError('unexpected math task memory')
                question = docs[0]['text']; answer = row['messages'][-1]['content']; reason = None
                if base.sha(base.norm(question)) != row['query_hash']: raise ValueError('question provenance changed')
                if row['query_hash'] == CONTRADICTORY_QUERY:
                    found_known = True
                    if answer.strip() != '12': raise ValueError('known selected source answer changed')
                    reason = 'independently_proven_inconsistent_positive_people_conditions'
                elif CROSSWORD.search(question): reason = 'crossword_external_lexical_knowledge_outside_math_memory_scope'
                result = arithmetic.verify_integer_calculation(question, answer)
                counts['arithmetic:' + result['status']] += 1
                if result['status'] != 'outside_scope':
                    proofs.append({'record_id': row['record_id'], 'query_hash': row['query_hash'],
                                   'question_sha256': base.sha(question), 'answer_sha256': base.sha(answer), **result})
                if result['status'] == 'incorrect': reason = 'exact_arithmetic_selected_answer_incorrect'
                if reason:
                    counts['held:' + reason] += 1
                    holds.append({'record_id': row['record_id'], 'query_hash': row['query_hash'], 'reason': reason,
                                  'question_sha256': base.sha(question), 'answer_sha256': base.sha(answer)})
        shards.append({'path': str(path), 'sha256': digest.hexdigest()})
    if not found_known: raise ValueError('independent counterexample not found in complete selected scope')
    if counts['selected_rows_seen'] != json.loads((SELECTED / 'dedup-summary.json').read_text())['selected']:
        raise ValueError('selected source scope coverage incomplete')
    base.save(STATE / 'query-holds.json', {'holds': holds, 'query_hashes': sorted({r['query_hash'] for r in holds}),
        'no_second_choice_fallback': True, 'previous_first_dispositions_unchanged': True})
    base.save(STATE / 'exact-arithmetic-proofs.json', proofs)
    result = {'counts': dict(counts), 'held_queries': len(holds), 'files': shards,
        'all_selected_source_questions_checked': True, 'policy_sha256': base.file_sha(STATE / 'policy.json'),
        'query_holds_sha256': base.file_sha(STATE / 'query-holds.json'),
        'arithmetic_proofs_sha256': base.file_sha(STATE / 'exact-arithmetic-proofs.json'),
        'all_other_answers_and_ppl_scores_unchanged': True, 'accepted_training_rows': 0,
        'model_requests': 0, 'all_records_semantically_verified': False}
    base.save(STATE / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__': main()
