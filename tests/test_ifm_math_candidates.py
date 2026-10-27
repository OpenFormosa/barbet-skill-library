import json
import unittest
from barbet_inference.answer_selection import input_identity
from scripts.normalize_ifm_math_memory_v1 import adapt_math, ambiguous_literal_queries, nonaccept_queries


class IFMMathCandidateTests(unittest.TestCase):
    def test_distinct_effort_answers_remain_comparable_and_not_preselected(self):
        source = {'repo': 'IFM/TxT360-3efforts', 'revision': 'fixed', 'path': 'math/high-0.parquet', 'sha256': 'fixed'}
        skills = {d: ('Follow the task.', []) for d in ['math', 'coding', 'tool_use']}
        question = 'Compute 2 + 3.'
        rows = []
        for n, answer in enumerate(['5', 'Adding two to three gives 5.']):
            raw = {'messages': json.dumps([{'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer, 'think': 'This field is never input memory.'}])}
            rows.append(adapt_math(raw, source, n, skills, {'holds': []}))
        self.assertEqual(rows[0]['query_hash'], rows[1]['query_hash'])
        self.assertNotEqual(rows[0]['record_id'], rows[1]['record_id'])
        self.assertEqual(input_identity(rows[0]['messages'], []), input_identity(rows[1]['messages'], []))
        for row in rows:
            self.assertNotIn('This field is never input memory.', json.dumps(row['messages']))
            self.assertNotIn('answer_selection', row['source_refs'])

    def test_literal_aliases_held_but_identical_versions_not(self):
        rows = [{'query_hash': 'same', 'literal_first_query_hash': 'A'},
                {'query_hash': 'same', 'literal_first_query_hash': 'B'},
                {'query_hash': 'okay', 'literal_first_query_hash': 'C'},
                {'query_hash': 'okay', 'literal_first_query_hash': 'C'}]
        self.assertEqual(ambiguous_literal_queries(rows), {'same'})

    def test_all_first_nonaccepts_need_query_coverage(self):
        audit = {'excluded_record_ids': ['revise', 'incomplete']}
        rows = [{'record_id': 'revise', 'query_hash': 'q1'}, {'record_id': 'incomplete', 'query_hash': 'q2'}]
        self.assertEqual(nonaccept_queries(audit, rows), {'q1', 'q2'})
        with self.assertRaisesRegex(ValueError, 'coverage'):
            nonaccept_queries(audit, rows[:1])


if __name__ == '__main__': unittest.main()
