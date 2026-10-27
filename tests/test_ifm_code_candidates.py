import json
import unittest
from barbet_inference.answer_selection import input_identity
from barbet_inference.sft import Rejected
from scripts.normalize_ifm_code_memory_v1 import adapt_code

SOURCE = {'repo': 'IFM/TxT360-3efforts', 'revision': 'pinned', 'path': 'code/high-0.parquet', 'sha256': 'pinned'}
SKILLS = {d: ('Use the supplied task and check the result.', []) for d in ['math', 'coding', 'tool_use']}


class IFMCodeCandidateTests(unittest.TestCase):
    def test_both_complete_answer_variants_survive_without_scores(self):
        rows = []
        for i, answer in enumerate(['```python\ndef twice(x):\n    return 2 * x\n```',
                                    '```python\ndef twice(x):\n    return x + x\n```']):
            raw = {'messages': json.dumps([{'role': 'user', 'content': 'Write Python function twice(x) for integer x.'},
                    {'role': 'assistant', 'content': answer, 'think': 'PRIVATE REASONING FIELD'}])}
            source = {**SOURCE, 'path': f'code/{"high" if i == 0 else "low"}-0.parquet'}
            rows.append(adapt_code(raw, source, i, SKILLS, {'holds': []}))
        self.assertEqual(input_identity(rows[0]['messages'], []), input_identity(rows[1]['messages'], []))
        self.assertNotEqual(rows[0]['record_id'], rows[1]['record_id'])
        for row in rows:
            self.assertNotIn('PRIVATE REASONING FIELD', json.dumps(row))
            self.assertNotIn('answer_selection', row['source_refs'])
            self.assertFalse(row['locally_executed_source_program'])

    def test_syntax_only_does_not_execute_source_and_invalid_blocks_are_held(self):
        raw = {'messages': json.dumps([{'role': 'user', 'content': 'Show Python.'},
               {'role': 'assistant', 'content': '```python\nraise RuntimeError("must never execute")\n```'}])}
        row = adapt_code(raw, SOURCE, 0, SKILLS, {'holds': []})
        self.assertEqual(row['source_refs']['python_static_blocks'], 1)
        for answer in ['def twice(x): return 2*x', '```python\ndef broken(:\n```']:
            raw = {'messages': json.dumps([{'role': 'user', 'content': 'Show Python.'},
                   {'role': 'assistant', 'content': answer}])}
            with self.assertRaises(Rejected):
                adapt_code(raw, SOURCE, 0, SKILLS, {'holds': []})


if __name__ == '__main__':
    unittest.main()
