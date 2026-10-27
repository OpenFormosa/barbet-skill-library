import json
import unittest

from testtime.contracts import PublicRequest, strict_json, validate_call
from testtime.execution import run_python
from testtime.retrieval import Document, LexicalIndex, pack_documents, terms
from testtime.state import State


class RetrievalTests(unittest.TestCase):
    def test_chinese_query_and_original_negation_survive(self):
        documents = [Document('a', '白塔有二十層。', 'fixture:a'),
                     Document('b', '青塔不是三十層，而是四十層。', 'fixture:b')]
        index = LexicalIndex(documents)
        self.assertTrue(terms('青塔有幾層'))
        ranked = index.search('青塔有幾層')
        self.assertEqual(ranked[0][0].id, 'b')
        text, selected, _ = pack_documents(ranked, len, 300)
        self.assertIn('不是三十層', text)
        self.assertEqual(selected[0]['sha256'], documents[1].sha256)

    def test_over_budget_source_is_explicitly_omitted(self):
        doc = Document('a', '長' * 500, 'fixture')
        text, selected, omitted = pack_documents([(doc, 2.0)], len, 100)
        self.assertEqual((text, selected, omitted), ('', [], ['a']))


class BoundaryTests(unittest.TestCase):
    def test_gold_cannot_enter_public_request(self):
        with self.assertRaises(ValueError):
            PublicRequest.from_dict({'id': '1', 'prompt': '題目', 'output_contract': 'choice', 'gold': 'A'})

    def test_duplicate_json_is_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"x":true,"x":false}')

    def test_boolean_is_not_integer_and_literal_not_repaired(self):
        tools = [{'name': 'operation', 'parameters': {'type': 'object',
            'properties': {'count': {'type': 'integer'}, 'id': {'type': 'string', 'enum': ['案-乙']}},
            'required': ['count', 'id'], 'additionalProperties': False}}]
        self.assertEqual(len(validate_call({'name': 'operation', 'arguments': {'count': True, 'id': 'case-乙'}}, tools)), 2)
        self.assertTrue(validate_call({'name': 'operation2', 'arguments': {}}, tools))


class StateTests(unittest.TestCase):
    def test_changed_source_invalidates_evidence(self):
        doc = Document('doc', '甲不是乙。', 'fixture')
        state = State('回答來源問題')
        e = state.add_evidence(doc, 0, 5, '甲不是乙。')
        self.assertTrue(state.validate_evidence(e.id, doc))
        self.assertFalse(state.validate_evidence(e.id, Document('doc', '甲就是乙。', 'fixture')))
        self.assertIn('not_entailment', e.status)
        with self.assertRaises(ValueError):
            state.add_evidence(doc, 0, 5, '甲就是乙。')

    def test_model_claim_is_not_executor_receipt(self):
        with self.assertRaises(ValueError):
            State('task').record_execution({'passed': True, 'evidence': 'I ran the tests'})


class ExecutorTests(unittest.TestCase):
    def test_correct_arithmetic_has_actual_receipt(self):
        result = run_python('from fractions import Fraction\nanswer = Fraction(1, 3) + Fraction(1, 6)', want_answer=True)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['answer'], '1/2')
        self.assertEqual(result['process_exit_code'], 0)

    def test_wrong_fibonacci_fails_real_test(self):
        result = run_python('def fib(n):\n    return -1', ['assert fib(0) == 0'])
        self.assertFalse(result['passed'])
        self.assertIn('AssertionError', result.get('error', ''))

    def test_public_success_does_not_imply_hidden_success(self):
        code = 'def double(x):\n    return 2'
        public = run_python(code, ['assert double(1) == 2'])
        hidden = run_python(code, ['assert double(3) == 6'])
        self.assertTrue(public['passed'], public)
        self.assertFalse(hidden['passed'])
        self.assertNotEqual(public['checks_sha256'], hidden['checks_sha256'])

    def test_filesystem_and_process_access_rejected(self):
        for code in ('import os\nanswer = os.listdir("/root")',
                     'answer = open("/etc/passwd").read()',
                     'answer = (1).__class__.__mro__'):
            result = run_python(code, want_answer=True)
            self.assertFalse(result['passed'], result)

    def test_infinite_loop_terminates(self):
        result = run_python('while True:\n    pass', timeout=4)
        self.assertFalse(result['passed'])
        self.assertIn(result['status'], ('worker_failed', 'timeout'))


if __name__ == '__main__':
    unittest.main()
