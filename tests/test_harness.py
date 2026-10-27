import json
import unittest

from testtime.contracts import PublicRequest, validate_call
from testtime.harness import Episode, Policy
from testtime.retrieval import Document


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps({'messages': messages, 'tools': kwargs.get('tools'),
            'thinking': kwargs['enable_thinking']}, ensure_ascii=False)

    def encode(self, text, **kwargs):
        return list(text)


def episode(request, policy, documents=None):
    cards = {k: {'text': 'Check the task.'} for k in ['number', 'python', 'tool_call', 'short_answer', 'choice']}
    return Episode(request, policy, documents or {}, cards, CharacterTokenizer(), 'mini', 7)


def answer(e, text):
    ids, _, _ = e.generate_request()
    e.accept('<think>check</think>' + text, prompt_tokens=len(ids), output_tokens=len(text),
        finish_reason='stop', batch_seconds=.1, batch_size=1, terminal_strings=[])


class HarnessTests(unittest.TestCase):
    def test_identifier_does_not_route_prompt(self):
        a = episode(PublicRequest('gsm8k:1', 'What is 3+4?', 'number'), Policy('bare'))
        b = episode(PublicRequest('tmmluplus:any:55', 'What is 3+4?', 'number'), Policy('bare'))
        self.assertEqual(a.generate_request()[0], b.generate_request()[0])

    def test_program_control_uses_identical_initial_prompt(self):
        req = PublicRequest('made-up', 'What is 1/3 + 1/6?', 'number')
        a = episode(req, Policy('control', program=True))
        b = episode(req, Policy('execute', program=True, execute=True))
        self.assertEqual(a.generate_request()[:2], b.generate_request()[:2])
        text = json.dumps({'code': 'from fractions import Fraction\nanswer=Fraction(1,3)+Fraction(1,6)', 'answer': '0.6'})
        answer(a, text)
        answer(b, text)
        self.assertEqual(a.prediction, '0.6')
        self.assertEqual(b.prediction, '1/2')

    def test_public_pass_does_not_trigger_gold_repair(self):
        req = PublicRequest('made-up', 'Implement double.', 'python', public_tests=['assert double(1)==2'])
        e = episode(req, Policy('repair', execute=True, repair=2))
        answer(e, 'def double(x):\n    return 2')
        self.assertTrue(e.done)
        self.assertEqual(e.result()['calls'], 1)
        self.assertNotIn('double(3)', e.rendered_prompt)

    def test_only_actual_public_feedback_drives_bounded_repair(self):
        req = PublicRequest('made-up', 'Implement double.', 'python', public_tests=['assert double(1)==2'])
        e = episode(req, Policy('repair', execute=True, repair=2, compact=True))
        for _ in range(3):
            answer(e, 'def double(x):\n    return -1')
        self.assertTrue(e.done)
        self.assertEqual(e.result()['calls'], 3)
        self.assertIn('AssertionError', e.rendered_prompt)

    def test_unclosed_thought_is_not_scored_as_answer(self):
        e = episode(PublicRequest('made-up', 'Choose.', 'choice'), Policy('bare'))
        e.generate_request()
        e.accept('The answer might be A', prompt_tokens=20, output_tokens=8,
            finish_reason='length', batch_seconds=.1, batch_size=1, terminal_strings=[])
        self.assertEqual(e.prediction, '')
        self.assertTrue(e.trace[0]['unclosed_thinking'])

    def test_evidence_check_rejects_invented_quote(self):
        doc = Document('d', '青塔不是三十層，而是四十層。', 'fixture')
        e = episode(PublicRequest('q', '青塔有幾層？', 'short_answer', document_ids=['d']),
                    Policy('evidence', retrieval=True, evidence=True), {'d': doc})
        answer(e, '{"document_id":"d","quote":"青塔是三十層"}')
        self.assertIn('evidence_error', e.trace[0])
        self.assertEqual(e.state.evidence, {})
        self.assertFalse(e.done)
        answer(e, '四十層')
        self.assertTrue(e.done)

    def test_malformed_tool_name_cannot_crash_validation(self):
        self.assertTrue(validate_call({'name': [], 'arguments': {}}, []))


if __name__ == '__main__':
    unittest.main()
