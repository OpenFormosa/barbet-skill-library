import unittest
from scripts.audit_memory_sft_math_v2 import body_for, safe_response


class MathAuditTransportTests(unittest.TestCase):
    def test_budget_and_payload_preserved(self):
        job = {'domain': 'math', 'messages': [{'role': 'user', 'content': '2 + 3?'}, {'role': 'assistant', 'content': '5'}]}
        body = body_for(job)
        self.assertEqual(body['max_tokens'], 8192)
        self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
        self.assertIn('2 + 3?', body['messages'][1]['content'])
        self.assertEqual(body_for({**job, 'domain': 'reading'})['max_tokens'], 900)

    def test_private_reasoning_never_saved(self):
        choice = {'message': {'content': '{"verdict":"ACCEPT"}', 'reasoning': 'private', 'reasoning_content': 'private'}}
        self.assertEqual(safe_response(choice), {'content': '{"verdict":"ACCEPT"}'})


if __name__ == '__main__': unittest.main()
