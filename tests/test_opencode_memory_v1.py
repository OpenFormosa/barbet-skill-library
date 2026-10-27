import json
import unittest

from barbet_inference.answer_selection import input_identity
from barbet_inference.sft import Rejected
from scripts.opencode_memory_adapter_v1 import adapt, source_quality, CRITERIA


class OpenCodeTests(unittest.TestCase):
    def setUp(self):
        self.raw = {'id': 'public-id', 'input': 'Write a function that returns its integer argument plus one.',
            'output': 'def increment(x):\n    return x + 1', 'domain': 'generic', 'generation_algorithm': 'self-instruct',
            'unit_tests': json.dumps(['assert POSTHOC_TEST_SECRET']), 'tests_execution_status': '["pass"]',
            'average_test_score': 1.0, 'llm_judgement': json.dumps({k: {'score': 5, 'justification': 'JUDGE_SECRET'} for k in CRITERIA})}
        self.source = {'repo': 'nvidia/OpenCodeInstruct', 'revision': 'fixed', 'path': 'data/first.parquet', 'sha256': 'hash'}

    def test_source_test_failure_wins_over_perfect_source_judge(self):
        with self.assertRaisesRegex(Rejected, 'source_test_failure'):
            source_quality({**self.raw, 'tests_execution_status': '["fail"]'})

    def test_posthoc_tests_and_judge_never_enter_prompt(self):
        row = adapt(self.raw, self.source, 0, ('Follow the supplied task.', []))
        prompt = json.dumps(row['messages'][:-1])
        self.assertNotIn('POSTHOC_TEST_SECRET', prompt); self.assertNotIn('JUDGE_SECRET', prompt)
        self.assertFalse(row['source_refs']['verification']['locally_executed'])

    def test_physical_source_shards_do_not_change_answer_condition(self):
        a = adapt(self.raw, self.source, 0, ('Follow the supplied task.', []))
        b = adapt({**self.raw, 'output': 'def increment(x):\n    return 1 + x'},
                  {**self.source, 'path': 'data/second.parquet', 'sha256': 'another-hash'}, 700,
                  ('Follow the supplied task.', []))
        self.assertEqual(input_identity(a['messages'], []), input_identity(b['messages'], []))
        self.assertNotEqual(a['record_id'], b['record_id'])

    def test_missing_or_inconsistent_tests_cannot_claim_verified(self):
        with self.assertRaisesRegex(Rejected, 'definition_status_mismatch'):
            source_quality({**self.raw, 'unit_tests': '[]'})
        with self.assertRaisesRegex(Rejected, 'score_inconsistent'):
            source_quality({**self.raw, 'average_test_score': 0.8})


if __name__ == '__main__': unittest.main()
