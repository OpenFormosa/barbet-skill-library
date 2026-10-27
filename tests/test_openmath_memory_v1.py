import unittest

from barbet_inference.answer_selection import input_identity
from barbet_inference.sft import Rejected
from scripts.diagnose_openmath_memory_v1 import adapt, KNOWN_BAD


class OpenMathMemoryTests(unittest.TestCase):
    def setUp(self):
        self.source = {'repo': 'nvidia/OpenMathInstruct-2', 'revision': 'pinned',
                       'path': 'data/train-00000-of-00032.parquet', 'sha256': 'fixed'}
        self.raw = {'problem': 'A box has 7 red beads and 8 blue beads. How many beads are there?',
                    'generated_solution': 'There are 15 beads.', 'expected_answer': 'TARGET_SECRET_15',
                    'problem_source': 'augmented_gsm8k'}

    def test_expected_answer_is_not_memory_or_conditioning(self):
        row = adapt(self.raw, self.source, 0, ('Use the supplied premises.', []))
        text = ' '.join(m['content'] for m in row['messages'][:-1])
        self.assertNotIn('TARGET_SECRET_15', text)
        self.assertEqual(row['memory']['documents'][0]['text'], self.raw['problem'])
        self.assertFalse(row['source_refs']['expected_answer_in_prompt'])
        self.assertIn('majority_vote_not_independent', row['verification_tier'])

    def test_answer_versions_keep_identical_complete_condition(self):
        a = adapt(self.raw, self.source, 0, ('Use the supplied premises.', []))
        b = adapt({**self.raw, 'generated_solution': '7 + 8 = 15 beads.'}, self.source, 2,
                  ('Use the supplied premises.', []))
        self.assertEqual(input_identity(a['messages'], []), input_identity(b['messages'], []))
        self.assertEqual(a['memory_placement'], b['memory_placement'])
        self.assertNotEqual(a['record_id'], b['record_id'])

    def test_confirmed_wrong_majority_cannot_reenter_by_changing_answer(self):
        for problem in KNOWN_BAD:
            with self.assertRaisesRegex(Rejected, 'independently_confirmed'):
                adapt({**self.raw, 'problem': problem, 'generated_solution': 'A changed answer'},
                      self.source, 0, ('Use the supplied premises.', []))


if __name__ == '__main__': unittest.main()
