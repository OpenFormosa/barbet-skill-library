import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/test_time_v1/score.py'
spec = importlib.util.spec_from_file_location('study_score', PATH)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class ScoringTests(unittest.TestCase):
    def test_thought_number_cannot_be_used_as_final_answer(self):
        self.assertIsNone(score.number('We have 4 groups, maybe 12?'))
        self.assertEqual(score.number('4 groups give 12.\n#### 12'), 12)
        self.assertEqual(score.number('1/2'), score.number('0.5000'))

    def test_typed_calls_preserve_boolean_and_literals(self):
        self.assertFalse(score.typed_equal({'x': True}, {'x': 1}))
        self.assertFalse(score.typed_equal({'x': '案 Ａ'}, {'x': '案 A'}))
        self.assertTrue(score.typed_equal({'x': 2}, {'x': 2.0}))

    def test_qa_normalization_does_not_erase_negation(self):
        self.assertNotEqual(score.normalize_qa('不是四十層'), score.normalize_qa('四十層'))
        self.assertEqual(score.normalize_qa('「四十 層」'), score.normalize_qa('四十層'))


if __name__ == '__main__':
    unittest.main()
