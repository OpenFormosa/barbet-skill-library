import unittest
from unittest.mock import patch

from barbet_inference.sft import Rejected
from scripts import normalize_natural_instructions_memory_v2 as full


class NIContextScopeTests(unittest.TestCase):
    def test_topic_check_accepts_literal_phrase_but_not_an_unprovided_alias_or_subword(self):
        self.assertTrue(full.topic_is_literal('Where is the O2 Arena?', 'O2 Arena'))
        self.assertFalse(full.topic_is_literal("What was Pierce Brosnan's first outing as 007?", 'James Bond'))
        self.assertFalse(full.topic_is_literal('Which cartoon is this?', 'art'))

    def test_first_negative_query_stops_before_any_normalization_or_new_answer(self):
        task = {'Definition': ['D']}; instance = {'input': 'I', 'output': ['A']}
        qh = full.base.sha(full.base.norm('D\n\nI'))
        with patch.object(full, 'HELD', {qh}, create=True), patch.object(full, 'ORIGINAL_NORMALIZE') as original:
            with self.assertRaisesRegex(Rejected, 'first_source_diagnostic'):
                full.normalize_instance(task, instance, {'path': 'tasks/x.json'}, 0)
            original.assert_not_called()


if __name__ == '__main__': unittest.main()
