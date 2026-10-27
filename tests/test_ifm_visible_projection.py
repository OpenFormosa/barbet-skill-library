import json
import unittest
from scripts.inventory_ifm_context_v1 import project


class IFMProjectionTests(unittest.TestCase):
    def test_reasoning_variants_do_not_change_visible_candidate(self):
        raw = [{'role': 'user', 'content': 'Add 2 and 3.'},
               {'role': 'assistant', 'think': 'not public final content', 'content': '5'}]
        first, _ = project({'messages': json.dumps(raw)})
        raw[-1]['think'] = 'different unpublished reasoning'
        second, _ = project({'messages': json.dumps(raw)})
        self.assertEqual(first, second)
        self.assertNotIn('think', first[-1])

    def test_prior_observation_changes_complete_input(self):
        raw = [{'role': 'user', 'content': 'Look up the count.'},
               {'role': 'assistant', 'tool_calls': [{'name': 'count', 'arguments': '{}'}]},
               {'role': 'tool', 'content': '3'}, {'role': 'assistant', 'content': '3'}]
        first, _ = project({'messages': json.dumps(raw)})
        raw[2]['content'] = '4'
        second, _ = project({'messages': json.dumps(raw)})
        self.assertNotEqual(first[:-1], second[:-1])


if __name__ == '__main__': unittest.main()
