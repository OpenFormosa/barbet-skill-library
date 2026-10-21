"""Fictional routing tasks; no source QA, target answers or model scores."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'routing'))
from extension import select, select_factual_property


def causal(text, tools=None, previous=None):
    return {'messages': (previous or []) + [{'role': 'user', 'content': text}],
            'tools': tools or []}


def skill(request):
    chosen = select(request)
    return chosen['skill'] if chosen else None


class FactualRoutingTests(unittest.TestCase):
    def test_specific_property_precedes_concept_wording(self):
        for text in [
            '什麼是白港塔的建築高度？', '何謂白港塔的高度？',
            '請說明白港塔的樓層數。', '說明北岬鐵道的軌距。',
            '第47號燈塔的高度是多少？', '北岬鐵道的軌距是多少？',
            '霧島峰的海拔是幾公尺？', '白港塔有幾層？',
            '請問白港塔多高？', '告訴我霧島峰的海拔高度。',
            'How tall is White Harbor Tower?',
            'What is the height of White Harbor Tower?',
            'What was the opening year of North Cape Railway?',
            'How many floors does White Harbor Tower have?',
        ]:
            with self.subTest(text=text):
                self.assertEqual(skill(causal(text)), 'factual-question-resolution')
                self.assertEqual(select_factual_property(causal(text))['rule'],
                                 'explicit_entity_property_question')

    def test_definition_and_mechanism_still_use_concept(self):
        for text in ['什麼是海拔高度？', '何謂軌距？', '解釋高度的測量原理。',
                     'Explain how a tower resists wind.', 'What is a median?']:
            with self.subTest(text=text):
                self.assertIsNone(select_factual_property(causal(text)))
        for text in ['什麼是海拔高度？', '何謂軌距？', '解釋高度的測量原理。']:
            self.assertEqual(skill(causal(text)), 'concept-explanation')

    def test_specific_operation_priorities(self):
        for text, expected in [
            ('翻譯：「什麼是白港塔的高度？」', 'translation-alignment'),
            ('改寫：「白港塔的高度是多少？」', 'text-transformation'),
            ('Write a function to compute the height of a tree.', 'code-task-framing'),
            ('計算 17 + 29。', 'numeric-expression-evaluation'),
            ('Compare the height of two towers.', 'comparison-by-dimensions'),
        ]:
            with self.subTest(text=text):
                self.assertIsNone(select_factual_property(causal(text)))
                self.assertEqual(skill(causal(text)), expected)

    def test_hook_abstains_on_source_live_state_and_multiple_tasks(self):
        for text in [
            '參考資料：白港塔有 17 層。\n問題：白港塔有幾層？',
            '根據文件，白港塔有幾層？', '「白港塔有幾層？」',
            '今天白港塔有幾層？', 'How tall is the tower now?',
            'How tall is the tower and why?', '白港塔多高；解釋測量原理。',
            '摘要：白港塔的高度是多少？',
        ]:
            with self.subTest(text=text):
                self.assertIsNone(select_factual_property(causal(text)))

    def test_number_in_entity_name_is_not_a_calculation(self):
        for entity in ['第47號燈塔', '第82號燈塔', '白港塔']:
            self.assertEqual(skill(causal(f'什麼是{entity}的高度？')),
                             'factual-question-resolution')

    def test_prior_visible_source_is_left_to_source_aware_router(self):
        for previous in [
            [{'role': 'user', 'content': '參考資料：白港塔的設計簡介。'}],
            [{'role': 'system', 'content': 'Answer only according to the document.'}],
            [{'role': 'tool', 'content': 'Source text: a tower profile.'}],
        ]:
            with self.subTest(previous=previous):
                self.assertIsNone(select_factual_property(causal('白港塔有幾層？', previous=previous)))

    def test_premise_value_does_not_choose_skill(self):
        routes = [select(causal(f'白港塔的高度是{value}公尺嗎？'))
                  for value in ['17', '29', '0.8', '-3']]
        self.assertTrue(all(route == routes[0] for route in routes))
        self.assertEqual(routes[0]['skill'], 'factual-question-resolution')

    def test_tools_do_not_imply_factual_lookup_was_executed(self):
        registry = [{'type': 'function', 'function': {'name': 'opaque', 'parameters': {}}}]
        request = causal('什麼是白港塔的高度？', tools=registry)
        self.assertIsNone(select_factual_property(request))
        self.assertEqual(skill(request), 'tool-capability-triage')
        self.assertNotIn('tool_call', select(request))

    def test_only_visible_causal_inputs(self):
        for function in [select, select_factual_property]:
            for key in ['answer', 'target', 'gold', 'source', 'split', 'score', 'future_messages']:
                with self.subTest(function=function.__name__, key=key), self.assertRaises(ValueError):
                    function(dict(causal('白港塔有幾層？'), **{key: 'unused'}))
            self.assertIsNone(function({'messages': [{'role': 'tool', 'content': 'result'}],
                                        'tools': []}))
            for bad in [None, {'messages': [], 'tools': {}},
                        {'messages': [{'role': 'gold', 'content': 'x'}], 'tools': []}]:
                with self.assertRaises(ValueError):
                    function(bad)

    def test_deterministic_nonmutating_and_no_old_prefill(self):
        request = causal('白港塔有幾層？')
        original = copy.deepcopy(request)
        self.assertEqual(select_factual_property(request), select_factual_property(original))
        self.assertEqual(request, original)
        previous = [{'role': 'system', 'content': '[BARBET_SKILL_CONTEXT_V1] existing'}]
        self.assertIsNone(select_factual_property(causal('白港塔有幾層？', previous=previous)))


if __name__ == '__main__':
    unittest.main()
