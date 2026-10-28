import unittest
import json
from barbet_inference.training import BarbetSFTCollator
from barbet_inference.readiness_quality import continuation_content_holds
from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.sft import pangolin_render
from barbet_inference.context_agent import ContextAgent
from types import SimpleNamespace


class BarbetTrainingTests(unittest.TestCase):
    def row(self, middle):
        return {'input_ids': [114689, *middle, 114690],
                'labels': [-100, *[-100 for _ in middle[:-1]], middle[-1], 114690]}

    def test_preserves_masks_and_right_pads_without_label_shift(self):
        a, b = self.row([10, 11]), self.row([12])
        batch = BarbetSFTCollator()([a, b])
        self.assertEqual(batch['labels'].tolist(), [a['labels'], [*b['labels'], -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(batch['input_ids'].tolist()[1][-1], 114691)

    def test_literal_pad_and_shifted_or_empty_targets_rejected(self):
        collate = BarbetSFTCollator()
        for row in [self.row([114691]), {'input_ids': [114689, 10, 114690], 'labels': [-100, 114690, -100]},
                    {'input_ids': [114689, 10, 114690], 'labels': [-100, -100, -100]}]:
            with self.assertRaises(ValueError): collate([row])

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError): BarbetSFTCollator(max_length=2)([self.row([10])])

    def test_barbet_completion_keeps_exact_prefix_without_future_observation(self):
        messages = [{'role': 'user', 'content': 'Look up the record.', 'loss': False},
                    {'role': 'assistant', 'content': 'Calling lookup.', 'loss': True},
                    {'role': 'tool', 'content': 'FUTURE_OBSERVATION_42', 'name': 'lookup',
                     'tool_call_id': 'c0', 'loss': False},
                    {'role': 'assistant', 'content': 'The value is 42.', 'loss': True}]
        text, spans = pangolin_render(messages)
        row = {'messages_json': json.dumps(messages), 'tools_json': '[]', 'text': text,
               'target_spans_json': json.dumps(spans)}
        actual = prepare_dataset_request(row, model='v26', mode='barbet_completion', target_turn=1)
        old = prepare_dataset_request(row, model='v26', mode='pangolin_completion', target_turn=1)
        self.assertEqual(actual, old)
        self.assertEqual(actual['prompt'], text[:spans[0][0]])
        self.assertNotIn('FUTURE_OBSERVATION_42', actual['prompt'])
        self.assertFalse(actual['add_special_tokens'])

    def test_barbet_agent_uses_exported_protocol_instead_of_checkpoint_template(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.text = text
                self.add_special_tokens = add_special_tokens
                return [1] * len(text)
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError('Checkpoint native template must not replace training protocol')
        tokenizer = Tokenizer()
        backend = SimpleNamespace(tokenizer=tokenizer, max_model_len=10000)
        agent = ContextAgent(backend, output_tokens=10, prompt_format='barbet_portable_v1')
        messages = [{'role': 'system', 'content': 'Use evidence.'},
                    {'role': 'user', 'content': 'Find the value.'}]
        ids = agent._tokens(messages)
        self.assertEqual(tokenizer.text, pangolin_render(messages)[0] + '<|assistant_channel|>')
        self.assertFalse(tokenizer.add_special_tokens)
        self.assertEqual(len(ids), len(tokenizer.text))

    def test_calendar_proof_is_independent_of_observation(self):
        def check(text):
            return continuation_content_holds('nvidia/Nemotron-SFT-Agentic-v2', 'tool_use',
                [{'role': 'assistant', 'loss': True, 'content': text}], has_tools=True)
        self.assertEqual(check('Monday, January 27, 2025'), [])
        result = check('Sunday, January 27, 2025')
        self.assertEqual(result[0]['actual_weekday'], 'monday')

    def test_old_action_format_only_held_in_its_source_and_interface(self):
        messages = [{'role': 'assistant', 'loss': True, 'content': "[users.password.reset]=>{sm_email|'user@example.com'}"}]
        self.assertTrue(continuation_content_holds('voidful/agent-sft', 'tool_use', messages, has_tools=True))
        self.assertEqual(continuation_content_holds('unrelated', 'coding', messages, has_tools=True), [])
        messages[0]['content'] = 'Explain the string [tool]=>{x|1}.'
        self.assertEqual(continuation_content_holds('voidful/agent-sft', 'tool_use', messages, has_tools=True), [])


if __name__ == '__main__': unittest.main()
