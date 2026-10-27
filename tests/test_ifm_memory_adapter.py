import json
import unittest
from scripts.ifm_memory_adapter_v1 import adapt, credential_gate
from barbet_inference.sft import Rejected

SOURCE = {'repo': 'IFM/TxT360-3efforts', 'revision': 'pinned', 'path': 'agent/low-0.parquet', 'sha256': 'frozen'}
SKILLS = {d: ('Test skill.', []) for d in ['coding', 'math', 'tool_use']}


def record(argument, user='Find red shoes.', extra=None):
    messages = [{'role': 'system', 'content': '', 'tools': [{'name': 'search',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'api_key': {'type': 'string'}}, 'required': ['query']}}]},
        {'role': 'user', 'content': user}, {'role': 'assistant', 'think': 'Not included.',
            'tool_calls': [{'name': 'search', 'arguments': json.dumps(argument)}]}]
    if extra: messages += extra
    return {'messages': json.dumps(messages)}


class IFMAdapterTests(unittest.TestCase):
    def test_fabricated_credentials_are_held(self):
        with self.assertRaisesRegex(Rejected, 'credential'):
            adapt(record({'query': 'red shoes', 'api_key': 'your_api_key_here'}), SOURCE, 0, SKILLS)

    def test_valid_handoff_does_not_invent_observation(self):
        row = adapt(record({'query': 'red shoes'}), SOURCE, 0, SKILLS)
        self.assertEqual(row['tool_observations'], 0)
        self.assertTrue(row['messages'][-1]['tool_calls'])
        self.assertNotIn('think', json.dumps(row['messages']))

    def test_undeclared_argument_is_not_tolerated_by_schema_default(self):
        with self.assertRaisesRegex(Rejected, 'undeclared'):
            adapt(record({'query': 'red shoes', 'fake_option': True}), SOURCE, 0, SKILLS)

    def test_observations_remain_chronological_and_unsupervised(self):
        raw = record({'query': 'red shoes'}, extra=[{'role': 'tool', 'name': 'search', 'content': 'Found one item.'},
                     {'role': 'assistant', 'content': 'One item was found.'}])
        row = adapt(raw, SOURCE, 0, SKILLS)
        self.assertEqual(row['tool_observations'], 1)
        self.assertFalse(next(m for m in row['messages'] if m['role'] == 'tool')['loss'])

    def test_author_is_not_an_authentication_parameter(self):
        credential_gate([{'role': 'assistant', 'tool_calls': [
            {'function': {'arguments': {'author': 'A derived author name'}}}]}], [])

    def test_gold_reference_in_input_cannot_become_memory(self):
        with self.assertRaisesRegex(Rejected, 'reference_solution'):
            adapt(record({'query': 'red shoes'}, user='<solution>correct response</solution>'), SOURCE, 0, SKILLS)


if __name__ == '__main__': unittest.main()
