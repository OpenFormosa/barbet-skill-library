import json
import unittest
from barbet_inference.call_text import parse_keyword_calls, UnsupportedCallText
from barbet_inference.sft import Rejected
from scripts.dolci_tool_projection_v1 import project_tool_messages


class DolciToolProjectionTests(unittest.TestCase):
    def tool(self, name):
        return {'name': name, 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'},
                'enabled': {'type': 'boolean'}, 'count': {'type': 'integer'}}, 'required': ['text', 'enabled', 'count']}}

    def messages(self):
        return [{'role': 'system', 'content': 'Use the supplied functions.', 'functions': json.dumps([self.tool('lookup')])},
                {'role': 'user', 'content': 'Look up A with enabled false and count -2.'},
                {'role': 'assistant', 'content': '', 'function_calls': "lookup(text=' A ', enabled=False, count=-2)"},
                {'role': 'environment', 'content': '{"value": 7, "status": "found"}'},
                {'role': 'assistant', 'content': 'The value is 7.'}]

    def test_literal_values_and_observation_loss_preserved(self):
        result = project_tool_messages(self.messages(), {'holds': []})
        messages = result['messages']; arguments = messages[2]['tool_calls'][0]['function']['arguments']
        self.assertEqual(arguments, {'text': ' A ', 'enabled': False, 'count': -2})
        self.assertEqual(messages[3]['content'], '{"value": 7, "status": "found"}')
        self.assertFalse(messages[3]['loss']); self.assertTrue(messages[2]['loss'])
        self.assertEqual(messages[3]['tool_call_id'], messages[2]['tool_calls'][0]['id'])

    def test_ambiguous_parallel_observation_is_held(self):
        rows = self.messages(); rows[2]['function_calls'] = "[lookup(text='A', enabled=False, count=1), lookup(text='B', enabled=True, count=2)]"
        with self.assertRaisesRegex(Rejected, 'ambiguous_tool_observation'):
            project_tool_messages(rows, {'holds': []})

    def test_no_registry_is_inferred(self):
        rows = self.messages(); rows[0]['functions'] = '[]'
        with self.assertRaisesRegex(Rejected, 'no_explicit_source_registry'):
            project_tool_messages(rows, {'holds': []})

    def test_calls_never_execute_expressions_or_collapse_duplicate_keys(self):
        for text in ["tool(x=__import__('os').getcwd())", "tool(x=a)", "tool(x={'a':1,'a':2})", 'tool(**{})',
                     'import os', 'tool(x=1e999)', 'tool(x=(1,2))', 'tool(1)', 'tool(x=True + 1)']:
            with self.assertRaises(UnsupportedCallText): parse_keyword_calls(text)
        self.assertEqual(parse_keyword_calls("api.lookup(text='A')\napi.lookup(text='B')")[1]['function']['name'], 'api.lookup')


if __name__ == '__main__': unittest.main()
