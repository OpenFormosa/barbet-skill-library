import unittest
from scripts.inventory_dolci_context_v1 import json_field_present, project
from scripts.inventory_dolci_context_v2 import field_present, non_json_call


class DolciSourceInventoryTests(unittest.TestCase):
    def test_empty_function_json_is_not_a_declared_tool(self):
        for value in [None, '', '[]', '{}', 'null']: self.assertFalse(json_field_present(value))
        self.assertTrue(json_field_present('[{"name":"given_function"}]'))
        with self.assertRaises(ValueError): json_field_present('not JSON')

    def test_unresolved_last_tool_turn_is_not_final_answer(self):
        row = {'source_dataset': 'explicit source', 'messages': [{'role': 'user', 'content': 'task'}, {'role': 'tool', 'content': 'result'}]}
        with self.assertRaisesRegex(ValueError, 'final assistant'): project(row)

    def test_source_call_text_is_counted_without_execution(self):
        text = "look_up(code='X1')"
        self.assertTrue(field_present(text))
        self.assertTrue(non_json_call(text))
        self.assertFalse(non_json_call('[{"name":"look_up"}]'))
        for empty in [None, '  ', '[]', '{}', 'null']: self.assertFalse(field_present(empty))


if __name__ == '__main__': unittest.main()
