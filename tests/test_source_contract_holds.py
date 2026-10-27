import copy
import json
from pathlib import Path
import unittest
from barbet_inference.source_contract_holds import held_tool_schemas


class ExactSourceContractHolds(unittest.TestCase):
    def test_only_the_known_inconsistent_schema_is_held(self):
        policy = json.loads((Path(__file__).resolve().parents[1] / 'configs/source_tool_contract_holds_v1.json').read_text())
        fn = policy['holds'][0]['function']
        self.assertEqual(len(held_tool_schemas([{'type': 'function', 'function': fn}], policy)), 1)
        changed = copy.deepcopy(fn)
        changed['description'] = 'Query the supplied blockchain address and token balance.'
        self.assertEqual(held_tool_schemas([{'type': 'function', 'function': changed}], policy), [])
        self.assertEqual(fn['description'], policy['holds'][0]['function']['description'])


if __name__ == '__main__': unittest.main()
