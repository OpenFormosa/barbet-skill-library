import unittest

from barbet_inference.code_source_scope import signals


class CodeSourceScopeTests(unittest.TestCase):
    def test_duplicate_literal_keys_and_python_equal_keys(self):
        got = signals("```python\na = {'s': 1, 's': 1}\nb = {True: 1, 1: 2}\n```", has_recorded_observation=False)
        self.assertEqual(len(got), 2)
        self.assertTrue(all(x['reason'] == 'duplicate_literal_dictionary_key_scope_hold' for x in got))

    def test_nested_dicts_are_independent_and_code_is_never_run(self):
        source = "```python\nraise RuntimeError('must never run')\na = {'one': {'s': 1}, 'two': {'s': 2}}\n```"
        self.assertEqual(signals(source, has_recorded_observation=False), [])

    def test_expected_prints_and_dry_run_are_not_actual_execution(self):
        source = "Running the script prints:\n```\nAll tests passed\n```\nIf you run this code, it will pass.\n```python\nprint('I tested the function')\n```"
        self.assertEqual(signals(source, has_recorded_observation=False), [])

    def test_explicit_execution_claim_needs_observation(self):
        for source in ["I ran the script.", "We've tested the function.", "We have run these tests."]:
            self.assertEqual(len(signals(source, has_recorded_observation=False)), 1)
            self.assertEqual(signals(source, has_recorded_observation=True), [])


if __name__ == '__main__':
    unittest.main()
