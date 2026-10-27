import json
import unittest

from barbet_inference.dataset_quality import terminal_completion_hold


class TerminalCompletionQualityTests(unittest.TestCase):
    def row(self, observations, complete=True):
        return {'upstream_source': 'nvidia/Nemotron-Terminal-Corpus', 'messages': [
            {'role': 'user', 'content': 'Produce the requested files and validate the JSON.'},
            *[{'role': 'tool', 'name': 'run_terminal', 'content': text} for text in observations],
            {'role': 'assistant', 'loss': True, 'content': json.dumps({'task_complete': complete})}]}

    def test_actual_failed_validation_followed_by_a_completion_claim_is_held(self):
        row = self.row(['  File "<string>", line 1\nSyntaxError: invalid syntax\nAll files present and valid.'])
        self.assertEqual(terminal_completion_hold(row), 'unresolved_terminal_error_before_completion_claim')

    def test_later_real_observation_supersedes_old_error(self):
        self.assertIsNone(terminal_completion_hold(self.row([
            'SyntaxError: invalid syntax', 'JSON validation passed; tests: 8 passed'])))

    def test_recorded_diagnostic_failure_is_allowed_without_a_completion_claim(self):
        self.assertIsNone(terminal_completion_hold(self.row(['AssertionError: expected 12 columns'], False)))

    def test_exception_word_in_prose_and_unrelated_sources_are_not_rejected(self):
        self.assertIsNone(terminal_completion_hold(self.row(['The documentation describes SyntaxError: invalid syntax.'])))
        row = self.row(['SyntaxError: invalid syntax']); row['upstream_source'] = 'unrelated'
        self.assertIsNone(terminal_completion_hold(row))


if __name__ == '__main__':
    unittest.main()
