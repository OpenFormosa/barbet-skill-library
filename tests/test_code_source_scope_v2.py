from unittest.mock import patch
import unittest
from tests import test_code_source_scope as original
from barbet_inference.code_source_scope_v2 import signals


class IndentationTests(unittest.TestCase):
    def test_markdown_indented_fence_is_not_syntax_error(self):
        text = 'Example:\n    ```python\n    base = configs.pop(base_index)\n    for cfg in configs:\n        merged.update(cfg)\n    ```'
        self.assertEqual(signals(text, has_recorded_observation=False), [])

    def test_indented_duplicate_is_still_found(self):
        text = '    ```python\n    a = {"s": 1, "s": 2}\n    ```'
        self.assertEqual(signals(text, has_recorded_observation=False)[0]['reason'], 'duplicate_literal_dictionary_key_scope_hold')


def load_tests(loader, tests, pattern):
    class OriginalChecksWithV2(original.CodeSourceScopeTests):
        def setUp(self):
            self.patcher = patch.object(original, 'signals', signals)
            self.patcher.start(); self.addCleanup(self.patcher.stop)
    tests.addTests(loader.loadTestsFromTestCase(OriginalChecksWithV2))
    return tests


if __name__ == '__main__': unittest.main()
