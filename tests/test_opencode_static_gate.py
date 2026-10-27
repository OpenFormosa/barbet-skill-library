from pathlib import Path
import tempfile
import unittest

from barbet_inference.sft import Rejected
from scripts.normalize_opencode_memory_v1 import syntax_gate


class StaticGateTest(unittest.TestCase):
    def test_source_statements_are_never_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'must-not-exist'
            source = "```python\nfrom pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n```"
            self.assertEqual(syntax_gate(source), 1)
            self.assertFalse(marker.exists())

    def test_incomplete_or_invalid_blocks_remain_held(self):
        for source in ['```python\ndef f(:\n    pass\n```', '```python\ndef f():\n    pass',
                       '```javascript\nconst a = 1;\n```']:
            with self.subTest(source=source), self.assertRaises(Rejected): syntax_gate(source)

    def test_every_python_block_is_inspected(self):
        self.assertEqual(syntax_gate('```python\nx = 1\n```\nText\n```py\ny = 2\n```'), 2)
        with self.assertRaises(Rejected):
            syntax_gate('```python\nx = 1\n```\n```python\ndef broken(:\n```')


if __name__ == '__main__': unittest.main()
