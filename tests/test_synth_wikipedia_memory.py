import unittest
from scripts.synth_wikipedia_memory_v1 import render_article, wikipedia_title
from barbet_inference.sft import Rejected


class WikipediaEvidenceTests(unittest.TestCase):
    def test_math_and_subscript_are_not_silently_removed(self):
        blocks, count = render_article('<p>Not 20: Dih<sub>10</sub> has '
            '<math alttext="x^2 + 1"><semantics><mi>wrong duplicate</mi></semantics></math>.</p>'
            '<table><tr><td>A</td><td>21</td></tr></table>')
        text = '\n'.join(blocks)
        self.assertIn('Not 20', text); self.assertIn('Dih_(10)', text)
        self.assertIn('x^2 + 1', text); self.assertNotIn('wrong duplicate', text)
        self.assertIn('| A | 21', text); self.assertEqual(count, 1)

    def test_missing_math_representation_is_held(self):
        with self.assertRaises(Rejected): render_article('<p><math><mi>x</mi></math></p>')

    def test_markup_cannot_execute_or_import_scripts(self):
        blocks, _ = render_article('<p>Before</p><script>open("/tmp/no", "w")</script><p>After</p>')
        self.assertEqual(blocks, ['Before', 'After'])

    def test_source_article_allowlist(self):
        self.assertEqual(wikipedia_title('https://en.wikipedia.org/wiki/Sea_of_Japan#Climate'), 'Sea of Japan')
        for value in ['https://en.wikipedia.org.evil.test/wiki/Title', 'file:///etc/passwd',
                      'https://en.wikipedia.org/wiki/User:Example', 'https://en.wikipedia.org/wiki/A?oldid=1']:
            with self.assertRaises(Rejected): wikipedia_title(value)


if __name__ == '__main__': unittest.main()
