import hashlib
import json
from pathlib import Path
import unittest

from barbet_inference.memory import MemoryItem, MemoryStore, RefinementLoop, Verification, compose_memory_prompt
from barbet_inference.sft import (MiniCPMRenderer, Rejected, augment_system, clean_text,
                                inference_messages, normalize_messages, pangolin_render,
                                portable_messages, prepare_tools, tokenize_spans)


TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {
    'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key'],
    'additionalProperties': False}}}]


class ContractTests(unittest.TestCase):
    def normalize(self, messages):
        tools, validators = prepare_tools(TOOLS)
        return normalize_messages(messages, tools, validators)

    def test_future_memory_not_retrieved(self):
        store = MemoryStore([MemoryItem('a', 'alpha version 1', 'source'),
                             MemoryItem('b', 'alpha version 2', 'tool', 3, 'tool_observation')])
        self.assertEqual([m.id for m in store.retrieve('alpha', turn=0)], ['a'])

    def test_future_memory_cannot_be_forced_into_prompt(self):
        with self.assertRaisesRegex(ValueError, 'future'):
            compose_memory_prompt('q', [MemoryItem('x', 'answer', 'future', 2)], turn=1)

    def test_unverified_assistant_memory_rejected(self):
        with self.assertRaises(ValueError):
            MemoryItem('x', 'guess', 'assistant', kind='assistant_guess')

    def test_positions_are_before_answer(self):
        item = MemoryItem('d', 'source fact', 'doc')
        front = compose_memory_prompt('question', [item], placement='front')
        back = compose_memory_prompt('question', [item], placement='back')
        self.assertLess(front.index('source fact'), front.index('QUESTION:'))
        self.assertGreater(back.index('source fact'), back.index('QUESTION:'))

    def test_orphan_observation_rejected(self):
        with self.assertRaisesRegex(Rejected, 'orphan'):
            self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': 'done'}])

    def test_invalid_typed_argument_rejected(self):
        with self.assertRaisesRegex(Rejected, 'typed_argument'):
            self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'assistant',
                'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 42}}}]}])

    def test_call_handoff_without_observation_is_valid(self):
        m = self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'assistant',
            'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 'a'}}}]}])
        self.assertTrue(m[-1]['loss'])
        self.assertEqual(len(inference_messages(m)), 1)

    def test_missing_observation_before_answer_rejected(self):
        with self.assertRaisesRegex(Rejected, 'missing_tool'):
            self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'assistant',
                'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 'a'}}}]},
                {'role': 'assistant', 'content': 'success'}])

    def test_failed_diagnostic_call_remains_supervised(self):
        m = self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'assistant',
            'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 'a'}}}]},
            {'role': 'tool', 'content': 'error: unavailable'},
            {'role': 'assistant', 'content': 'The lookup failed; I cannot verify the value.'}])
        self.assertTrue(m[1]['loss'])
        self.assertFalse(m[2]['loss'])

    def test_duplicate_call_id_rejected(self):
        with self.assertRaisesRegex(Rejected, 'duplicate_call_id'):
            self.normalize([{'role': 'user', 'content': 'q'}, {'role': 'assistant',
                'tool_calls': [{'id': 'x', 'function': {'name': 'lookup', 'arguments': {'key': k}}}
                               for k in ('a', 'b')]}])

    def test_no_sft_or_future_fields_in_request(self):
        m = [{'role': 'system', 'content': 'rule'}, {'role': 'user', 'content': 'q'},
             {'role': 'assistant', 'content': 'PRIVATE_TARGET', 'loss': True}]
        request = inference_messages(m)
        self.assertNotIn('PRIVATE_TARGET', json.dumps(request))
        self.assertNotIn('loss', json.dumps(request))

    def test_private_reasoning_removed(self):
        self.assertEqual(clean_text('<think>private</think>Answer', assistant=True), 'Answer')
        with self.assertRaises(Rejected):
            clean_text('<think>private', assistant=True)

    def test_text_blocks_preserved_multimodal_rejected(self):
        self.assertEqual(clean_text([{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]), 'a\nb')
        with self.assertRaises(Rejected):
            clean_text([{'type': 'image_url', 'image_url': 'x'}])

    def test_json_adapter_preserves_whitespace_and_types(self):
        args = {'s': ' \n</function>&測試\n ', 'x': [True, None, {'a': 2}]}
        m = [{'role': 'system', 'content': 's'}, {'role': 'assistant', 'content': '',
              'tool_calls': [{'function': {'name': 'f', 'arguments': args}}], 'loss': True}]
        p = portable_messages(m, TOOLS)
        self.assertEqual(json.loads(p[-1]['content'])['tool_calls'][0]['function']['arguments'], args)
        self.assertIn('lookup', p[0]['content'])

    def test_refine_needs_actual_bound_feedback(self):
        seen = []
        def generate(history):
            seen.append(history[:])
            return 'wrong' if len(seen) == 1 else 'right'
        def verify(s):
            return Verification(s == 'right', 'equality check', hashlib.sha256(s.encode()).hexdigest(), 'fixture')
        r = RefinementLoop(generate, verify).run([{'role': 'user', 'content': 'q'}])
        self.assertEqual(r['status'], 'verified_within_scope')
        self.assertEqual(len(r['attempts']), 2)
        self.assertNotIn('FEEDBACK', json.dumps(seen[0]))
        self.assertIn('FEEDBACK', json.dumps(seen[1]))

    def test_stale_verification_rejected(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            RefinementLoop(lambda _: 'new', lambda _: Verification(True, 'ok', 'old', 'test')).run([])

    def test_refinement_stalls_instead_of_repeating(self):
        v = lambda s: Verification(False, 'failed', hashlib.sha256(s.encode()).hexdigest(), 'test')
        self.assertEqual(RefinementLoop(lambda _: 'same', v).run([])['status'], 'stalled')

    def test_actual_token_boundaries_and_native_template(self):
        from tokenizers import Tokenizer
        root = Path('/root/barbet-context-sft-data')
        mini = root / 'data/raw_memory_harness_v2/minicpm5-tokenizer'
        if not mini.exists():
            self.skipTest('pinned tokenizer fixture not present')
        m = augment_system(self.normalize([{'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'lookup', 'arguments': {'key': ' \n測試\n '}}}]}]), 'skill')
        for renderer, tok in [(lambda: pangolin_render(m, TOOLS), root/'data/blobs/tokenizer/tokenizer.json'),
                              (lambda: MiniCPMRenderer((mini/'chat_template.jinja').read_text()).render(m, TOOLS), mini/'tokenizer.json')]:
            text, spans = renderer()
            ids, labels = tokenize_spans(Tokenizer.from_file(str(tok)), text, spans)
            self.assertEqual(len(ids), len(labels))
            self.assertGreater(sum(x != -100 for x in labels), 0)
            self.assertEqual(labels[0], -100)
            self.assertTrue(all(b in (-100, a) for a, b in zip(ids, labels)))


if __name__ == '__main__':
    unittest.main()
