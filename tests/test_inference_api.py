"""Application-boundary tests, including real execution and an HTTP test server."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from barbet_inference import Document, Harness, Policy, PublicRequest, SkillLibrary, make_request
from barbet_inference.backends import Completion, Generation, HTTPBackend
from barbet_inference.cli import main, read_records, resolve_model_path
from barbet_inference.runtime import validate_policy, validate_request


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['role'] + ': ' + m['content'] for m in messages)

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class ScriptedBackend:
    tokenizer = CharacterTokenizer()
    max_model_len = 49152
    max_batch_size = 8
    terminal_strings = []
    metadata = {'backend': 'test_fixture', 'model': 'scripted'}

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.generated = []

    def generate(self, batch):
        self.generated.extend(batch)
        return [Completion(next(self.outputs), len(g.token_ids), 10, 'stop', .1, len(batch)) for g in batch]


class HarnessAPITests(unittest.TestCase):
    def test_free_text_does_not_inherit_literal_short_answer_instructions(self):
        harness = Harness(ScriptedBackend(['這是一段完整的自由文字回覆。']),
                          policy=Policy(name='text', skill=True, literal_answer=True, thinking=False))
        result = harness.run(make_request('請寫一段產品介紹。', 'text'))
        self.assertEqual(result['prediction'], '這是一段完整的自由文字回覆。')
        self.assertEqual(result['inference']['output_contract'], 'text')
        self.assertNotIn('照抄來源', result['trace'][0]['prompt'])
        self.assertNotIn('最短答案', result['trace'][0]['prompt'])
        self.assertEqual(result['validation']['status'], 'not_checked')
        with self.assertRaisesRegex(ValueError, 'non-public'):
            validate_request({'id':'x','prompt':'x','output_contract':'text','gold':'hidden'})

    def test_execution_failure_repairs_using_actual_feedback(self):
        backend = ScriptedBackend(['def double(x):\n    return x', 'def double(x):\n    return x * 2'])
        harness = Harness(backend, policy=Policy(name='repair', thinking=False, execute=True, repair=2, compact=True))
        events = []
        result = harness.run(PublicRequest('code', 'Implement double(x).', 'python',
                            public_tests=['assert double(3) == 6']), on_call=events.append)
        self.assertEqual(result['calls'], 2)
        self.assertEqual(result['validation']['status'], 'public_tests_passed')
        self.assertEqual(result['validation']['semantic_correctness'], 'unknown')
        self.assertFalse(result['trace'][0]['execution']['passed'])
        self.assertIn('assert double(3) == 6', result['trace'][0]['prompt'])
        second_prompt = backend.tokenizer.decode(backend.generated[1].token_ids)
        self.assertIn('AssertionError', second_prompt)
        self.assertIn('Host state:', second_prompt)
        self.assertEqual([e['call_index'] for e in events], [1, 2])
        self.assertEqual(result['output_tokens'], 20)

    def test_one_call_ceiling_keeps_failed_execution_visible(self):
        backend = ScriptedBackend(['def double(x):\n    return -1'])
        harness = Harness(backend, policy=Policy(name='bounded', thinking=False, execute=True, repair=2), max_calls=1)
        result = harness.run(PublicRequest('code', 'Implement double.', 'python', public_tests=['assert double(3) == 6']))
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['validation']['status'], 'execution_failed')

    def test_no_execution_does_not_claim_a_test_pass(self):
        harness = Harness(ScriptedBackend(['def f(): return 1']), policy=Policy(name='plain', thinking=False))
        with patch('testtime.harness.run_python') as execute:
            result = harness.run(PublicRequest('code', 'Implement f.', 'python', public_tests=['assert f() == 1']))
        execute.assert_not_called()
        self.assertEqual(result['validation'], {'status': 'not_checked', 'passed': None, 'semantic_correctness': 'unknown'})

    def test_document_corpus_is_available_without_benchmark_document_ids(self):
        docs = [Document('white', '白塔有二十層。', 'fixture:white'), Document('blue', '青塔不是三十層，而是四十層。', 'fixture:blue')]
        harness = Harness(ScriptedBackend(['四十層']), documents=docs,
                          policy=Policy(name='retrieval', retrieval=True, thinking=False, top_k=1))
        result = harness.run(make_request('青塔有幾層？'))
        self.assertEqual(result['selected_sources'][0]['id'], 'blue')
        self.assertEqual(result['prediction'], '四十層')
        self.assertIn('不是三十層', result['trace'][0]['prompt'])

    def test_all_returned_batch_calls_logged_before_host_failure(self):
        backend = ScriptedBackend(['A', 'B'])
        harness = Harness(backend, policy=Policy(name='plain', thinking=False))
        events = []
        with patch('testtime.harness.Episode.accept', side_effect=RuntimeError('host failed')):
            with self.assertRaisesRegex(RuntimeError, 'host failed'):
                list(harness.run_many([make_request('q1', 'choice', id='1'), make_request('q2', 'choice', id='2')], on_call=events.append))
        self.assertEqual([e['id'] for e in events], ['1', '2'])

    def test_manual_skill_is_loaded_with_provenance(self):
        harness = Harness(ScriptedBackend([]), skills=['arithmetic-modeling'])
        row = harness.prepare(make_request('2 + 3 = ?', 'number'))
        self.assertIn('Skill: arithmetic-modeling', row['prompt'])
        source = row['inference']['manual_skills'][0]
        self.assertEqual(source['name'], 'arithmetic-modeling')
        self.assertEqual(len(source['source_sha256']), 64)
        self.assertIn('do_not_use_when:', row['prompt'])

    def test_unknown_skill_and_source_fail_before_generation(self):
        backend = ScriptedBackend([])
        with self.assertRaisesRegex(ValueError, 'unknown skill'):
            Harness(backend, skills=['../../secret'])
        harness = Harness(backend)
        with self.assertRaisesRegex(ValueError, 'unknown document'):
            harness.run(PublicRequest('q', 'Question', 'short_answer', document_ids=['missing']))
        self.assertEqual(backend.generated, [])

    def test_context_limit_and_duplicate_ids_fail_before_generation(self):
        backend = ScriptedBackend([])
        backend.max_model_len = 100
        harness = Harness(backend)
        with self.assertRaisesRegex(ValueError, 'context ceiling'):
            harness.run(make_request('Question'))
        with self.assertRaisesRegex(ValueError, 'duplicate request'):
            list(harness.run_many([make_request('a'), make_request('b')]))
        self.assertEqual(backend.generated, [])

    def test_unfinished_thought_is_not_an_answer(self):
        harness = Harness(ScriptedBackend(['Let me work this out: 42']), policy=Policy(name='thinking'))
        result = harness.run(make_request('Question', 'number'))
        self.assertEqual(result['validation']['status'], 'unfinished_thinking')
        self.assertEqual(result['prediction'], '')

    def test_undefined_fraction_is_not_a_valid_numeric_result(self):
        harness = Harness(ScriptedBackend(['#### 1/0']), policy=Policy(name='plain', thinking=False))
        self.assertEqual(harness.run(make_request('Question', 'number'))['validation']['status'], 'format_invalid')

    def test_public_boundary_and_argument_types(self):
        for extra in ('gold', 'answer', 'hidden_tests'):
            with self.assertRaisesRegex(ValueError, 'non-public'):
                validate_request({'id': 'q', 'prompt': 'q', 'output_contract': 'number', extra: '42'})
        with self.assertRaisesRegex(ValueError, 'public_tests'):
            validate_request({'id': 'q', 'prompt': 'q', 'output_contract': 'python', 'public_tests': 'assert True'})
        with self.assertRaisesRegex(ValueError, 'registered tools'):
            make_request('Call tool', 'tool_call')

    def test_unsupported_schema_cannot_silently_bypass_validation(self):
        tools = [{'name': 'f', 'parameters': {'type': 'object', 'properties': {
            'x': {'type': 'integer', 'minimum': 5}}}}]
        with self.assertRaisesRegex(ValueError, 'unsupported tool schema'):
            make_request('Call f', 'tool_call', tools=tools)

    def test_invalid_policy_cannot_expand_calls_or_use_truthy_strings(self):
        for policy in (Policy(name='bad', repair=3), Policy(name='bad', execute='false'),
                       Policy(name='bad', output_tokens=0), Policy(name='bad', contract_overrides={'gold': {}})):
            with self.assertRaises(ValueError):
                validate_policy(policy, 3)
        with self.assertRaisesRegex(ValueError, 'two calls'):
            validate_policy(Policy(name='bad', evidence=True), 1)


@contextmanager
def server(reply, status=200):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append({'path': self.path, 'headers': dict(self.headers),
                             'body': json.loads(self.rfile.read(int(self.headers['Content-Length'])))})
            self.send_response(status)
            if status == 302:
                self.send_header('Location', '/should-not-follow')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
        def log_message(self, *args):
            pass
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}/v1', received
    finally:
        httpd.shutdown(); thread.join(); httpd.server_close()


class HTTPBackendTests(unittest.TestCase):
    def test_native_token_ids_and_real_usage_survive_http(self):
        response = {'choices': [{'text': '#### 7', 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}
        with server(response) as (url, received):
            backend = HTTPBackend('tokenizer-id', base_url=url, served_model='served', api_key='test-secret',
                                  tokenizer=CharacterTokenizer(), eos_ids=[99])
            result = backend.generate([Generation([1, 2, 3], {'max_tokens': 20, 'seed': 4})])[0]
        self.assertEqual(received[0]['path'], '/v1/completions')
        self.assertEqual(received[0]['body']['prompt'], [1, 2, 3])
        self.assertFalse(received[0]['body']['skip_special_tokens'])
        self.assertEqual(result.output_tokens, 4)
        self.assertNotIn('test-secret', json.dumps(backend.metadata))

    def test_missing_usage_is_not_fabricated(self):
        with server({'choices': [{'text': 'A', 'finish_reason': 'stop'}]}) as (url, _):
            backend = HTTPBackend('fake', base_url=url, tokenizer=CharacterTokenizer())
            with self.assertRaisesRegex(RuntimeError, 'actual token usage'):
                backend.generate([Generation([1], {})])

    def test_http_failure_does_not_echo_secrets_or_retry(self):
        with server({'error': 'test-secret'}, 401) as (url, received):
            backend = HTTPBackend('fake', base_url=url, api_key='test-secret', tokenizer=CharacterTokenizer())
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as caught:
                backend.generate([Generation([1], {})])
        self.assertNotIn('test-secret', str(caught.exception))
        self.assertEqual(len(received), 1)

    def test_redirect_is_not_followed(self):
        with server({}, 302) as (url, received):
            backend = HTTPBackend('fake', base_url=url, tokenizer=CharacterTokenizer())
            with self.assertRaisesRegex(RuntimeError, 'HTTP 302'):
                backend.generate([Generation([1], {})])
        self.assertEqual(len(received), 1)


class CLITests(unittest.TestCase):
    def test_offline_cache_does_not_require_repository_readme_files(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{}')
            hub = SimpleNamespace(try_to_load_from_cache=lambda *a, **k: str(config))
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                self.assertEqual(resolve_model_path('org/model', 'pinned', True), directory)

    def test_library_listing_requires_no_model_dependencies(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(['list-skills']), 0)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), len(SkillLibrary().list()))
        self.assertIn('verified-state-handoff', {r['name'] for r in rows})

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.jsonl'
            target.write_text('keep me')
            with patch('barbet_inference.cli.TokenizerBackend') as backend, redirect_stderr(io.StringIO()):
                code = main(['run', '--family', 'mini', '--prompt', 'q', '--contract', 'number', '--output', str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(target.read_text(), 'keep me')
            backend.assert_not_called()

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.jsonl'
            source.write_text('{"id":"a","id":"b"}\n')
            with self.assertRaisesRegex(ValueError, 'duplicate key'):
                read_records(source)

    def test_prepare_does_not_create_generation_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'prepared.jsonl'
            with patch('barbet_inference.cli.TokenizerBackend', return_value=ScriptedBackend([])), \
                 patch('barbet_inference.cli.VLLMBackend') as backend:
                code = main(['prepare', '--family', 'mini', '--prompt', '2+3', '--contract', 'number', '--output', str(target)])
            self.assertEqual(code, 0)
            row = json.loads(target.read_text())
            self.assertIn('2+3', row['prompt'])
            self.assertIn('prompt_tokens', row)
            backend.assert_not_called()


if __name__ == '__main__':
    unittest.main()
