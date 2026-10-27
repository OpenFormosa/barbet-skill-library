"""Application harness around the unchanged, sealed experimental core."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from importlib.resources import files
import json
import re

from testtime.contracts import PublicRequest, strict_json
from testtime.harness import Episode, Policy
from testtime.retrieval import Document, LexicalIndex

from .backends import Generation
from .library import SkillLibrary

CONTRACTS = ('text', 'short_answer', 'number', 'python', 'tool_call', 'choice')
FORMATS = {
    'text': '依照題目要求作答。',
    'short_answer': '只輸出問題所問的最短答案，不加解說。',
    'number': 'Return the final numeric answer in the format: #### <number>.',
    'python': 'Return only the complete Python implementation, with the requested function signature.',
    'tool_call': 'Return exactly one call to a registered tool using the requested literal values.',
    'choice': '最後只輸出一個選項字母：A、B、C 或 D。',
}


def make_request(prompt, output_contract='short_answer', *, id='request-1', **kwargs):
    if output_contract not in CONTRACTS:
        raise ValueError('unknown output contract')
    return validate_request(PublicRequest(id=id, prompt=prompt + '\n\n' + FORMATS[output_contract],
                                         output_contract=output_contract, **kwargs))


def validate_request(value):
    raw = value if isinstance(value, dict) else asdict(value)
    # The frozen core has a plain-string answer channel. Keep the application
    # contract explicit while validating its fields through that core boundary.
    free_text = raw.get('output_contract') == 'text'
    req = PublicRequest.from_dict({**raw, 'output_contract': 'short_answer'} if free_text else raw)
    if not isinstance(req.id, str) or not req.id.strip():
        raise ValueError('request id must be a nonempty string')
    if not isinstance(req.prompt, str) or not req.prompt.strip():
        raise ValueError('request prompt must be a nonempty string')
    if not isinstance(req.setup, str):
        raise ValueError('request setup must be a string')
    for name in ('document_ids', 'public_tests'):
        values = getattr(req, name)
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(name + ' must be a list of strings')
    if len(set(req.document_ids)) != len(req.document_ids):
        raise ValueError('duplicate document id in request')
    if not isinstance(req.tools, list):
        raise ValueError('tools must be a list')
    names = []
    for tool in req.tools:
        if not isinstance(tool, dict):
            raise ValueError('each tool must be an object')
        definition = tool.get('function', tool)
        if not isinstance(definition, dict) or not isinstance(definition.get('name'), str):
            raise ValueError('tool name is required')
        params = definition.get('parameters')
        if not isinstance(params, dict) or params.get('type') != 'object':
            raise ValueError('tool parameters must have type object')
        validate_schema(params)
        names.append(definition['name'])
    if len(names) != len(set(names)):
        raise ValueError('duplicate tool name')
    if req.output_contract == 'tool_call' and not names:
        raise ValueError('tool_call requests require registered tools')
    return replace(req, output_contract='text') if free_text else req


def validate_schema(schema):
    supported = {'type', 'description', 'title', 'properties', 'required', 'additionalProperties', 'items', 'enum'}
    if not isinstance(schema, dict) or set(schema) - supported:
        raise ValueError('unsupported tool schema keyword; use the documented schema subset')
    kind = schema.get('type')
    if kind not in {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}:
        raise ValueError('tool schema requires a supported type')
    if 'enum' in schema and (not isinstance(schema['enum'], list) or not schema['enum']):
        raise ValueError('enum must be a nonempty list')
    if kind == 'object':
        props, required = schema.get('properties', {}), schema.get('required', [])
        if (not isinstance(props, dict) or not isinstance(required, list)
                or any(not isinstance(x, str) or x not in props for x in required)
                or type(schema.get('additionalProperties', True)) is not bool):
            raise ValueError('invalid object schema')
        for value in props.values():
            validate_schema(value)
    if kind == 'array':
        validate_schema(schema.get('items'))


def recommended_policy():
    value = strict_json(files('testtime').joinpath('recommended_policy.json').read_text())
    return Policy(name=value['name'], **value['config'])


def validate_policy(policy, max_calls):
    if type(max_calls) is not int or not 1 <= max_calls <= 3:
        raise ValueError('max_calls must be 1, 2 or 3')
    if not isinstance(policy, Policy) or not isinstance(policy.name, str):
        raise ValueError('policy must be a named Policy')
    if not isinstance(policy.contract_overrides, dict) or set(policy.contract_overrides) - set(CONTRACTS):
        raise ValueError('unknown contract override')
    for contract in CONTRACTS:
        p = policy.for_request(PublicRequest('check', 'check', contract))
        for key in ('skill', 'retrieval', 'program', 'execute', 'compact', 'evidence',
                    'thinking', 'literal_answer', 'knowledge_retrieval'):
            if type(getattr(p, key)) is not bool:
                raise ValueError(key + ' must be boolean')
        for key in ('input_tokens', 'output_tokens', 'continuation_input_tokens', 'top_k'):
            if type(getattr(p, key)) is not int or getattr(p, key) < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ('repair', 'self_revise'):
            if type(getattr(p, key)) is not int or not 0 <= getattr(p, key) <= 2:
                raise ValueError(key + ' must be between zero and two')
        if p.evidence and max_calls < 2:
            raise ValueError('evidence-first policy requires at least two calls')
        if p.tool_format not in {'native', 'json'} or p.program_format not in {'json', 'python_stdout'}:
            raise ValueError('unknown generation interface')


class Harness:
    def __init__(self, backend, *, family='mini', policy=None, documents=(),
                 skills=(), library=None, max_calls=3, seed=20261027):
        if family not in {'mini', 'qwen'}:
            raise ValueError('supported model families: mini, qwen')
        self.backend, self.family = backend, family
        self.policy = policy or recommended_policy()
        validate_policy(self.policy, max_calls)
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        self.max_calls, self.seed = max_calls, seed
        docs = [d if isinstance(d, Document) else Document(**d) for d in documents]
        if any(not isinstance(getattr(d, k), str) for d in docs for k in ('id', 'text', 'source', 'title')):
            raise ValueError('document fields must be strings')
        if any(not d.id for d in docs) or len({d.id for d in docs}) != len(docs):
            raise ValueError('document ids must be nonempty and unique')
        self.documents = {d.id: d for d in docs}
        raw_cards = files('testtime').joinpath('cards.json').read_bytes()
        self.cards_sha256 = hashlib.sha256(raw_cards).hexdigest()
        self.cards = {c['id']: c for c in strict_json(raw_cards.decode())['cards']}
        self.manual_skills = []
        if skills:
            text, self.manual_skills = (library or SkillLibrary()).render(skills)
            self.cards = {contract: {'id': contract, 'text': text} for contract in CONTRACTS}
        self.index = None
        if self.policy.knowledge_retrieval or any(v.get('knowledge_retrieval') for v in self.policy.contract_overrides.values()):
            self.index = LexicalIndex(docs)

    def _episode(self, value):
        req = validate_request(value)
        if req.output_contract == 'python' and req.public_tests:
            req = replace(req, prompt=req.prompt + '\n\nPublic tests (preserve these checks):\n'
                          + '\n'.join(req.public_tests))
        # Providing a corpus to a reading request opts into searching that corpus.
        if req.output_contract in {'text', 'short_answer'} and not req.document_ids and self.documents:
            req = replace(req, document_ids=list(self.documents))
        missing = set(req.document_ids) - self.documents.keys()
        if missing:
            raise ValueError('unknown document ids: ' + ', '.join(sorted(missing)))
        p = self.policy.for_request(req)
        public_contract = req.output_contract
        if public_contract == 'text':
            req = replace(req, output_contract='short_answer')
            p = replace(p, literal_answer=False, evidence=False, skill=False)
        p = replace(p, repair=min(p.repair, self.max_calls - 1),
                    self_revise=min(p.self_revise, self.max_calls - 1))
        if self.manual_skills:
            p = replace(p, skill=True)
        episode = Episode(req, p, self.documents, self.cards, self.backend.tokenizer,
                          self.family, self.seed, self.index)
        episode.public_output_contract = public_contract
        return episode

    def _generation(self, episode):
        ids, sampling, _ = episode.generate_request()
        if len(ids) + sampling['max_tokens'] > self.backend.max_model_len:
            raise ValueError('prompt plus output budget exceeds backend context ceiling: ' + episode.request.id)
        return Generation(ids, sampling)

    def prepare(self, request):
        episode = self._episode(request)
        generation = self._generation(episode)
        return {'id': episode.request.id, 'prompt': episode.rendered_prompt,
                'prompt_tokens': len(generation.token_ids), 'sampling': generation.sampling,
                'selected_sources': [{k:v for k,v in s.items() if k not in {'text','title'}} for s in episode.selected],
                'omitted_sources': episode.omitted, 'inference': self._metadata(episode)}

    def _metadata(self, episode):
        return {'schema_version': 1, 'backend': self.backend.metadata, 'family': self.family,
                'policy': asdict(episode.policy), 'max_calls': self.max_calls,
                'cards_sha256': self.cards_sha256, 'manual_skills': self.manual_skills,
                'output_contract': episode.public_output_contract,
                'request_sha256': hashlib.sha256(json.dumps(asdict(episode.request), sort_keys=True,
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                'correctness': 'not_evaluated'}

    def run(self, request, *, on_call=None):
        return next(self.run_many([request], on_call=on_call))

    def run_many(self, requests, *, on_call=None):
        requests = [validate_request(r) for r in requests]
        if len({r.id for r in requests}) != len(requests):
            raise ValueError('duplicate request id')
        # Fail before generating any answer for a malformed request or unknown source.
        episodes = [self._episode(r) for r in requests]
        for episode in episodes:
            self._generation(episode)
        batch_size = self.backend.max_batch_size
        for offset in range(0, len(episodes), batch_size):
            group = episodes[offset:offset + batch_size]
            while any(not e.done for e in group):
                active = [e for e in group if not e.done]
                if any(len(e.trace) >= self.max_calls for e in active):
                    raise RuntimeError('configured generation call ceiling reached')
                outputs = self.backend.generate([self._generation(e) for e in active])
                if len(outputs) != len(active):
                    raise RuntimeError('backend returned the wrong batch size')
                # Log all returned completions before host execution can fail.
                if on_call:
                    for episode, output in zip(active, outputs):
                        on_call({'event': 'completion', 'id': episode.request.id,
                            'call_index': len(episode.trace) + 1, 'seed': self.seed,
                            'prompt': episode.rendered_prompt, 'sampling': episode.last_sample,
                            **asdict(output)})
                for episode, output in zip(active, outputs):
                    episode.accept(output.text, prompt_tokens=output.prompt_tokens,
                        output_tokens=output.output_tokens, finish_reason=output.finish_reason,
                        batch_seconds=output.seconds, batch_size=output.batch_size,
                        terminal_strings=self.backend.terminal_strings)
            for episode in group:
                result = episode.result()
                result['validation'] = validation_summary(episode)
                result['inference'] = self._metadata(episode)
                yield result


def validation_summary(episode):
    last = episode.trace[-1]
    if last['unclosed_thinking']:
        return {'status': 'unfinished_thinking', 'passed': False, 'semantic_correctness': 'unknown'}
    if episode.request.output_contract == 'tool_call':
        passed = not last['schema_errors']
        return {'status': 'schema_valid' if passed else 'schema_invalid', 'passed': passed,
                'errors': last['schema_errors'], 'semantic_correctness': 'unknown'}
    if 'execution' in last:
        receipt = last['execution']
        checks = bool(episode.request.public_tests)
        status = ('public_tests_passed' if checks else 'execution_succeeded') if receipt['passed'] else 'execution_failed'
        return {'status': status, 'passed': receipt['passed'], 'receipt': receipt,
                'semantic_correctness': 'unknown'}
    contract, final = episode.request.output_contract, last['final'].strip()
    if contract == 'choice':
        valid = bool(re.fullmatch('[ABCD]', final))
    elif contract == 'number':
        value = final.rsplit('####', 1)[-1].strip().replace(',', '')
        valid = bool(re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?', value))
        if valid and '/' in value:
            valid = not bool(re.fullmatch(r'[-+]?0+', value.rsplit('/', 1)[1]))
    else:
        return {'status': 'not_checked', 'passed': None, 'semantic_correctness': 'unknown'}
    return {'status': 'format_valid' if valid else 'format_invalid', 'passed': valid,
            'semantic_correctness': 'unknown'}
