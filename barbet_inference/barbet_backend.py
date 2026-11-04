"""Hash-bound native Barbet inference using the published precision-aware loader.

This adapter supports sequential greedy decoding. It does not claim Megatron
training parity or 1M serving validation. Tool encoding is a renderer contract:
the existing ContextAgent uses barbet_portable_v1 for tools.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

from .backends import Completion


class PangolinChatTokenizer:
    def __init__(self, tokenizer, template):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.raw = tokenizer
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters['tojson'] = lambda value, **kw: json.dumps(value, **kw)
        env.globals['raise_exception'] = self._fail
        self.chat_template = template
        self.template = env.from_string(template)

    @staticmethod
    def _fail(message):
        raise ValueError(message)

    def encode(self, text, add_special_tokens=False):
        return self.raw.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.raw.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, *, tools=(), tokenize=True,
                            add_generation_prompt=False, **kwargs):
        text = self.template.render(messages=messages, tools=list(tools),
            bos_token='<s>', eos_token='</s>', add_generation_prompt=add_generation_prompt,
            **kwargs)
        return self.encode(text) if tokenize else text


class NativeBarbetBackend:
    max_batch_size = 1

    def __init__(self, folder, manifest, *, device='cuda:0', max_model_len=8192):
        from barbet_pipeline.common import sha_file
        folder = Path(folder).resolve()
        required = {'model.safetensors', 'config.json', 'tokenizer.json', 'chat_template.jinja',
                    'modeling_barbet.py', 'configuration_barbet.py', 'load_barbet.py',
                    'megatron_long_context.py'}
        if not manifest.get('revision') or not required <= set(manifest.get('files', {})):
            raise ValueError('Explicit model revision and all executable/config hashes required')
        for name in required:
            if sha_file(folder / name) != manifest['files'][name]['sha256']:
                raise ValueError(f'Native Barbet identity mismatch: {name}')
        config = json.loads((folder / 'config.json').read_text())
        if type(max_model_len) is not int or not 1 <= max_model_len <= config['max_position_embeddings']:
            raise ValueError('Invalid total context budget')
        spec = importlib.util.spec_from_file_location('barbet_release_' + manifest['revision'], folder / 'load_barbet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model, raw = module.load_barbet(folder, device=device)
        self._generate = module.generate_tokens
        self.tokenizer = PangolinChatTokenizer(raw, (folder / 'chat_template.jinja').read_text())
        self.eos_ids = [config['eos_token_id']]
        self.terminal_strings = [self.tokenizer.decode(self.eos_ids)]
        self.max_model_len = max_model_len
        self.metadata = {'backend': 'barbet_native_loader', 'model': manifest['model_id'],
            'revision': manifest['revision'], 'loader_sha256': manifest['files']['load_barbet.py']['sha256'],
            'max_model_len': max_model_len, 'architecture_context': config['max_position_embeddings'],
            'precision': 'published_BF16_and_FP32_preserved', 'sampling': 'greedy_only',
            'long_context_generation_validation': 'NOT_RUN', 'megatron_training_parity': 'NOT_RUN',
            'cross_call_cache': 'full_reprefill_preserving_visible_history'}

    def generate(self, batch):
        if len(batch) != 1:
            raise ValueError('Native Barbet currently accepts one sequence at a time')
        g = batch[0]
        if set(g.sampling) - {'max_tokens', 'temperature'} or g.sampling.get('temperature', 0) != 0:
            raise ValueError('Only explicit greedy generation is supported')
        maximum = g.sampling.get('max_tokens')
        if type(maximum) is not int or maximum < 1 or not g.token_ids:
            raise ValueError('Positive prompt and generation budgets required')
        if len(g.token_ids) + maximum > self.max_model_len:
            raise ValueError('Prompt plus output exceeds budget; no truncation')
        if any(type(x) is not int or not 0 <= x < self.model.config.vocab_size for x in g.token_ids):
            raise ValueError('Invalid Pangolin token ID')
        started = time.monotonic()
        ids = self._generate(self.model, g.token_ids, max_new_tokens=maximum)
        return [Completion(self.tokenizer.decode(ids), len(g.token_ids), len(ids),
            'stop' if ids and ids[-1] in self.eos_ids else 'length', time.monotonic() - started)]
