"""Local vLLM and vLLM-compatible HTTP completions; no training or fallback."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from urllib import error, parse, request


@dataclass(frozen=True)
class Generation:
    token_ids: list[int]
    sampling: dict


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    seconds: float
    batch_size: int = 1


def load_tokenizer(model, revision=None, local_files_only=False):
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install barbet-skill-library[tokenizer] for model tokenization.') from exc
    options = {'revision': revision, 'local_files_only': local_files_only, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(model, **options)
    if not tokenizer.chat_template:
        raise ValueError('model tokenizer must provide a chat template')
    config = AutoConfig.from_pretrained(model, **options)
    text_config = getattr(config, 'text_config', config)
    eos = getattr(config, 'eos_token_id', None)
    if eos is None:
        eos = getattr(text_config, 'eos_token_id', tokenizer.eos_token_id)
    eos = sorted({x for x in (eos if isinstance(eos, list) else [eos]) if x is not None})
    return tokenizer, eos


class TokenizerBackend:
    """Prompt preparation only; does not instantiate a generation engine."""
    max_batch_size = 1

    def __init__(self, model, *, revision=None, max_model_len=49152,
                 local_files_only=False, tokenizer=None, eos_ids=None):
        if type(max_model_len) is not int or max_model_len < 1:
            raise ValueError('max_model_len must be a positive integer')
        if tokenizer is None:
            tokenizer, eos_ids = load_tokenizer(model, revision, local_files_only)
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids if eos_ids is not None else []
        self.terminal_strings = [tokenizer.decode([i], skip_special_tokens=False) for i in self.eos_ids]
        self.max_model_len = max_model_len
        self.metadata = {'backend': 'tokenizer_only', 'model': str(model), 'revision': revision,
                         'max_model_len': max_model_len}

    def generate(self, batch):
        raise RuntimeError('TokenizerBackend prepares prompts only')


class VLLMBackend(TokenizerBackend):
    def __init__(self, model, *, family='mini', revision=None, max_model_len=49152,
                 batch_size=8, gpu_memory_utilization=.7, tensor_parallel_size=1,
                 eager=True, local_files_only=False, tokenizer=None, eos_ids=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError('gpu_memory_utilization must be between zero and one')
        super().__init__(model, revision=revision, max_model_len=max_model_len,
                         local_files_only=local_files_only, tokenizer=tokenizer, eos_ids=eos_ids)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError('Install barbet-skill-library[vllm] for local inference.') from exc
        import importlib.metadata
        options = dict(model=str(model), revision=revision, tokenizer_revision=revision,
            dtype='bfloat16', max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size,
            enforce_eager=eager, max_num_seqs=batch_size, enable_prefix_caching=True,
            max_num_batched_tokens=16384, generation_config='vllm', trust_remote_code=False)
        if family == 'qwen':
            options['limit_mm_per_prompt'] = {'image': 0, 'video': 0}
        self.engine = LLM(**options)
        self.sampling_params = SamplingParams
        self.max_batch_size = batch_size
        self.metadata.update(backend='vllm', dtype='bfloat16', eager=eager,
            tensor_parallel_size=tensor_parallel_size,
            packages={p: importlib.metadata.version(p) for p in ('vllm', 'transformers', 'torch')})

    def generate(self, batch):
        started = time.monotonic()
        outputs = self.engine.generate([{'prompt_token_ids': g.token_ids} for g in batch],
            [self.sampling_params(**g.sampling, stop_token_ids=self.eos_ids,
                                  skip_special_tokens=False) for g in batch], use_tqdm=False)
        seconds = time.monotonic() - started
        if len(outputs) != len(batch):
            raise RuntimeError('vLLM returned the wrong batch size')
        return [Completion(out.outputs[0].text, len(g.token_ids), len(out.outputs[0].token_ids),
                           out.outputs[0].finish_reason, seconds, len(batch))
                for g, out in zip(batch, outputs)]


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPBackend(TokenizerBackend):
    """Requires vLLM extensions on /v1/completions; not a chat-only API adapter.

    Sends already-rendered token IDs, preserving the native template. The remote
    server must use the same tokenizer and revision. No automatic HTTP retries.
    """
    def __init__(self, model, *, base_url, served_model=None, api_key=None,
                 timeout=120, revision=None, max_model_len=49152,
                 local_files_only=False, tokenizer=None, eos_ids=None):
        url = parse.urlsplit(base_url)
        if (url.scheme not in {'http', 'https'} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError('base_url must be an HTTP(S) URL without credentials, query, or fragment')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        super().__init__(model, revision=revision, max_model_len=max_model_len,
                         local_files_only=local_files_only, tokenizer=tokenizer, eos_ids=eos_ids)
        self.url = base_url.rstrip('/') + '/completions'
        self.served_model = served_model or str(model)
        self._api_key = api_key
        self.timeout = timeout
        self.opener = request.build_opener(_NoRedirect())
        self.metadata.update(backend='vllm_http', base_url=base_url.rstrip('/'),
            served_model=self.served_model, server_revision_verified=False)

    def generate(self, batch):
        if len(batch) != 1:
            raise ValueError('HTTPBackend accepts one request at a time')
        generation = batch[0]
        payload = {'model': self.served_model, 'prompt': generation.token_ids,
            **generation.sampling, 'stop_token_ids': self.eos_ids,
            'skip_special_tokens': False, 'stream': False, 'n': 1, 'echo': False}
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = 'Bearer ' + self._api_key
        req = request.Request(self.url, data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        started = time.monotonic()
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                value = json.load(response)
        except error.HTTPError as exc:
            # Server error bodies can echo authorization headers or private prompts.
            code = exc.code
            exc.close()
            raise RuntimeError(f'inference server returned HTTP {code}; no retry was made') from None
        except (error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError('inference transport failed; remote completion/cost may be unknown; no retry was made') from None
        except (ValueError, UnicodeError):
            raise RuntimeError('inference server returned invalid JSON') from None
        try:
            choices = value['choices']
            if len(choices) != 1:
                raise ValueError()
            choice, usage = choices[0], value['usage']
            if not isinstance(choice['text'], str) or choice['finish_reason'] not in {'stop', 'length'}:
                raise ValueError()
            for key in ('prompt_tokens', 'completion_tokens'):
                if type(usage[key]) is not int or usage[key] < 0:
                    raise ValueError()
            return [Completion(choice['text'], usage['prompt_tokens'], usage['completion_tokens'],
                choice['finish_reason'], time.monotonic() - started)]
        except (KeyError, TypeError, ValueError, IndexError):
            raise RuntimeError('inference response requires one text completion, finish reason, and actual token usage') from None
