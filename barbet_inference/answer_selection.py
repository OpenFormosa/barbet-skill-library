"""Conditional answer likelihood; a ranking signal after independent quality gates.

No generation, training, truncated-context approximation or score imputation.
"""
from __future__ import annotations

import hashlib
import math

from .memory import canonical
from .sft import tokenize_spans


def input_identity(messages, tools=()):
    """Do not confuse identical first questions with identical evidence/contracts."""
    aliases = {}
    for m in messages:
        for c in m.get('tool_calls', []):
            aliases[c['id']] = 'call_' + str(len(aliases))
    inputs = []
    for m in messages:
        if m['role'] == 'assistant' and m.get('loss', True):
            continue
        value = {k: m[k] for k in ('role', 'content', 'name', 'tool_call_id', 'tool_calls') if k in m}
        if 'tool_call_id' in value:
            value['tool_call_id'] = aliases.get(value['tool_call_id'], value['tool_call_id'])
        inputs.append(value)
    return hashlib.sha256(canonical({'inputs': inputs, 'tools': list(tools)}).encode()).hexdigest()


def select_minimum_ppl(candidates):
    """Fail closed unless all eligible competitors have a comparable real score."""
    eligible = [c for c in candidates if c.get('eligible') is True]
    if not eligible:
        raise ValueError('no eligible answers')
    group = eligible[0]['input_hash']
    protocol = eligible[0]['protocol_sha256']
    for c in eligible:
        if c['input_hash'] != group or c['protocol_sha256'] != protocol:
            raise ValueError('incomparable input or scoring protocol')
        if (c.get('status') != 'scored' or c.get('answer_tokens', 0) <= 0
                or not isinstance(c.get('mean_nll'), (int, float))
                or not math.isfinite(c['mean_nll']) or c['mean_nll'] < 0):
            raise ValueError('missing or invalid competing score')
    return min(eligible, key=lambda c: (c['mean_nll'], c['record_id']))


class NativeAnswerRenderer:
    """Use the pinned release template; locate targets without altering its bytes."""
    def __init__(self, template, tokenizer):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        import json
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters['tojson'] = lambda value, **kw: json.dumps(value, **kw)
        env.globals['raise_exception'] = self._fail
        self.template = env.from_string(template)
        self.tokenizer = tokenizer

    @staticmethod
    def _fail(message):
        raise ValueError(message)

    def render(self, messages, tools=()):
        def render_part(part, generation=False):
            return self.template.render(messages=part, tools=list(tools), bos_token='<s>',
                                        eos_token='</s>', add_generation_prompt=generation)
        text = render_part(messages)
        spans = []
        for i, message in enumerate(messages):
            if message['role'] != 'assistant' or not message.get('loss', True):
                continue
            prefix = render_part(messages[:i], True)
            ending = render_part(messages[:i + 1])
            if not text.startswith(ending) or not ending.startswith(prefix):
                raise ValueError('native template target prefix mismatch')
            spans.append((len(prefix), len(ending), i))
        ids, labels = tokenize_spans(self.tokenizer, text, spans)
        if labels[0] != -100:
            raise ValueError('first token cannot have a causal target')
        return {'input_ids': ids, 'labels': labels, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'input_hash': input_identity(messages, tools),
                'answer_tokens': sum(x != -100 for x in labels), 'sequence_tokens': len(ids)}


def score_batch(model, items, *, maximum_tokens=32768, logit_chunk_size=128):
    """Right-pad independent sequences; score only shifted answer-token positions.

    Padding is strictly after every real token, so causal attention/Mamba cannot
    expose it to any retained hidden state. A live padded-versus-single check is
    required before corpus scoring. BF16 checkpoint parameters are not recast.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not items:
        return []
    length = max(len(x['input_ids']) for x in items)
    if length > 131072 or (len(items) > 1 and length * len(items) > maximum_tokens):
        raise ValueError('batch budget exceeded; no truncation')
    device = model.model.embed_tokens.weight.device
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ids = torch.full((len(items), length), model.config.pad_token_id, device=device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for i, item in enumerate(items):
            if len(item['input_ids']) != len(item['labels']) or item['labels'][0] != -100:
                raise ValueError('invalid causal labels')
            ids[i, :len(item['input_ids'])] = torch.tensor(item['input_ids'], device=device)
            labels[i, :len(item['labels'])] = torch.tensor(item['labels'], device=device)
        hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        results = []
        for i, item in enumerate(items):
            positions = (labels[i, 1:] != -100).nonzero().flatten()
            if not len(positions):
                raise ValueError('zero scored answer tokens')
            total = torch.zeros((), device=device, dtype=torch.float64)
            # Keep the original sequence-aligned matrix shapes used by the
            # published loss. Gathering only target states can select a different
            # BF16 GEMM kernel and change close rankings. Skip only whole chunks
            # that have no targets; each retained chunk is otherwise unchanged.
            shifted = torch.full_like(labels[i], -100)
            shifted[:-1] = labels[i, 1:]
            chunks = (positions // logit_chunk_size).unique().tolist()
            for number in chunks:
                start = number * logit_chunk_size
                end = min(start + logit_chunk_size, len(item['input_ids']))
                logits = model.lm_head(hidden[i, start:end]).float()
                loss = F.cross_entropy(logits, shifted[start:end], ignore_index=-100, reduction='none')
                if not torch.isfinite(loss).all():
                    raise ValueError('nonfinite answer NLL')
                total += loss.double().sum()
            nll = total.item()
            mean = nll / len(positions)
            results.append({'status': 'scored', 'answer_tokens': len(positions), 'nll_sum': nll,
                            'mean_nll': mean, 'ppl': math.exp(mean),
                            'sequence_tokens': len(item['input_ids']), 'text_sha256': item['text_sha256'],
                            'input_hash': item['input_hash']})
        return results
