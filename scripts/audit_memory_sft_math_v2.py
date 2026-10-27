"""Prospective math audit transport with a larger measured completion reserve.

The first IFM source cohort remains frozen, including incomplete dispositions.
This wrapper may only audit new jobs. It preserves prompt, schema, validation,
teacher and two-worker scheduling; it never saves private model reasoning.
"""
from datetime import datetime, timezone
import json
import time
import urllib.request

from scripts import audit_memory_sft_v2 as shared

MATH_RESERVE = 8192


def body_for(job):
    math = job['domain'] == 'math'
    payload = {'messages': job['messages'], 'tools': job.get('tools', []),
               'observation_origin': job.get('observation_origin', 'provided_document')}
    return {'model': shared.MODEL, 'messages': [{'role': 'system', 'content': shared.PROMPT},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        'temperature': 0, 'max_tokens': MATH_RESERVE if math else 900,
        'chat_template_kwargs': {'enable_thinking': math},
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'sft-audit', 'schema': shared.SCHEMA}}}


def safe_response(choice):
    # In particular, do not retain message.reasoning or reasoning_content.
    return {'content': choice['message']['content']}


def review(job):
    folder = shared.STATE / 'semantic-audit-v1' / job['id']; body = body_for(job)
    response = folder / 'response.json'; receipt = folder / 'receipt.json'; request = folder / 'request.json'
    if request.exists() and not receipt.exists(): raise ValueError('interrupted first math audit; no replay')
    shared.dump(request, body)
    if not receipt.exists():
        started = time.monotonic()
        req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
        with shared.teacher_slot():
            with urllib.request.urlopen(req, timeout=1200) as handle: raw = json.load(handle)
        choice = raw['choices'][0]; shared.dump(response, safe_response(choice))
        shared.dump(receipt, {'model': raw['model'], 'revision': shared.REV, 'request_id': raw['id'],
            'finish_reason': choice['finish_reason'], 'usage': raw.get('usage'),
            'request_sha256': shared.digest(body), 'seconds': time.monotonic() - started,
            'reasoning_saved': False, 'created_at': datetime.now(timezone.utc).isoformat(),
            'math_completion_reserve': MATH_RESERVE, 'previous_first_audits_replayed': False})
    meta = json.loads(receipt.read_text())
    if meta['request_sha256'] != shared.digest(body): raise ValueError('first request binding differs')
    result = {'id': job['id'], 'record_id': job.get('record_id'), 'domain': job['domain'], 'usage': meta['usage']}
    if meta['model'] != shared.MODEL or meta['finish_reason'] != 'stop':
        result.update(verdict='HOLD', issues=['incomplete or wrong-model response'])
    else:
        try: result.update(shared.validate(json.loads(json.loads(response.read_text())['content'])))
        except Exception as exc: result.update(verdict='HOLD', issues=['strict audit format: ' + type(exc).__name__])
    shared.dump(folder / 'disposition.json', result)
    print(json.dumps({k: result[k] for k in ['id', 'domain', 'verdict']}), flush=True)
    return result
