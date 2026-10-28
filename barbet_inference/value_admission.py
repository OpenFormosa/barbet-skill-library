"""Fail-closed identity binding for a contextual values admission record.

The caller owns the immutable reviewed-system allowlist and prior query holds.
This module never treats a missing review, a source name or a low PPL as approval.
"""
import hashlib
import json
from barbet_inference.memory import canonical


def bind_contract_identity(row, trusted_system_hashes):
    messages=json.loads(row['messages_json']);tools=json.loads(row['tools_json'])
    if not messages or messages[0].get('role')!='system' or messages[0].get('loss'):
        raise ValueError('Untrusted system layout')
    digest=hashlib.sha256(messages[0]['content'].encode()).hexdigest()
    if digest not in trusted_system_hashes:raise ValueError('System instructions lack a reviewed exact binding')
    identity=hashlib.sha256(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools}).encode()).hexdigest()
    if identity!=row['record_id']:raise ValueError('Complete input/target identity differs')
    return digest


def disposition(*, query_held, sensitive, contextual_verdict=None, bound_contract=False):
    if query_held:return 'HOLD_PRIOR_QUERY'
    if contextual_verdict is not None and contextual_verdict!='ACCEPT':return 'HOLD_FIRST_NONACCEPT'
    if contextual_verdict=='ACCEPT':return 'CONTEXTUAL_ACCEPT'
    if bound_contract:return 'PROVEN_SOURCE_BOUND_CONTRACT'
    if sensitive:return 'HOLD_PENDING_CONTEXTUAL_REVIEW'
    return 'SCREEN_NEGATIVE_NOT_SEMANTICALLY_CERTIFIED'
