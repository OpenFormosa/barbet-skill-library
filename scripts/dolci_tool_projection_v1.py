"""Lossless source-format projection for diagnostics, not dataset admission.

Only explicit source function registries are allowed. An environment message
without a unique pending call remains ambiguous; no order-based result matching
or invented schema is used. Original observation text stays intact.
"""
import json
from barbet_inference.call_text import parse_keyword_calls, UnsupportedCallText
from barbet_inference.memory import canonical
from barbet_inference.sft import prepare_tools, normalize_messages, Rejected, RESERVED
from barbet_inference.tool_contract import trajectory_has_undeclared_arguments
from barbet_inference.source_contract_holds import held_tool_schemas
from scripts.ifm_memory_adapter_v1 import credential_gate


def project_tool_messages(source_messages, contract_policy):
    if not isinstance(source_messages, list): raise Rejected('source_messages_not_array')
    if RESERVED.search(canonical(source_messages)): raise Rejected('source_reserved_delimiter_or_thinking')
    registry = []; seen = set(); messages = []
    for source in source_messages:
        role = source.get('role')
        if role not in {'system', 'user', 'assistant', 'environment'}: raise Rejected('unsupported_source_role')
        functions = source.get('functions')
        if functions and functions.strip() not in {'', '[]', 'null'}:
            if role != 'system': raise Rejected('source_registry_outside_system')
            try: values = json.loads(functions)
            except (TypeError, ValueError) as exc: raise Rejected('invalid_explicit_source_registry') from exc
            if not isinstance(values, list): raise Rejected('explicit_registry_not_array')
            for value in values:
                key = canonical(value)
                if key not in seen: registry.append(value); seen.add(key)
        message = {'role': 'tool' if role == 'environment' else role, 'content': source.get('content')}
        calls = source.get('function_calls')
        if calls and calls.strip() not in {'', '[]', 'null'}:
            if role != 'assistant': raise Rejected('source_function_call_role')
            try: message['tool_calls'] = parse_keyword_calls(calls)
            except UnsupportedCallText as exc: raise Rejected('unsupported_literal_source_call: ' + str(exc)) from exc
        messages.append(message)
    if not registry: raise Rejected('no_explicit_source_registry')
    tools, validators = prepare_tools(registry)
    if held_tool_schemas(tools, contract_policy): raise Rejected('known_unresolved_source_tool_contract')
    projected = normalize_messages(messages, tools, validators)
    if trajectory_has_undeclared_arguments(projected, tools): raise Rejected('undeclared_source_call_argument')
    credential_gate(projected, tools)
    return {'messages': projected, 'tools': tools,
        'observation_origin': 'upstream_recorded_or_simulated',
        'source_programs_executed': False, 'schema_inferred_from_calls': False,
        'source_content_license_admitted': False, 'accepted_training_rows': 0}
