"""High-precision recorded error/retry tags, never proof of real execution."""
import json
import re


def meaningful_error(value):
    if value is None or value is False or value == 0: return False
    if isinstance(value, str): return value.strip().lower() not in {'', 'none', 'null', 'false', '0', 'no error'}
    if isinstance(value, dict): return any(meaningful_error(v) for v in value.values())
    if isinstance(value, list): return any(meaningful_error(v) for v in value)
    return bool(value)


def observation_status(message):
    try: value = json.loads(message['content'])
    except (ValueError, TypeError): value = None
    if isinstance(value, dict):
        error = (meaningful_error(value.get('error')) or value.get('success') is False or value.get('ok') is False
                 or str(value.get('status', '')).lower() in {'error', 'failed', 'failure', 'timeout'}
                 or any(type(value.get(k)) is int and value[k] != 0 for k in ['exit_code','returncode']))
        success = (not error and (value.get('success') is True or value.get('ok') is True
                   or str(value.get('status','')).lower() in {'ok','success','passed'}
                   or any(type(value.get(k)) is int and value[k] == 0 for k in ['exit_code','returncode'])))
        return bool(error), bool(success)
    if message.get('name') in {'run_terminal', 'terminal', 'execute', 'execute_python'}:
        text = message['content'].lstrip()
        if re.match(r'Traceback \(most recent call last\):|(?:SyntaxError|AssertionError|TypeError|ValueError):', text):
            return True, False
    return False, False


def recorded_refinement_evidence(messages):
    errors = {}; retry_ids = {}; counts = {'explicit_error_observations': 0, 'same_tool_retries': 0,
        'retry_observations_without_explicit_error': 0, 'retry_observations_with_explicit_success': 0}
    for turn, message in enumerate(messages):
        if message['role'] == 'assistant' and message.get('loss'):
            for call in message.get('tool_calls', []):
                name = call['function']['name']
                if name in errors:
                    counts['same_tool_retries'] += 1
                    retry_ids[call['id']] = {'name': name, 'after_error_turn': errors[name]}
        elif message['role'] == 'tool':
            error, success = observation_status(message)
            if message.get('tool_call_id') in retry_ids:
                if not error: counts['retry_observations_without_explicit_error'] += 1
                if success: counts['retry_observations_with_explicit_success'] += 1
                retry_ids.pop(message['tool_call_id'])
            if error:
                counts['explicit_error_observations'] += 1; errors[message['name']] = turn
            elif message.get('name') in errors:
                errors.pop(message['name'])
    return counts
