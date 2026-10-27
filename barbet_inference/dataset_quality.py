"""Conservative source-specific holds over recorded data, never code execution."""
import json
import re


TERMINAL_ERROR = re.compile(
    r'(?m)^\s*(?:Traceback \(most recent call last\):|'
    r'(?:SyntaxError|IndentationError|TabError|AssertionError|NameError|TypeError|ValueError|ImportError|ModuleNotFoundError):)')


def terminal_completion_hold(row):
    """Hold a completion claim after a last terminal observation has a traceback.

This identifies an unresolved validation concern, not proof that every printed
error invalidates a task. A later terminal observation supersedes the old one.
"""
    if row.get('upstream_source') != 'nvidia/Nemotron-Terminal-Corpus':
        return None
    messages = row.get('messages')
    if messages is None:
        messages = json.loads(row['messages_json'])
    final = messages[-1]
    if final['role'] != 'assistant' or not final.get('loss', True):
        return None
    try:
        claim = json.loads(final.get('content', ''))
    except (ValueError, TypeError):
        return None
    if not isinstance(claim, dict) or claim.get('task_complete') is not True:
        return None
    terminal = [m for m in messages[:-1] if m['role'] == 'tool' and m.get('name') == 'run_terminal']
    if terminal and TERMINAL_ERROR.search(terminal[-1]['content']):
        return 'unresolved_terminal_error_before_completion_claim'
    return None
