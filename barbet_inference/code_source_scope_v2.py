"""Respect Markdown indentation before static parsing of Python fences."""
import ast
import textwrap
from barbet_inference.code_source_scope import FENCES, PAST_EXECUTION


def signals(text, *, has_recorded_observation):
    findings = []
    for block_index, match in enumerate(FENCES.finditer(text)):
        if match[1].strip().lower() not in {'python', 'python3', 'py'}:
            continue
        try:
            tree = ast.parse(textwrap.dedent(match[2]))
        except (SyntaxError, ValueError, RecursionError):
            findings.append({'reason': 'explicit_python_block_not_parseable', 'block': block_index})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for key in node.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, (str, bytes, int, float, bool, type(None))):
                    continue
                if key.value in seen:
                    findings.append({'reason': 'duplicate_literal_dictionary_key_scope_hold',
                                     'block': block_index, 'line': key.lineno,
                                     'first_line': seen[key.value], 'key_repr': repr(key.value)[:160]})
                else:
                    seen[key.value] = key.lineno
    prose = FENCES.sub('\n[quoted code or output]\n', text)
    if not has_recorded_observation:
        for match in PAST_EXECUTION.finditer(prose):
            findings.append({'reason': 'past_execution_claim_without_recorded_observation', 'claim': match[0]})
    return findings
