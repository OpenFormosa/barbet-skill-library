"""Bounded static signals for source admission; never execute candidate code.

Signals cause conservative question-wide holds, not a universal bug verdict.
Expected output and illustrative test programs are not observed executions.
"""
import ast
import re

FENCES = re.compile(r'```([^\n`]*)\n(.*?)```', re.S)
PAST_EXECUTION = re.compile(
    r"\b(?:(?:I|we)\s+(?:have\s+)?(?:ran|executed|tested)|"
    r"(?:I've|we've|I have|we have)\s+(?:run|executed|tested))\s+"
    r"(?:the|this|these|your)\s+(?:code|script|program|tests?|function|implementation)\b", re.I)


def signals(text, *, has_recorded_observation):
    findings = []
    for block_index, match in enumerate(FENCES.finditer(text)):
        if match[1].strip().lower() not in {'python', 'python3', 'py'}:
            continue
        try:
            tree = ast.parse(match[2])
        except (SyntaxError, ValueError, RecursionError):
            findings.append({'reason': 'explicit_python_block_not_parseable', 'block': block_index})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for key in node.keys:
                # Python's own equality includes e.g. True == 1. No evaluation
                # of source expressions, calls, names or comprehensions.
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
            findings.append({'reason': 'past_execution_claim_without_recorded_observation',
                             'claim': match[0]})
    return findings
