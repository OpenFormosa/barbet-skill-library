"""Parse literal keyword-call text into tool requests, without executing it.

This accepts calls, a list of calls, or newline-separated calls. It never infers
tool schemas, permits executable argument expressions, or runs a callback.
"""
import ast
import math


class UnsupportedCallText(ValueError): pass


def _literal(node, depth=0):
    if depth > 64: raise UnsupportedCallText('argument nesting bound')
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (str, bool, int, float))):
        value = node.value
        if isinstance(value, float) and not math.isfinite(value): raise UnsupportedCallText('nonfinite argument')
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, depth + 1)
        if type(value) not in (int, float): raise UnsupportedCallText('non-numeric sign')
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List): return [_literal(n, depth + 1) for n in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None: raise UnsupportedCallText('dictionary expansion')
            key = _literal(key, depth + 1)
            if not isinstance(key, str) or key in result: raise UnsupportedCallText('non-string or duplicate dictionary key')
            result[key] = _literal(value, depth + 1)
        return result
    raise UnsupportedCallText('argument must be a literal JSON-compatible value')


def _name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if not isinstance(node, ast.Name): raise UnsupportedCallText('call target must be a plain qualified name')
    parts.append(node.id)
    return '.'.join(reversed(parts))


def parse_keyword_calls(text):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 65536:
        raise UnsupportedCallText('empty or oversized call text')
    try: module = ast.parse(text.strip(), mode='exec')
    except (SyntaxError, ValueError, RecursionError) as exc: raise UnsupportedCallText('call syntax') from exc
    if sum(1 for _ in ast.walk(module)) > 4096: raise UnsupportedCallText('call syntax tree bound')
    nodes = []
    for statement in module.body:
        if not isinstance(statement, ast.Expr): raise UnsupportedCallText('only call expressions are allowed')
        if isinstance(statement.value, ast.List): nodes.extend(statement.value.elts)
        else: nodes.append(statement.value)
    if not 1 <= len(nodes) <= 32: raise UnsupportedCallText('call count bound')
    result = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args: raise UnsupportedCallText('only literal keyword calls are supported')
        name = _name(node.func); arguments = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in arguments: raise UnsupportedCallText('keyword expansion or duplicate argument')
            arguments[keyword.arg] = _literal(keyword.value)
        result.append({'type': 'function', 'function': {'name': name, 'arguments': arguments}})
    return result
