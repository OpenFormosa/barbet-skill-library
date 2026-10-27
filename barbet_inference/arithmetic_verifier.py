"""Exact, bounded arithmetic verification without eval or source-code execution.

Only an entire explicitly requested integer calculation is eligible. Arbitrary
natural-language math, functions, names and code are outside this verifier.
"""
from fractions import Fraction
import re

REQUEST = re.compile(r'^(?:Calculate|Evaluate|Compute)(?: the value of| the expression)?\s+(.+?)\.\s*(?:Provide|Give) your answer as a single integer\.?$', re.I | re.S)
INTEGER = re.compile(r'^[+-]?\d+$')
TOKEN = re.compile(r'\s*(\d+(?:\.\d+)?|\*\*|[+*/^(){}-]|\\(?:frac|times|cdot))')


class UnsupportedArithmetic(ValueError): pass


def evaluate_arithmetic(text):
    if len(text) > 2000: raise UnsupportedArithmetic('expression byte scope')
    tokens = []; cursor = 0
    while cursor < len(text):
        if not text[cursor:].strip(): break
        match = TOKEN.match(text, cursor)
        if not match: raise UnsupportedArithmetic('unrecognized arithmetic token')
        tokens.append(match.group(1)); cursor = match.end()
        if len(tokens) > 256: raise UnsupportedArithmetic('expression token scope')
    position = 0

    def peek(): return tokens[position] if position < len(tokens) else None

    def pop():
        nonlocal position
        value = peek()
        if value is None: raise UnsupportedArithmetic('incomplete expression')
        position += 1; return value

    def bounded(value):
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise UnsupportedArithmetic('arithmetic result scope')
        return value

    def group(depth):
        opening = pop()
        if opening not in {'(', '{'}: raise UnsupportedArithmetic('group required')
        value = expression(depth + 1)
        if pop() != (')' if opening == '(' else '}'): raise UnsupportedArithmetic('unbalanced group')
        return value

    def atom(depth):
        if depth > 64: raise UnsupportedArithmetic('expression nesting scope')
        if peek() in {'(', '{'}: return group(depth)
        if peek() == r'\frac':
            pop(); numerator = group(depth); denominator = group(depth)
            if denominator == 0: raise UnsupportedArithmetic('division by zero')
            return bounded(numerator / denominator)
        value = pop()
        if not re.fullmatch(r'\d+(?:\.\d+)?', value): raise UnsupportedArithmetic('number required')
        if len(value) > 100: raise UnsupportedArithmetic('integer literal scope')
        return Fraction(value)

    def power(depth):
        value = atom(depth)
        if peek() in {'^', '**'}:
            pop(); exponent = unary(depth + 1)
            if exponent.denominator != 1 or abs(exponent.numerator) > 64:
                raise UnsupportedArithmetic('exponent scope')
            if value == 0 and exponent <= 0: raise UnsupportedArithmetic('undefined zero power')
            value = bounded(value ** exponent.numerator)
        return value

    def unary(depth):
        if depth > 64: raise UnsupportedArithmetic('expression nesting scope')
        if peek() in {'+', '-'}:
            sign = pop(); value = unary(depth + 1)
            return -value if sign == '-' else value
        return power(depth)

    def term(depth):
        value = unary(depth)
        while peek() in {'*', '/', r'\times', r'\cdot'}:
            operation = pop(); other = unary(depth)
            if operation == '/' and other == 0: raise UnsupportedArithmetic('division by zero')
            value = bounded(value / other if operation == '/' else value * other)
        return value

    def expression(depth):
        value = term(depth)
        while peek() in {'+', '-'}:
            operation = pop(); other = term(depth)
            value = bounded(value + other if operation == '+' else value - other)
        return value

    if not tokens: raise UnsupportedArithmetic('empty expression')
    result = expression(0)
    if position != len(tokens): raise UnsupportedArithmetic('trailing expression tokens')
    return result


def verify_integer_calculation(question, answer):
    match = REQUEST.fullmatch(question.strip())
    if not match: return {'status': 'outside_scope'}
    expression = match.group(1).strip()
    for left, right in [(r'\(', r'\)'), ('$', '$')]:
        if expression.startswith(left) and expression.endswith(right):
            expression = expression[len(left):-len(right)].strip(); break
    target = answer.strip()
    if target.startswith('**') and target.endswith('**'): target = target[2:-2].strip()
    if target.startswith(r'\boxed{') and target.endswith('}'): target = target[7:-1].strip()
    if not INTEGER.fullmatch(target) or len(target) > 100: return {'status': 'outside_scope'}
    try: expected = evaluate_arithmetic(expression)
    except UnsupportedArithmetic as exc: return {'status': 'outside_scope', 'reason': str(exc)}
    if expected.denominator != 1: return {'status': 'outside_scope', 'reason': 'requested integer but expression is nonintegral'}
    return {'status': 'verified' if int(target) == expected.numerator else 'incorrect',
            'expression': expression, 'expected_integer': str(expected.numerator), 'supplied_integer': target,
            'method': 'bounded exact rational arithmetic; no eval, source program execution or language-model judgment'}
