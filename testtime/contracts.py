"""Public request and strict schema validation for optional inference harnesses."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math


def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate key: ' + k)
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite: ' + x)))


@dataclass
class PublicRequest:
    id: str
    prompt: str
    output_contract: str
    tools: list = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    public_tests: list[str] = field(default_factory=list)
    setup: str = ''

    @classmethod
    def from_dict(cls, value):
        allowed = set(cls.__dataclass_fields__)
        if set(value) - allowed:
            raise ValueError('non-public fields: ' + ','.join(sorted(set(value) - allowed)))
        req = cls(**value)
        if req.output_contract not in {'number', 'python', 'tool_call', 'short_answer', 'choice'}:
            raise ValueError('unknown requested output contract')
        return req


def validate_value(value, schema, path='$'):
    errors = []
    kind = schema.get('type')
    type_ok = {'string': isinstance(value, str), 'integer': type(value) is int,
        'number': type(value) in (int, float) and math.isfinite(value),
        'boolean': type(value) is bool, 'object': isinstance(value, dict),
        'array': isinstance(value, list), 'null': value is None}.get(kind, True)
    if not type_ok:
        return [f'{path}: expected {kind}']
    if 'enum' in schema and not any(type(value) is type(x) and value == x for x in schema['enum']):
        errors.append(f'{path}: not in permitted enum')
    if kind == 'object':
        props = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(f'{path}.{key}: required')
        for key, v in value.items():
            if key not in props:
                if schema.get('additionalProperties', True) is False:
                    errors.append(f'{path}.{key}: unexpected field')
            else:
                errors.extend(validate_value(v, props[key], path + '.' + key))
    if kind == 'array':
        for i, v in enumerate(value):
            errors.extend(validate_value(v, schema.get('items', {}), f'{path}[{i}]'))
    return errors


def validate_call(call, tools):
    if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
        return ['expected exactly name and arguments']
    schemas = {t.get('function', t)['name']: t.get('function', t)['parameters'] for t in tools}
    if not isinstance(call['name'], str) or call['name'] not in schemas:
        return ['unknown tool name; choose an exact registered name']
    return validate_value(call['arguments'], schemas[call['name']])
