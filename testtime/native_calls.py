"""Strict native syntax decoding, adapted from our earlier frozen study runtime.

Source: minicpm-skill-study runtime.py at 6858976. No semantic repair.
Both native families use the same syntax/value validation rules.
"""
import ast
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key:' + key)
        result[key] = value
    return result


def strict_json(text):
    return json.loads(text, object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite_json:' + x)))


def decode_xml_value(text, schema):
    # Both published XML parsers treat outer whitespace as tag formatting.
    # A string's internal whitespace and quote characters remain literal.
    value = text.strip()
    kind = schema.get('type') if isinstance(schema, dict) else None
    if kind == 'string':
        return value
    if kind == 'boolean':
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return value  # Never guess false for arbitrary text or quoted booleans.
    if kind == 'null' and value == 'null':
        return None
    if kind in ('integer', 'number'):
        try:
            parsed = strict_json(value)
            if type(parsed) in (int, float) and math.isfinite(parsed):
                return parsed
        except (ValueError, TypeError, OverflowError):
            pass
        return value
    if kind in ('array', 'object'):
        try:
            parsed = strict_json(value)
        except (ValueError, TypeError) as exc:
            if str(exc).startswith(('duplicate_json_key:', 'nonfinite_json:')):
                raise
            try:
                # MiniCPM's own chat template renders structured args as Python
                # literals. This is syntax decoding, never key/value repair.
                tree = ast.parse(value, mode='eval')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Dict):
                        keys = [ast.literal_eval(k) for k in node.keys]
                        if len(set(keys)) != len(keys):
                            raise ValueError('duplicate_literal_key')
                parsed = ast.literal_eval(tree)
                canonical(parsed)
            except (ValueError, TypeError, SyntaxError) as inner:
                if str(inner) == 'duplicate_literal_key':
                    raise
                return value
        return parsed
    try:
        return strict_json(value)
    except (ValueError, TypeError):
        return value


def _plain_json_calls(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text).removesuffix('```').strip()
    data = strict_json(text)
    if isinstance(data, dict) and 'tool_calls' in data:
        data = data['tool_calls']
    if not isinstance(data, list):
        data = [data]
    calls = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError('non_object_call')
        item = item.get('function', item)
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            raise ValueError('missing_function_name')
        args = item.get('arguments', item.get('parameters'))
        if isinstance(args, str):
            args = strict_json(args)
        if not isinstance(args, dict):
            raise ValueError('non_object_arguments')
        calls.append({'name': item['name'], 'arguments': args})
    return calls


def strict_parse_calls(text: str, tools: list | None = None):
    """Decode native syntax without dropping fields, renaming tools, or fixing values.

    Returns (all_calls, error_or_none). Unknown tool names and extra fields are
    deliberately retained. Duplicate keys/params or unfinished syntax fail.
    Both model families get identical value-decoding and JSON rules.
    """
    schemas = {t.get('function', t).get('name'): t.get('function', t).get('parameters', {})
               for t in tools or []}
    text = text.strip()
    try:
        if '<function' not in text:
            if '<tool_call>' in text:
                blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
                if text.count('<tool_call>') != len(blocks) or text.count('</tool_call>') != len(blocks):
                    raise ValueError('unclosed_tool_call')
                calls = []
                for block in blocks:
                    calls.extend(_plain_json_calls(block))
                return calls, None
            return _plain_json_calls(text), None
        # Shield CDATA payloads containing apparent closing tags.
        cdata = {}
        def protect(match):
            key = '__NATIVE_CDATA_' + sha256(match.group(0)) + '_' + str(len(cdata)) + '__'
            cdata[key] = match.group(0)
            return key
        shielded = re.sub(r'<!\[CDATA\[.*?\]\]>', protect, text, flags=re.S)
        def restore(value):
            for key, literal in cdata.items():
                value = value.replace(key, literal)
            return value
        pattern = re.compile(r'<function(?:\s+name\s*=\s*([\'\"])(.*?)\1\s*|=([^>]+))>(.*?)</function>', re.S)
        matches = list(pattern.finditer(shielded))
        remainder = pattern.sub('', shielded)
        if not matches or '<function' in remainder or '</function>' in remainder:
            raise ValueError('unclosed_or_malformed_function')
        if shielded.count('<tool_call>') != shielded.count('</tool_call>'):
            raise ValueError('unclosed_tool_call')
        calls = []
        for match in matches:
            name = (match.group(2) if match.group(1) else match.group(3)).strip()
            props = schemas.get(name, {}).get('properties', {})
            args = {}
            if match.group(1):
                try:
                    node = ET.fromstring(restore(match.group(0)))
                except ET.ParseError:
                    # The published MiniCPM parser supports unescaped literal
                    # payloads ("A < B & C") via a closed-tag fallback. Decode
                    # that syntax just like Qwen's native pseudo-XML; do not
                    # invent delimiters, discard params, or alter payloads.
                    body = match.group(4).strip()
                    if body.startswith('<arguments>') and body.endswith('</arguments>'):
                        body = body[len('<arguments>'):-len('</arguments>')]
                    pp = re.compile(r'<param\s+name\s*=\s*([\'\"])(.*?)\1\s*>(.*?)</param>', re.S)
                    params = list(pp.finditer(body))
                    if pp.sub('', body).strip():
                        raise ValueError('unclosed_or_malformed_parameter')
                    for param in params:
                        key, value = param.group(2), restore(param.group(3)).strip()
                        if key in args:
                            raise ValueError('duplicate_parameter:' + key)
                        if value.startswith('<![CDATA[') and value.endswith(']]>'):
                            value = value[9:-3]
                        args[key] = decode_xml_value(value, props.get(key, {}))
                else:
                    children = list(node)
                    if len(children) == 1 and children[0].tag == 'arguments':
                        children = list(children[0])
                    for param in children:
                        if param.tag != 'param' or 'name' not in param.attrib or len(param):
                            raise ValueError('malformed_parameter')
                        key = param.attrib['name']
                        if key in args:
                            raise ValueError('duplicate_parameter:' + key)
                        args[key] = decode_xml_value(param.text or '', props.get(key, {}))
            else:
                body = match.group(4)
                pp = re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.S)
                params = list(pp.finditer(body))
                if pp.sub('', body).strip():
                    raise ValueError('unclosed_or_malformed_parameter')
                for param in params:
                    key, value = param.group(1).strip(), restore(param.group(2))
                    if key in args:
                        raise ValueError('duplicate_parameter:' + key)
                    if value.startswith('<![CDATA[') and value.endswith(']]>'):
                        value = value[9:-3]
                    args[key] = decode_xml_value(value, props.get(key, {}))
            calls.append({'name': name, 'arguments': args})
        return calls, None
    except (ValueError, TypeError, ET.ParseError, KeyError) as exc:
        return [], type(exc).__name__ + ':' + str(exc)


def extract_final(raw: str, thinking: bool, tools=None, terminal_strings=()):
    content = raw.rstrip()
    # Remove only trailing native stop markers, not interior literal strings.
    changed = True
    while changed:
        changed = False
        for ending in terminal_strings:
            if ending and content.endswith(ending):
                content = content[:-len(ending)].rstrip()
                changed = True
    if '</think>' in content:
        return content.rsplit('</think>', 1)[1].strip(), False, False
    if thinking or '<think>' in content:
        # A sole complete terminal native call may implicitly end thinking.
        # Reject examples followed by prose or multiple tool calls.
        if tools:
            starts = [m.start() for m in re.finditer(r'<tool_call>|<function(?:\s+name|=)', content)]
            if starts:
                start = starts[0]
                candidate = content[start:].strip()
                if candidate.endswith(('</tool_call>', '</function>')):
                    calls, err = strict_parse_calls(candidate, tools)
                    if len(calls) == 1 and err is None:
                        return candidate, True, True
        return '', True, False
    return content.strip(), False, False
