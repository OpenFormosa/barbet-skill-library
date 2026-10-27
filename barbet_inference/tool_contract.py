"""Conservative function-interface curation beyond permissive JSON Schema defaults."""
import re


def undeclared_arguments(arguments, schema):
    """Return root arguments absent from an explicitly enumerated interface.

    JSON Schema defaults additionalProperties to true. Many function schemas omit
    this keyword while listing their actual parameters. This guard does not call
    such schemas invalid or mutate them: it holds unconfirmed fields. Explicit
    open dictionaries and pattern properties remain supported.
    """
    if not isinstance(arguments,dict) or not isinstance(schema,dict):return []
    properties=set();patterns=[];enumerated=False;open_extension=False
    def visit(s):
        nonlocal enumerated,open_extension
        if not isinstance(s,dict):return
        if isinstance(s.get('properties'),dict):properties.update(s['properties']);enumerated=True
        patterns.extend(s.get('patternProperties',{}))
        if s.get('additionalProperties') is True or isinstance(s.get('additionalProperties'),dict):open_extension=True
        for key in ('allOf','anyOf','oneOf'):
            for child in s.get(key,[]):visit(child)
    visit(schema)
    if not enumerated or open_extension:return []
    return sorted(k for k in arguments if k not in properties and not any(re.search(p,k) for p in patterns))


def trajectory_has_undeclared_arguments(messages, tools):
    schemas={t['function']['name']:t['function']['parameters'] for t in tools}
    for message in messages:
        for call in message.get('tool_calls',[]):
            fn=call['function']
            if undeclared_arguments(fn['arguments'],schemas[fn['name']]):return True
    return False
