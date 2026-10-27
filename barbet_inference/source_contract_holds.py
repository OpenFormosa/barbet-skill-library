"""Exact provenance-bound tool metadata holds; never reconstruct a schema."""
import hashlib
from .memory import canonical


def held_tool_schemas(tools, policy):
    known = {item['function_sha256']: item['id'] for item in policy['holds']}
    found = []
    for item in tools:
        function = item.get('function', item)
        digest = hashlib.sha256(canonical(function).encode()).hexdigest()
        if digest in known: found.append({'hold_id': known[digest], 'function_sha256': digest})
    return found
