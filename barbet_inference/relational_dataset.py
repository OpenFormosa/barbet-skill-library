"""Explicit, portable restoration of a dataset's registered relation memory.

The application must call this adapter deliberately; it does not discover or
execute tools from source text. Only the bounded, read-only lookup is returned.
"""
import json
from barbet_inference.relational_memory import Fact, RelationalMemory


def restore_relational_memory(row):
    metadata = row.get('memory')
    if metadata is None:
        metadata = json.loads(row['memory_json'])
    descriptor = metadata.get('registered_relation_snapshot')
    if (not isinstance(descriptor, dict)
            or descriptor.get('format') != 'barbet-finite-relations-v1'
            or descriptor.get('tool_name') != 'relation_lookup'
            or descriptor.get('visibility') != 'registered_tool_state; observations become model-visible only after calls'):
        raise ValueError('unsupported explicit relation-memory descriptor')
    raw = descriptor.get('facts')
    if not isinstance(raw,list) or len(raw)>128:
        raise ValueError('invalid complete relation snapshot')
    facts=[]
    for item in raw:
        if not isinstance(item,dict) or set(item)!={'id','subject','relation','object'}:
            raise ValueError('invalid fact fields')
        facts.append(Fact(**item))
    memory=RelationalMemory(facts,max_facts=128,max_result=32)
    if memory.snapshot_sha256!=descriptor.get('snapshot_sha256'):
        raise ValueError('registered relation snapshot hash differs')
    return memory


def relation_tool_callbacks(row):
    """Caller explicitly registers these read-only callbacks in its harness."""
    memory=restore_relational_memory(row)
    return {'relation_lookup':memory.lookup}
