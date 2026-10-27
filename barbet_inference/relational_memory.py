"""Bounded relational retrieval over explicitly supplied immutable facts.

All callbacks operate on data registered by the application. Retrieved text
cannot add a tool or grant access. A missing fact means absent from this finite
snapshot, never false in the real world. This module does not execute code.
"""
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import json

from barbet_inference.memory import canonical, Verification


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str
    relation: str
    object: str

    def __post_init__(self):
        if not all(isinstance(v, str) and 0 < len(v) <= 256 for v in asdict(self).values()):
            raise ValueError('fact fields must be bounded nonempty strings')


class QueryBudgetExceeded(ValueError):
    pass


class RelationalMemory:
    """Read-only exact lookup and joins; overflow is explicit, never truncated."""
    def __init__(self, facts, *, max_facts=128, max_result=32):
        if type(max_facts) is not int or not 1 <= max_facts <= 1024:
            raise ValueError('invalid snapshot bound')
        if type(max_result) is not int or not 1 <= max_result <= 128:
            raise ValueError('invalid result bound')
        facts = tuple(facts)
        if len(facts) > max_facts or not all(isinstance(f, Fact) for f in facts):
            raise ValueError('invalid bounded fact snapshot')
        if len({f.id for f in facts}) != len(facts):
            raise ValueError('duplicate evidence identifier')
        if len({(f.subject, f.relation, f.object) for f in facts}) != len(facts):
            raise ValueError('duplicate literal relation')
        self.facts = tuple(sorted(facts, key=lambda f: f.id)); self.max_result = max_result
        self.snapshot_sha256 = hashlib.sha256(canonical([asdict(f) for f in self.facts]).encode()).hexdigest()
        self._out = defaultdict(list); self._in = defaultdict(list)
        for fact in self.facts:
            self._out[(fact.subject, fact.relation)].append(fact)
            self._in[(fact.object, fact.relation)].append(fact)

    def lookup(self, entity_ids, relation_id, direction='out'):
        if (not isinstance(entity_ids, list) or not entity_ids or len(entity_ids) > self.max_result
                or not all(isinstance(e, str) and e for e in entity_ids)
                or len(set(entity_ids)) != len(entity_ids)):
            raise ValueError('lookup needs bounded distinct entity IDs')
        if not isinstance(relation_id, str) or not relation_id or direction not in {'out', 'in'}:
            raise ValueError('invalid relation or direction')
        index = self._out if direction == 'out' else self._in
        facts = sorted((f for e in entity_ids for f in index.get((e, relation_id), [])), key=lambda f: f.id)
        endpoints = sorted({f.object if direction == 'out' else f.subject for f in facts})
        if len(facts) > 128 or len(endpoints) > self.max_result:
            raise QueryBudgetExceeded('complete relation result exceeds budget; no partial result returned')
        return {'snapshot_sha256': self.snapshot_sha256, 'scope': 'supplied_records_only',
                'complete': True, 'facts': [asdict(f) for f in facts], 'entity_ids': endpoints}

    @staticmethod
    def validate_query(query):
        if not isinstance(query, dict) or set(query) != {'operation', 'branches'}:
            raise ValueError('invalid query fields')
        operation = query['operation']; branches = query['branches']
        if operation not in {'project', 'count', 'union', 'intersection', 'difference'}:
            raise ValueError('unsupported operation')
        required = 1 if operation in {'project', 'count'} else 2
        if not isinstance(branches, list) or len(branches) != required:
            raise ValueError('wrong number of query branches')
        for branch in branches:
            if not isinstance(branch, dict) or set(branch) != {'start', 'path'}:
                raise ValueError('invalid branch fields')
            if not isinstance(branch['start'], str) or not branch['start']:
                raise ValueError('invalid start entity')
            if not isinstance(branch['path'], list) or not 1 <= len(branch['path']) <= 3:
                raise ValueError('path must have one to three hops')
            for step in branch['path']:
                if (not isinstance(step, dict) or set(step) != {'relation', 'direction'}
                        or not isinstance(step['relation'], str) or not step['relation']
                        or step['direction'] not in {'out', 'in'}):
                    raise ValueError('invalid path step')

    def evaluate(self, query):
        self.validate_query(query); branches = []; trace = []; evidence = set()
        for branch_index, branch in enumerate(query['branches']):
            ids = [branch['start']]
            for step_index, step in enumerate(branch['path']):
                inputs = ids
                if ids:
                    found = self.lookup(ids, step['relation'], step['direction']); ids = found['entity_ids']
                    step_evidence = [f['id'] for f in found['facts']]
                else:
                    ids = []; step_evidence = []
                evidence.update(step_evidence)
                trace.append({'branch': branch_index, 'step': step_index, 'input_ids': inputs,
                              'relation': step['relation'], 'direction': step['direction'],
                              'output_ids': ids, 'evidence_ids': step_evidence})
            branches.append(set(ids))
        op = query['operation']
        if op in {'project', 'count'}: result = branches[0]
        elif op == 'union': result = branches[0] | branches[1]
        elif op == 'intersection': result = branches[0] & branches[1]
        else: result = branches[0] - branches[1]
        if len(result) > self.max_result: raise QueryBudgetExceeded('combined result exceeds budget')
        target = {'count': len(result)} if op == 'count' else {'answer_ids': sorted(result)}
        target['evidence_ids'] = sorted(evidence)
        return {'target': target, 'trace': trace, 'snapshot_sha256': self.snapshot_sha256,
                'scope': 'exact set query over supplied records; not current world truth'}

    def verify(self, query, candidate):
        digest = hashlib.sha256(candidate.encode()).hexdigest()
        expected = self.evaluate(query)['target']
        try:
            def unique_keys(pairs):
                result = {}
                for k,v in pairs:
                    if k in result: raise ValueError('duplicate answer key')
                    result[k] = v
                return result
            decoded = json.loads(candidate, object_pairs_hook=unique_keys)
            # Canonical equality avoids Python's True == 1 ambiguity.
            passed = canonical(decoded) == canonical(expected)
        except (ValueError, TypeError):
            passed = False
        return Verification(passed, 'Exact supplied-record query and evidence IDs match.' if passed else
            'The answer or evidence IDs do not match the complete supplied-record query. Recheck entity, direction, path and set operation.',
            digest, 'exact supplied-record relation query only; no external factual validation')

    def lookup_tool(self):
        return {'type': 'function', 'function': {'name': 'relation_lookup',
            'description': 'Look up one relation from entity IDs in the registered finite memory snapshot. Returns all matching facts and endpoint IDs within a fixed budget; overflow is an error, never silent truncation. Missing records are not claims about the world.',
            'parameters': {'type': 'object', 'properties': {
                'entity_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': self.max_result, 'uniqueItems': True},
                'relation_id': {'type': 'string'}, 'direction': {'type': 'string', 'enum': ['out', 'in']}},
                'required': ['entity_ids', 'relation_id', 'direction'], 'additionalProperties': False}}}
