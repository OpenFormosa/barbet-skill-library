"""One deterministic, source-grounded relation task per eligible anchor entity.

This is procedural data derivation with an exact finite-record verifier, not
teacher prose generation. No alternate answer or memory-position views count
as additional tasks. Names and facts come only from the pinned training graph.
"""
from functools import lru_cache
import json
import sqlite3

import numpy as np

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.relational_memory import Fact, RelationalMemory, QueryBudgetExceeded
from barbet_inference.sft import normalize_messages, prepare_tools, augment_system, Rejected
from scripts import build_memory_sft_v2 as base
from scripts.inventory_wikidata5m_primary_v2 import STATE as INVENTORY, PRIOR

SOURCE = 'Wikidata5M/original-inductive-train'
URL = 'https://deepgraphlearning.github.io/project/wikidata5m'
SKILLS = ['grounded-evidence-answer']


class SourceGraph:
    def __init__(self):
        self.triples = np.load(PRIOR / 'train-triples.npy', mmap_mode='r')
        self.heads = np.load(INVENTORY / 'train-usable-heads.npy', mmap_mode='r')
        self.offsets = np.load(INVENTORY / 'train-usable-head-offsets.npy', mmap_mode='r')
        self.order = np.load(INVENTORY / 'train-usable-head-order.npy', mmap_mode='r')
        self.db = sqlite3.connect(f'file:{PRIOR / "aliases.sqlite"}?mode=ro', uri=True)
        self.relations = dict(self.db.execute('SELECT id,label FROM relations'))

    @lru_cache(maxsize=65536)
    def label(self, entity):
        found = self.db.execute('SELECT label FROM entities WHERE id=?', (int(entity),)).fetchone()
        if not found: raise Rejected('missing_source_entity_label')
        return found[0]

    def outgoing(self, entity, maximum):
        pos = int(np.searchsorted(self.heads, entity))
        if pos == len(self.heads) or self.heads[pos] != entity: return []
        start, stop = map(int, self.offsets[pos:pos+2])
        if stop-start > maximum: raise Rejected('complete_adjacency_above_task_budget')
        return [int(v) for v in self.order[start:stop]]

    def facts(self, indexes):
        return [Fact('train:'+str(i+1), 'Q'+str(int(self.triples[i,0])),
                     'P'+str(int(self.triples[i,1])), 'Q'+str(int(self.triples[i,2])))
                for i in sorted(set(indexes))]

    def task(self, anchor):
        anchor = int(anchor); first = self.outgoing(anchor, 32)
        if not first: raise Rejected('anchor_without_bounded_outgoing_facts')
        groups = {}
        for i in first: groups.setdefault(int(self.triples[i,1]), []).append(i)
        routes = []
        # Prefer paths with informative source relations; P31/P279 are still
        # valid facts, with their concentration reported separately.
        for relation, indices in groups.items():
            context = set(first)
            try:
                for tail in {int(self.triples[i,2]) for i in indices}:
                    context.update(self.outgoing(tail, 24))
            except Rejected:
                continue
            if len(context) > 96: continue
            memory = RelationalMemory(self.facts(context))
            second_relations = {int(self.triples[i,1]) for i in context-set(first)}
            # Self/cycle edges may legitimately reuse first facts; include them
            # only if they yield a nontrivial complete answer.
            for second in sorted(second_relations):
                branch = {'start': 'Q'+str(anchor), 'path': [
                    {'relation':'P'+str(relation),'direction':'out'},
                    {'relation':'P'+str(second),'direction':'out'}]}
                query = {'operation':'project','branches':[branch]}
                try: result = memory.evaluate(query)['target']['answer_ids']
                except QueryBudgetExceeded: continue
                if not result or result == ['Q'+str(anchor)]: continue
                priority = (int(relation in {31,279}) + int(second in {31,279}),
                            base.sha(f'graph-route-v1:{anchor}:{relation}:{second}'))
                routes.append((priority, branch, context, set(result)))
        routes.sort(key=lambda v:v[0])
        if routes:
            _, branch, context, endpoints = routes[0]; branches = [branch]
            bucket = int(base.sha('graph-operation-v1:'+str(anchor))[:8],16) % 5
            operation = 'count' if bucket == 1 else 'project'
            if bucket >= 2:
                for _, other, other_context, other_endpoints in routes[1:]:
                    if other['path'][-1]['relation'] != branch['path'][-1]['relation']: continue
                    if not endpoints & other_endpoints or endpoints == other_endpoints: continue
                    combined = context | other_context
                    if len(combined) > 128: continue
                    op = {2:'union',3:'intersection',4:'difference'}[bucket]
                    if op == 'difference' and not endpoints-other_endpoints: continue
                    trial = {'operation':op,'branches':[branch,other]}
                    try: RelationalMemory(self.facts(combined)).evaluate(trial)
                    except QueryBudgetExceeded: continue
                    branches = [branch,other]; context = combined; operation = op; break
        else:
            relation = min(groups, key=lambda r:(r in {31,279}, base.sha(f'graph-one-hop-v1:{anchor}:{r}')))
            branches = [{'start':'Q'+str(anchor), 'path':[{'relation':'P'+str(relation),'direction':'out'}]}]
            context = set(first); operation = 'count' if int(base.sha('graph-operation-v1:'+str(anchor))[:8],16)%5 == 1 else 'project'
        query = {'operation':operation,'branches':branches}; facts = self.facts(context)
        memory = RelationalMemory(facts); result = memory.evaluate(query)
        if not memory.verify(query,canonical(result['target'])).passed:
            raise ValueError('derived target failed actual exact verifier')
        return query, memory, result, sorted(context)

    def describe_branch(self, branch):
        start = int(branch['start'][1:]); text = f"start at {canonical(self.label(start))} ({branch['start']})"
        for step in branch['path']:
            relation = int(step['relation'][1:]); text += f", then follow the outgoing {canonical(self.relations[relation])} ({step['relation']}) relation"
        return text

    def make_row(self, anchor, skills):
        query, memory, result, indexes = self.task(anchor); descriptions = [self.describe_branch(b) for b in query['branches']]
        op = query['operation']
        if op == 'project': task = 'Return the distinct endpoint entity IDs when you ' + descriptions[0] + '.'
        elif op == 'count': task = 'Count the distinct endpoint entity IDs when you ' + descriptions[0] + '.'
        else:
            task = 'Build set A when you ' + descriptions[0] + '. Build set B when you ' + descriptions[1] + '. '
            task += {'union':'Return all IDs in A or B, with duplicates removed.',
                     'intersection':'Return only IDs present in both A and B.',
                     'difference':'Return IDs in A that are absent from B.'}[op]
        target_key = 'count' if op == 'count' else 'answer_ids'
        question = ('Use only the supplied finite relation records, not outside knowledge or current-world assumptions. '
            'Follow directed relations exactly; repeated endpoints count once. '+task+
            f' Return one JSON object with {target_key} and evidence_ids. '+
            ('count is an integer. ' if op == 'count' else 'answer_ids is a lexicographically sorted array of entity IDs. ')+
            'evidence_ids must be the sorted union of all matching record IDs traversed at every step of every branch, '
            'including traversed records whose branches end without a further match. Do not include unrelated records. '
            'An empty result means no match in these supplied records; it does not mean false in the world.')
        qh = base.sha(base.norm(question)); mode = int(base.sha('graph-memory-mode-v1:'+str(anchor))[:8],16) % 3
        target = canonical(result['target']); raw_tools = []; calls = []
        if mode == 2:
            raw_tools = [memory.lookup_tool()]
            messages = [{'role':'user','content':question+' The records are available through relation_lookup; read them before answering.'}]
            for step in result['trace']:
                if not step['input_ids']: continue
                args = {'entity_ids':step['input_ids'],'relation_id':step['relation'],'direction':step['direction']}
                observation = memory.lookup(**args); call_id = 'relation_'+str(len(calls))
                calls.append({'arguments':args,'observation_sha256':base.sha(canonical(observation))})
                messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function',
                    'function':{'name':'relation_lookup','arguments':args}}]},
                    {'role':'tool','tool_call_id':call_id,'name':'relation_lookup','content':canonical(observation)}])
            placement = 'chronological'; items = []
        else:
            items = []
            for fact in memory.facts:
                text = canonical({'subject':fact.subject,'subject_label':self.label(int(fact.subject[1:])),
                    'relation':fact.relation,'relation_label':self.relations[int(fact.relation[1:])],
                    'object':fact.object,'object_label':self.label(int(fact.object[1:]))})
                items.append(MemoryItem(fact.id,text,URL,observation_origin='pinned_public_structured_record'))
            placement = 'front' if mode == 0 else 'back'
            messages = [{'role':'user','content':compose_memory_prompt(question,items,placement=placement)}]
        messages.append({'role':'assistant','content':target})
        tools, validators = prepare_tools(raw_tools); messages = normalize_messages(messages,tools,validators)
        messages = augment_system(messages,skills[0]); rid = base.sha(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools}))
        family = base.sha('wikidata5m-inductive-anchor-v1:Q'+str(anchor))
        return {'record_id':rid,'query_hash':qh,'family_id':family,'source_family_id':family,
            'template_family_id':family,'source_task_key':'Q'+str(anchor),'split':'unassigned',
            'domain':'tool_use' if mode == 2 else 'reading','source':SOURCE,'upstream_source':SOURCE,'license':'cc0-1.0',
            'messages':messages,'tools':tools,'memory':{'documents':[item.__dict__ for item in items]},
            'skills':skills[1],'skill_ids':SKILLS,'verification_tier':'exact_query_over_pinned_public_structured_records',
            'observation_origin':'local_deterministic_relation_lookup' if mode == 2 else 'pinned_public_structured_record',
            'memory_placement':placement,'actual_retrieval':mode == 2,'recorded_error_followup':False,
            'assistant_turns':sum(m['role']=='assistant' and m['loss'] for m in messages),
            'tool_observations':len(calls),'locally_executed_source_program':False,
            'source_refs':{'attribution_url':URL,'license':'cc0-1.0','source_anchor_id':'Q'+str(anchor),
                'original_training_source_line_numbers':[i+1 for i in indexes],
                'graph_member_sha256':'0ec75c242d5193604bf8055a64fce3dedbf1d9d13ab555ed727b3252af4178ba',
                'query':query,'query_schema_id':'finite-relations-'+op+'-'+'-'.join(str(len(b['path'])) for b in query['branches']),
                'snapshot_sha256':memory.snapshot_sha256,'exact_computation_trace':result['trace'],
                'actual_tool_calls':calls,'one_task_per_anchor':True,'memory_placement_views_not_duplicated':True,
                'new_semantic_teacher_generation':False,'target_origin':'deterministic exact finite-record set query',
                'world_truth_independently_verified':False,'source_programs_executed':False}}
