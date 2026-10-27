"""One finite-record lexical retrieval task per distinct base concept.

The exact source URI, sense, relation polarity, attribution and row license are
retained. A source assertion is not promoted to independently verified truth.
"""
from dataclasses import asdict
from functools import lru_cache
import json
import sqlite3

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.relational_memory import Fact, RelationalMemory, QueryBudgetExceeded
from barbet_inference.sft import Rejected, prepare_tools, normalize_messages, augment_system
from scripts import build_memory_sft_v2 as base
from scripts.prepare_conceptnet_lexical_scope_v1 import STATE as INVENTORY
from scripts.inventory_conceptnet57_english_v1 import STATE as ORIGINAL

SOURCE='ConceptNet/5.7-original-lexical-english'
URL='https://github.com/commonsense/conceptnet5/wiki/Downloads'
SKILLS=['grounded-evidence-answer']


class SourceGraph:
    def __init__(self):
        self.db=sqlite3.connect(f'file:{INVENTORY / "lexical-index.sqlite"}?mode=ro',uri=True)
        self.original=sqlite3.connect(f'file:{ORIGINAL / "english-assertions.sqlite"}?mode=ro',uri=True)

    @lru_cache(maxsize=32768)
    def _outgoing(self,subject):
        # One over the largest supported complete adjacency is sufficient to
        # reject overflow. Never use a silently truncated relation snapshot.
        return tuple(Fact('cn57:'+str(n),s,r,o) for n,s,r,o in self.db.execute(
            'SELECT source_line,subject,relation,object FROM facts WHERE subject=? ORDER BY source_line LIMIT 33',(subject,)))

    def outgoing(self,subject,maximum):
        facts=self._outgoing(subject)
        if len(facts)>maximum:raise Rejected('complete_lexical_adjacency_above_budget')
        return facts

    def task(self,base_uri,subject):
        first=self.outgoing(subject,32)
        if not first:raise Rejected('anchor_without_source_facts')
        groups={}
        for fact in first:groups.setdefault(fact.relation,[]).append(fact)
        routes=[]
        for relation,facts in groups.items():
            context={f.id:f for f in first}
            try:
                for endpoint in {f.object for f in facts}:
                    context.update((f.id,f) for f in self.outgoing(endpoint,24))
            except Rejected:continue
            if len(context)>96:continue
            memory=RelationalMemory(sorted(context.values(),key=lambda f:f.id))
            for second in sorted({f.relation for f in context.values()}):
                branch={'start':subject,'path':[{'relation':relation,'direction':'out'},
                    {'relation':second,'direction':'out'}]}
                try:ends=memory.evaluate({'operation':'project','branches':[branch]})['target']['answer_ids']
                except QueryBudgetExceeded:continue
                if not ends or ends==[subject]:continue
                routes.append((base.sha(f'cn57-route-v1:{base_uri}:{relation}:{second}'),branch,context,set(ends)))
        routes.sort(key=lambda r:r[0]);bucket=int(base.sha('cn57-operation-v1:'+base_uri)[:8],16)%5
        operation='count' if bucket==1 else 'project'
        if routes:
            _,branch,context,endpoints=routes[0];branches=[branch]
            if bucket>=2:
                for _,other,other_context,other_endpoints in routes[1:]:
                    if other['path'][-1]['relation']!=branch['path'][-1]['relation']:continue
                    if not endpoints & other_endpoints or endpoints==other_endpoints:continue
                    combined={**context,**other_context}
                    if len(combined)>128:continue
                    op={2:'union',3:'intersection',4:'difference'}[bucket]
                    if op=='difference' and not endpoints-other_endpoints:continue
                    try:RelationalMemory(list(combined.values())).evaluate({'operation':op,'branches':[branch,other]})
                    except QueryBudgetExceeded:continue
                    branches=[branch,other];context=combined;operation=op;break
        else:
            relation=min(groups,key=lambda r:base.sha(f'cn57-single-v1:{base_uri}:{r}'))
            branches=[{'start':subject,'path':[{'relation':relation,'direction':'out'}]}]
            context={f.id:f for f in first}
        query={'operation':operation,'branches':branches}
        memory=RelationalMemory(sorted(context.values(),key=lambda f:f.id));result=memory.evaluate(query)
        if not memory.verify(query,canonical(result['target'])).passed:raise ValueError('exact lexical target verification failed')
        return query,memory,result

    def source_record(self,fact):
        line=int(fact.id.split(':')[1])
        found=self.original.execute('SELECT assertion_uri,subject,relation,object,metadata_json FROM facts WHERE source_line=?',(line,)).fetchone()
        if not found or tuple(found[1:4])!=(fact.subject,fact.relation,fact.object):
            raise ValueError('lexical fact differs from original source')
        metadata=json.loads(found[4])
        return {'fact_id':fact.id,'source_line':line,'assertion_uri':found[0],
            'attribution_url':'https://conceptnet.io'+found[0],'metadata':metadata}

    def make_row(self,base_uri,subject,skills):
        query,memory,result=self.task(base_uri,subject);op=query['operation']
        key='count' if op=='count' else 'answer_ids'
        question=('Query these finite ConceptNet 5.7 lexical records using the exact URI strings in this specification: '+canonical(query)+'. '
            'Each branch starts from its start URI and follows its path in order. direction out means subject to object. '
            'Keep sense and part-of-speech suffixes; do not replace a relation, invert an edge or infer additional facts. '
            'project returns distinct endpoints of the first branch; count counts those endpoints; union, intersection and difference '
            'mean A union B, A intersection B and A minus B for the first and second branches. '
            f'Return one JSON object with {key} and evidence_ids. '+
            ('count is an integer. ' if op=='count' else 'answer_ids is a lexicographically sorted array of complete URI strings. ')+
            'evidence_ids is the sorted union of every matching record ID traversed at every step of every branch, '
            'including traversed records whose next step has no match. Exclude unrelated records. '
            'An absent match means absent from this provided snapshot, not false in the world. '
            'This is a query over attributed source assertions, not a request to assert that every lexical relation is universally true.')
        mode=int(base.sha('cn57-placement-v1:'+base_uri)[:8],16)%3;raw_tools=[];items=[];calls=[]
        if mode==2:
            raw_tools=[memory.lookup_tool()]
            messages=[{'role':'user','content':question+' Read the registered records with relation_lookup before answering.'}]
            for step in result['trace']:
                if not step['input_ids']:continue
                args={'entity_ids':step['input_ids'],'relation_id':step['relation'],'direction':step['direction']}
                observation=memory.lookup(**args);call_id='lexical_'+str(len(calls))
                calls.append({'arguments':args,'observation_sha256':base.sha(canonical(observation))})
                messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function',
                    'function':{'name':'relation_lookup','arguments':args}}]},
                    {'role':'tool','tool_call_id':call_id,'name':'relation_lookup','content':canonical(observation)}])
            placement='chronological'
        else:
            for f in memory.facts:
                items.append(MemoryItem(f.id,canonical({'subject':f.subject,'relation':f.relation,'object':f.object}),
                    URL,observation_origin='pinned_public_lexical_record'))
            placement='front' if mode==0 else 'back'
            messages=[{'role':'user','content':compose_memory_prompt(question,items,placement=placement)}]
        messages.append({'role':'assistant','content':canonical(result['target'])})
        tools,validators=prepare_tools(raw_tools);messages=augment_system(normalize_messages(messages,tools,validators),skills[0])
        source_records=[self.source_record(f) for f in memory.facts]
        licenses={r['metadata']['license'] for r in source_records}
        if not licenses <= {'cc:by/4.0','cc:by-sa/4.0'}:raise Rejected('unsupported_lexical_license')
        license_='cc-by-sa-4.0' if 'cc:by-sa/4.0' in licenses else 'cc-by-4.0'
        family=base.sha('cn57-base-concept-v1:'+base_uri)
        return {'record_id':base.sha(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools})),
            'query_hash':base.sha(base.norm(question)),'family_id':family,'source_family_id':family,'template_family_id':family,
            'source_task_key':base_uri,'split':'unassigned','domain':'tool_use' if mode==2 else 'reading',
            'source':SOURCE,'upstream_source':SOURCE,'license':license_,'messages':messages,'tools':tools,
            'memory':{'documents':[item.__dict__ for item in items],'registered_relation_snapshot':{
                'format':'barbet-finite-relations-v1','tool_name':'relation_lookup',
                'visibility':'registered_tool_state; observations become model-visible only after calls',
                'snapshot_sha256':memory.snapshot_sha256,'facts':[asdict(f) for f in memory.facts]}},
            'skills':skills[1],'skill_ids':SKILLS,'verification_tier':'exact_query_over_pinned_public_lexical_records',
            'observation_origin':'local_deterministic_relation_lookup' if mode==2 else 'pinned_public_lexical_record',
            'memory_placement':placement,'actual_retrieval':mode==2,'recorded_error_followup':False,
            'assistant_turns':sum(m['role']=='assistant' and m['loss'] for m in messages),
            'tool_observations':len(calls),'locally_executed_source_program':False,
            'source_refs':{'attribution_url':URL,'license':license_,'source_base_concept':base_uri,
                'source_anchor_uri':subject,'source_assertions':source_records,
                'query':query,'query_schema_id':'finite-lexical-'+op+'-'+'-'.join(str(len(b['path'])) for b in query['branches']),
                'snapshot_sha256':memory.snapshot_sha256,'exact_computation_trace':result['trace'],
                'actual_tool_calls':calls,'one_task_per_base_concept':True,'sense_variants_not_additional_questions':True,
                'memory_placement_views_not_duplicated':True,'new_semantic_teacher_generation':False,
                'target_origin':'deterministic exact finite-record set query','world_truth_independently_verified':False,
                'source_programs_executed':False}}
