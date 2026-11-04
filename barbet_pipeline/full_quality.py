"""Full-row quality checks. Reference fidelity is never semantic certification."""
from functools import lru_cache
from pathlib import Path
import json,re
import pyarrow.parquet as pq
from .common import canonical,read_json,sha_file
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.dataset_quality import terminal_completion_hold
from barbet_inference.readiness_quality import continuation_content_holds
from barbet_inference.source_contract_holds import held_tool_schemas

WORK=Path('/root/barbet-context-sft-data')

class QualityHold(ValueError):pass

def independent_graph_target(facts,query):
    outputs=[];evidence=set()
    for branch in query['branches']:
        current={branch['start']}
        for step in branch['path']:
            following=set()
            if step['direction'] not in {'out','in'}:raise QualityHold('unknown_graph_direction')
            for fact in facts:
                src,dst=(fact.subject,fact.object) if step['direction']=='out' else (fact.object,fact.subject)
                if fact.relation==step['relation'] and src in current:
                    following.add(dst);evidence.add(fact.id)
            current=following
        outputs.append(current)
    op=query['operation']
    if op in {'project','count'}:
        if len(outputs)!=1:raise QualityHold('wrong_graph_arity')
        result=outputs[0]
    else:
        if len(outputs)!=2:raise QualityHold('wrong_graph_arity')
        a,b=outputs
        if op=='union':result=a|b
        elif op=='intersection':result=a&b
        elif op=='difference':result=a-b
        else:raise QualityHold('unknown_graph_operation')
    return {('count' if op=='count' else 'answer_ids'):(len(result) if op=='count' else sorted(result)),
            'evidence_ids':sorted(evidence)}

@lru_cache(maxsize=1)
def sources():
    files=read_json(WORK/'reports/memory-harness-v2/build-source-lock.json')['files']
    return {(f['repo'],f['path']):f for f in files}

@lru_cache(maxsize=8)
def source_table(repo,file):
    item=sources()[(repo,file)]
    if sha_file(item['local'])!=item['sha256']:raise ValueError('Pinned raw source changed')
    return pq.read_table(item['local']).to_pylist(),item

@lru_cache(maxsize=64)
def ni_task(file,expected_sha):
    path=WORK/'data/raw_memory_harness_v2/scale10m-natural-instructions'/file
    if sha_file(path)!=expected_sha:raise ValueError('Pinned NI source changed')
    return read_json(path)

def source_gold(row,refs,messages):
    source=row['source'];target=messages[-1]['content']
    if source=='allenai/natural-instructions':
        task=ni_task(refs['file'],refs['file_sha256']);instance=task['Instances'][refs['row_index']]
        if instance['id']!=refs['upstream_id'] or target.strip() not in [str(x).strip() for x in instance['output']]:
            raise QualityHold('NI_target_differs_from_pinned_reference')
        memory=json.loads(row['memory_json']);documents=memory.get('documents',[]) if isinstance(memory,dict) else memory
        targets=[d for d in documents if d.get('id')=='target-input']
        if len(targets)!=1 or targets[0]['text']!=instance['input']:
            raise QualityHold('NI_target_input_memory_mismatch')
        return 'source_reference','pinned_instruction_reference_and_target_input_not_independent_semantic_judgment'
    if source in {'rajpurkar/squad_v2','hotpotqa/hotpot_qa'}:
        rows,item=source_table(source,refs['file']);raw=rows[refs['row_index']]
        if item['sha256']!=refs['file_sha256']:raise ValueError('Raw source identity mismatch')
        identity=raw.get('id',raw.get('_id'))
        if identity!=refs['upstream_id']:raise QualityHold('source_question_identity_mismatch')
        answer=json.loads(target)
        if source=='rajpurkar/squad_v2':
            expected=raw['answers']['text'][0] if raw['answers']['text'] else None
            # The source adapter's explicit unavailable marker is contractual.
            if expected is None:
                if answer.get('answer') not in {None,'INSUFFICIENT_EVIDENCE'}:raise QualityHold('unsupported_SQuAD_answer')
            elif answer.get('answer')!=expected:raise QualityHold('SQuAD_gold_mismatch')
            for value,start in zip(raw['answers']['text'],raw['answers']['answer_start']):
                if raw['context'][start:start+len(value)]!=value:raise QualityHold('source_gold_span_mismatch')
            return 'verified','source_gold_span_and_sufficiency_label_not_world_truth'
        if answer.get('answer')!=raw['answer']:raise QualityHold('Hotpot_gold_mismatch')
        context=raw['context'];support=raw['supporting_facts'];expected_ids=[]
        for title,sent in zip(support['title'],support['sent_id']):
            if title not in context['title']:raise QualityHold('Hotpot_support_document_missing')
            i=context['title'].index(title)
            if not 0<=sent<len(context['sentences'][i]):raise QualityHold('Hotpot_support_sentence_missing')
            expected_ids.append(f'd{i}s{sent}')
        if sorted(answer.get('evidence_ids',[]))!=sorted(set(expected_ids)):raise QualityHold('Hotpot_support_ids_mismatch')
        return 'verified','pinned_gold_answer_and_supporting_sentence_ids_not_world_truth'
    return None

def check_content(row,messages,tools,policy):
    refs=json.loads(row['source_refs_json']);source=row['source'];checks=[];receipts=[]
    if any('<pad>' in m['content'] for m in messages):raise QualityHold('literal_native_PAD_in_content')
    for issue in continuation_content_holds(source,row['domain'],messages,has_tools=bool(tools)):
        raise QualityHold(issue['reason'])
    issue=terminal_completion_hold({**row,'messages':messages})
    if issue:raise QualityHold(issue)
    if held_tool_schemas(tools,policy):raise QualityHold('known_inconsistent_source_tool_schema')
    if source.startswith(('Wikidata5M/','ConceptNet/')):
        memory=restore_relational_memory(row);expected=independent_graph_target(memory.facts,refs['query'])
        final=next(m for m in reversed(messages) if m['role']=='assistant' and m['loss'])
        if canonical(json.loads(final['content']))!=canonical(expected):raise QualityHold('independent_graph_target_mismatch')
        pending={}
        for m in messages:
            for call in m.get('tool_calls',[]):
                if call['function']['name']!='relation_lookup':raise QualityHold('unexpected_graph_tool')
                args=call['function']['arguments'];result=memory.lookup(**args);pending[call['id']]=result
                receipts.append({'tool':'relation_lookup','arguments':args,'result':result,'origin':'local_deterministic_reexecution'})
            if m['role']=='tool':
                if canonical(json.loads(m['content']))!=canonical(pending.pop(m['tool_call_id'])):
                    raise QualityHold('graph_observation_mismatch')
        if pending:raise QualityHold('missing_graph_tool_observation')
        return 'verified','independent_finite_graph_query_and_tool_reexecution',receipts
    gold=source_gold(row,refs,messages)
    if gold:return *gold,receipts
    if source=='OpenFormosa/taiwan-civic-v1':
        memory=json.loads(row['memory_json']);docs=memory.get('documents',[]) if isinstance(memory,dict) else memory
        target=messages[-1]['content'];tier=refs['verification_tier']
        if tier=='exact_source_article_extract':
            prefix=f'《{refs["law_name"]}》{refs["article"]}（所附版本原文）：\n'
            if not target.startswith(prefix) or not any(d['text'].endswith('條文原文：\n'+target[len(prefix):]) and d['source']==refs['source_url'] for d in docs):
                raise QualityHold('civic_source_quote_mismatch')
            return 'verified','exact_attributed_source_extract_not_legal_applicability',receipts
        if tier=='exact_attributed_historical_official_quote':
            quote=re.search('「(.*?)」',target)
            if not quote or quote[1] not in docs[0]['text'] or '外交部' not in target:raise QualityHold('official_quote_mismatch')
            return 'verified','exact_attributed_historical_quote',receipts
        if tier=='authored_normative_case_not_external_fact':
            if docs[0]['source']!='authored:OpenFormosa/taiwan-civic-policy-v1':raise QualityHold('normative_source_mislabelled')
            return 'source_reference','owner_normative_case_prior_values_admission_retained',receipts
        raise QualityHold('unknown_civic_tier')
    return 'reference_only','all_turns_structure_and_source_lineage_checked_semantic_answer_not_independently_verified',receipts
