"""Additional views of existing roots with tool IDs removed from the question.

The model must resolve a visible display name and field meaning to opaque IDs.
The host still only dereferences the selected value. These are not new roots.
"""
from __future__ import annotations
from collections import Counter
import copy
from pathlib import Path
import random

from .common import canonical,digest,read_json,save_new,sha_file
from .factory import library_card
from .executors import Executor,tool_schema
from .oracle import verify_execution
from .visibility import actor_messages
from .export import learner_row,write_shard,load_tokenizer


def faded_task(root,frame):
    original=root['variants']['original']
    if original['kind']!='binding':raise ValueError('Binding root required')
    spec=copy.deepcopy(original)
    rng=random.Random(root['root_task_id']+'faded')
    aliases=['青杉','白石','晴川','向野','青禾','星港'];rng.shuffle(aliases)
    meanings={'source_id':'來源識別碼','destination_id':'目的地識別碼','optional':'選填資料'}
    fields={k:'field_'+digest([root['root_task_id'],k])[:8] for k in meanings}
    rows=[]
    for old,name in zip(original['rows'],aliases):
        rows.append({'id':old['id'],'display_name':name,
                     **{fields[k]:old[k] for k in meanings}})
    spec['rows']=rows;spec['field']=fields[original['field']]
    selected=next(r for r in rows if r['id']==spec['row_id'])
    task=frame.format(table=spec['table_id'],entity=selected['display_name'],attribute=meanings[original['field']])
    if spec['row_id'] in task or spec['field'] in task:
        raise ValueError('Faded question leaks target tool IDs')
    if sum(r['display_name']==selected['display_name'] for r in rows)!=1:
        raise ValueError('Ambiguous display name')
    public={'task':task,'context':[{'id':spec['table_id'],'title':'Fictional named records','revision':'1','text':canonical(rows)},
        {'id':'field-guide','title':'Current column meanings','revision':'1',
         'text':canonical({meaning:fields[k] for k,meaning in meanings.items()})}],
        'skills':[library_card('literal-typed-tool-arguments')],
        'tools':[tool_schema('resolve_value')],'memory_placement':spec['memory_placement']}
    executor=Executor(tables={spec['table_id']:rows})
    messages=actor_messages(public,forbidden_canaries=[root['private_canary']])
    args={k:spec[k] for k in ['table_id','row_id','field']}
    messages.append({'role':'assistant','content':'','loss':True,'tool_calls':[{'id':'binding_0','type':'function',
        'function':{'name':'resolve_value','arguments':args}}]})
    output=executor.call('resolve_value',args)
    if not verify_execution(spec,output):raise ValueError('Faded binding execution disagrees with oracle')
    messages.extend([{'role':'tool','name':'resolve_value','tool_call_id':'binding_0','content':canonical(output),'loss':False},
                     {'role':'assistant','content':canonical(output['result']['value']),'loss':True}])
    return spec,public,messages,executor


def compile_faded(wording,base_private,output,private_output,tokenizer_path):
    wording,base_private,output,private_output=map(Path,[wording,base_private,output,private_output])
    decisions=read_json(wording/'decisions.json');protocol=read_json(wording/'protocol.json')
    root_review=read_json(wording/'root-review.json')
    candidates=[(d,protocol['splits'][d['id']],sha_file(wording/'decisions.json'))
                for d in decisions if d['id'] in root_review['accepted_ids']]
    fresh_reviews=[]
    for name in ['fresh-sealed-frames','fresh-sealed-frames-v2']:
        fresh=wording/name
        if not fresh.exists():continue
        review=read_json(fresh/'root-review.json')
        candidates += [(d,'sealed_test',sha_file(fresh/'decisions.json'))
                       for d in read_json(fresh/'decisions.json') if d['id'] in review['accepted_ids']]
        fresh_reviews.append({'protocol_sha256':sha_file(fresh/'protocol.json'),
            'decisions_sha256':sha_file(fresh/'decisions.json'),
            'root_review_sha256':sha_file(fresh/'root-review.json')})
    review_identity=digest({'teacher_decisions':sha_file(wording/'decisions.json'),
        'root_review':sha_file(wording/'root-review.json'),
        'fresh_reviews':fresh_reviews})
    frames={};seen=set()
    for d,split,review_sha in candidates:
        if not d['accepted']:continue
        identity=digest(' '.join(d['frame'].casefold().split()))
        if identity in seen:continue
        seen.add(identity)
        frames.setdefault(split,[]).append({**d,'teacher_review_sha256':review_sha})
    if set(frames)!={'train','dev','calibration','sealed_test'}:
        raise ValueError('Every split requires an independent accepted faded frame')
    tokenizer=load_tokenizer(tokenizer_path);files=[];private_files=[];counts=Counter();b_entries=[]
    save_new(output/'plan.json',{'root_lineage':'same roots as primary pilot, never count as new root tasks',
        'question_frame_review_sha256':review_identity,
        'driver_sha256':sha_file(Path(__file__)),'new_independent_roots':0,'prompt_fading':'display names and meanings replace tool row IDs and opaque field codes'})
    for split,pool in frames.items():
        rows=[];tokens=[];private_specs=[]
        for root in read_json(base_private/(split+'-private-roots.json')):
            if root['semantic_family_id']!='binding':continue
            frame=pool[len(rows)%len(pool)]
            spec,public,messages,executor=faded_task(root,frame['frame'])
            metadata={'record_id':digest([root['root_task_id'],'faded_binding','semantic-v3']),
                'root_task_id':root['root_task_id'],'source_group_id':root['source_group_id'],
                'semantic_family_id':'binding','program_graph_id':digest(['resolve_value','answer']),
                'surface_template_id':digest(' '.join(frame['frame'].casefold().split())),
                'counterfactual_group_id':root['root_task_id'],'variant':'faded_binding','split':split,
                'source':'OpenFormosa/authored-semantic-v3','license':'apache-2.0',
                'verification_scope':'visible_name_and_field_guide_binding_plus_independent_typed_value_oracle',
                'environment_json':canonical({'tables':executor.tables,'evidence':{},'skills':{},'state_ledger':{'revision':0,'operations':{}},'scope':'public fictional records'})}
            refs={'parent_original_view_id':digest([root['root_task_id'],'original','semantic-v3']),
                'question_frame_review_sha256':review_identity,
                'teacher_decision_sha256':frame['teacher_review_sha256'],
                'independent_natural_source':False,'counts_as_new_root':False,
                'private_tool_ids_absent_from_question':True,'driver_sha256':sha_file(Path(__file__))}
            r,t=learner_row(tokenizer=tokenizer,metadata=metadata,public=public,messages=messages,
                           receipts=executor.receipts,source_refs=refs)
            rows.append(r);tokens.append(t);private_specs.append(spec)
            counts['rows:'+split]+=1;counts['input_tokens:'+split]+=r['pangolin_tokens']
            counts['supervised_tokens:'+split]+=r['pangolin_supervised_tokens']
            if split=='train':b_entries.append({k:r[k] for k in ['record_id','root_task_id','semantic_family_id','pangolin_tokens','pangolin_supervised_tokens','variant']})
        where=output if split in {'train','dev'} else private_output/'learner'
        written=write_shard(where,split+'-faded-binding-0000',rows,tokens)
        (files if split in {'train','dev'} else private_files).extend(written)
        save_new(private_output/(split+'-specifications.json'),private_specs)
    result={'counts':dict(counts),'new_root_tasks':0,'files':files,'private_files_count':len(private_files),
            'surface_templates_disjoint':True,'question_contains_correct_tool_row_id_or_field_code':False,
            'learnability_transfer_retention':'NOT_RUN'}
    save_new(output/'summary.json',result);save_new(output/'B-additional-views.json',b_entries)
    save_new(private_output/'files.json',private_files)
    return result
