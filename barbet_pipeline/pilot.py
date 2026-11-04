"""Compile a bounded, split-isolated semantic pilot from reviewed question frames."""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import re

from .common import canonical,digest,read_json,save_new,sha_file
from .factory import make_root,wording_contract,build_trajectory,public_world,skill_library,KINDS
from .wording import validate_frame
from .export import learner_row,write_shard,load_tokenizer
from .diagnostics import oracle_conditions


def collect_frames(wording):
    wording=Path(wording)
    paths=sorted((wording/'batches').glob('*/decisions.json'))+sorted((wording/'frame-batches').glob('*/decisions.json'))
    if not paths:raise ValueError('No reviewed wording frames')
    pools=defaultdict(list);owners={};held=[];reviewed=[];rejected_roots=set()
    for path in paths:
        batch=read_json(path)
        for decision,desc in zip(batch['decisions'],batch['roots']):
            root=make_root(desc['split'],desc['ordinal']);contract=wording_contract(root)
            frame=decision.get('frame','');reason=None
            if not decision['accepted']:reason='first_teacher_hold'
            else:
                validate_frame(frame,contract)
                if re.search(r'該怎麼做|如何操作|how (?:do I|to)\b|explain (?:how|the steps)',frame,re.I):
                    reason='method_request_not_matching_execute_or_answer_contract'
                ident=digest(' '.join(frame.casefold().split()))
                if ident in owners and owners[ident]!=desc['split']:
                    reason='cross_split_surface_template_duplicate'
                if ident in owners and owners[ident]==desc['split']:
                    reason='duplicate_frame_not_independent_wording'
            if reason:
                held.append({'root_task_id':root['root_task_id'],'split':desc['split'],'ordinal':desc['ordinal'],
                             'reason':reason,'decision_sha256':sha_file(path)})
                rejected_roots.add(root['root_task_id']);continue
            owners[ident]=desc['split']
            accepted={'frame':frame,'surface_template_id':ident,'split':desc['split'],
                      'semantic_family_id':root['semantic_family_id'],
                      'decision_sha256':sha_file(path),'teacher_first_accept':True,
                      'wording_scope':'reviewed_frame_reused_across_procedural_instances'}
            pools[(desc['split'],root['semantic_family_id'])].append(accepted);reviewed.append(accepted)
    return pools,reviewed,held,rejected_roots


def compile_pilot(wording,output,private_output,tokenizer_path,targets=None):
    output,private_output=Path(output),Path(private_output)
    targets=targets or {'train':2000,'dev':200,'calibration':200,'sealed_test':200}
    if (output/'summary.json').exists():raise FileExistsError('Pilot already sealed')
    pools,frames,holds,excluded=collect_frames(wording)
    missing=[(s,k) for s in targets for k in KINDS if not pools.get((s,k))]
    if missing:raise ValueError('Unreviewed frame coverage: '+repr(missing))
    save_new(output/'frame-admission.json',{'accepted_frame_count':len(frames),
        'by_split':dict(Counter(f['split'] for f in frames)),'holds':holds,
        'template_ids':[{k:f[k] for k in ['surface_template_id','split','semantic_family_id','decision_sha256']} for f in frames],
        'independent_natural_source_count':0})
    save_new(private_output/'reviewed-frames.json',frames)
    tokenizer=load_tokenizer(tokenizer_path)
    counts=Counter();files=[];private_files=[];all_roots=set();groups={};a=[];b=[];oracle_files=[];limits=[]
    library=skill_library()
    code={p.name:sha_file(p) for p in Path(__file__).parent.glob('*.py')}
    save_new(output/'build-plan.json',{'targets':targets,'code':code,'tokenizer_sha256':sha_file(tokenizer_path),
        'frames_sha256':sha_file(private_output/'reviewed-frames.json'),
        'public_splits':['train','dev'],'private_splits':['calibration','sealed_test'],
        'program_composition_transfer':'NOT_CREATED_atomic_graphs_intentionally_shared',
        'schema_transfer':'NOT_CERTIFIED_in_atomic_pilot','new_natural_source_transfer':'NOT_RUN'})
    for split,target in targets.items():
        rows=[];tokens=[];accepted=0;ordinal=0;shard=0
        root_receipts=[]
        while accepted<target and ordinal<target+2000:
            root=make_root(split,ordinal);ordinal+=1
            if root['root_task_id'] in excluded:continue
            pool=pools.get((split,root['semantic_family_id']),[])
            if not pool:
                if root['semantic_family_id'] not in limits:limits.append(root['semantic_family_id'])
                continue
            frame=pool[(ordinal//7)%len(pool)]
            if root['root_task_id'] in all_roots:raise ValueError('Root crossed splits')
            all_roots.add(root['root_task_id']);groups[root['source_group_id']]=split
            built=[]
            for variant in ['original','counterfactual','invariance']:
                trajectory=build_trajectory(root,variant,frame['frame'])
                spec=root['variants'][variant]
                public,runtime=public_world(spec,frame['frame'])
                env={'tables':runtime.tables,'evidence':runtime.evidence,'skills':runtime.skills,
                     'state_ledger':{'revision':runtime.ledger.revision,'operations':runtime.ledger.operations},
                     'scope':'public_tool_state_not_private_oracle; only retrieve through registered tools'}
                graph=[r['tool'] for r in trajectory['execution_receipts']]+['answer']
                metadata={'record_id':digest([root['root_task_id'],variant,'semantic-v3']),
                    'root_task_id':root['root_task_id'],'source_group_id':root['source_group_id'],
                    'semantic_family_id':root['semantic_family_id'],'program_graph_id':digest(graph),
                    'surface_template_id':frame['surface_template_id'],'counterfactual_group_id':root['root_task_id'],
                    'variant':variant,'split':split,'source':'OpenFormosa/authored-semantic-v3','license':'apache-2.0',
                    'verification_scope':'independent_symbolic_oracle_and_actual_deterministic_execution_plus_reviewed_question_frame',
                    'environment_json':canonical(env)}
                refs={'origin':root['origin'],'independent_natural_source':False,
                    'question_frame_sha256':digest(frame['frame']),'question_review_sha256':frame['decision_sha256'],
                    'pair_proof':root['pair_proof'],'factory_sha256':code['factory.py'],
                    'oracle_sha256':code['oracle.py'],'executor_sha256':code['executors.py'],
                    'skills':[{'id':s['id'],'version':s['version'],'view_sha256':digest(s['text']),
                        'source_sha256':library.get(s['id'])['source_sha256'] if not s['id'].startswith('priority_') else None}
                         for s in public['skills']]}
                r,t=learner_row(tokenizer=tokenizer,metadata=metadata,public=public,
                    messages=trajectory['messages'],receipts=trajectory['execution_receipts'],source_refs=refs)
                if root['private_canary'] in canonical(r):raise ValueError('Oracle canary in public learner row')
                built.append((r,t))
            # Admit the entire matched group together, never a surviving half-pair.
            for r,t in built:
                rows.append(r);tokens.append(t);counts['rows:'+split]+=1
                counts['family:'+split+':'+r['semantic_family_id']]+=1
                counts['input_tokens:'+split]+=r['pangolin_tokens'];counts['supervised_tokens:'+split]+=r['pangolin_supervised_tokens']
                if split=='train':
                    entry={k:r[k] for k in ['record_id','root_task_id','semantic_family_id','pangolin_tokens','pangolin_supervised_tokens','variant']}
                    b.append(entry)
                    if r['variant']=='original':a.append(entry)
            root_receipts.append(root);accepted+=1;counts['roots:'+split]+=1
            if len(rows)>=600:
                destination=output if split in {'train','dev'} else private_output/'learner'
                written=write_shard(destination,f'{split}-{shard:04d}',rows,tokens)
                (files if split in {'train','dev'} else private_files).extend(written)
                rows=[];tokens=[];shard+=1
        if rows:
            destination=output if split in {'train','dev'} else private_output/'learner'
            written=write_shard(destination,f'{split}-{shard:04d}',rows,tokens)
            (files if split in {'train','dev'} else private_files).extend(written)
        oracle_files.append(save_new(private_output/(split+'-private-roots.json'),root_receipts))
        if root_receipts:
            first=root_receipts[0];frame=pools[(split,first['semantic_family_id'])][0]['frame']
            save_new(private_output/(split+'-oracle-diagnostic-fixture.json'),oracle_conditions(first,'original',frame))
        if accepted<target:raise ValueError(f'Missing reviewed frame coverage for {split}: accepted {accepted}/{target}; no silent substitution')
    for label,entries in [('A_ordinary',a),('B_semantic_contrasts',b)]:
        save_new(output/(label+'-manifest.json'),{'records':entries,
            'supervised_tokens_one_pass':sum(r['pangolin_supervised_tokens'] for r in entries),
            'status':'DATA_VIEW_ONLY_NOT_A_MATCHED_TRAINING_RUN',
            'requires_same_parent_optimizer_general_pool_and_matched_supervised_budget':True,
            'max_root_exposures_proposed':3,'automatic_replay_to_2M_or_8M':False})
    result={'counts':dict(counts),'accepted_reviewed_frames':len(frames),'held_frames':len(holds),
            'files':files,'private_files_count':len(private_files),'private_roots_hashes':[{'sha256':x['sha256'],'bytes':x['bytes']} for x in oracle_files],
            'root_and_source_groups_disjoint':True,'surface_templates_disjoint':True,
            'program_graphs_disjoint':False,'program_graph_scope':'shared atomic operations; no unseen-composition claim',
            'independent_natural_sources':0,'new_schema_transfer':'NOT_TESTED','capability_evaluations':'NOT_RUN',
            'training_ready':False,'training_blockers':['training parent not selected','Megatron parity not established',
                'general retention mix incomplete','matched A/B exposure not executed','unseen composition and natural transfer tests pending']}
    save_new(output/'summary.json',result)
    save_new(private_output/'files.json',private_files)
    return result
