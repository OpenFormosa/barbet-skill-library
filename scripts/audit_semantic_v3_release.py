"""Recheck exported pilot records against private specs and real public execution.

The output contains aggregate counts and artifact hashes only. Private roots,
calibration/test records and diagnostic answers never enter the publication.
"""
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq

from barbet_pipeline.common import canonical,digest,read_json,save_new,sha_file
from barbet_pipeline.executors import Executor,StateLedger
from barbet_pipeline.oracle import require_discriminating,verify_execution
from barbet_pipeline.visibility import actor_messages


def audit(state,private,output):
    state,private=Path(state),Path(private)
    roots={};owners={};frame_owners={};counts=Counter();coverage=Counter();views=defaultdict(set)
    for split in ['train','dev','calibration','sealed_test']:
        for root in read_json(private/(split+'-private-roots.json')):
            rid=root['root_task_id']
            assert rid not in roots
            roots[rid]=root
            source=root['source_group_id']
            assert source not in owners or owners[source]==split
            owners[source]=split
            require_discriminating(*(root['variants'][v] for v in ['original','counterfactual','invariance']))
            counts['roots:'+split]+=1
            spec=root['variants']['original'];kind=spec['kind']
            if kind=='comparison':
                coverage['comparison:'+spec['operation']]+=1
                assert any(row[spec['field']]==spec['threshold'] for row in spec['rows'])
            if kind=='ordering':
                coverage['ordering:'+spec['operation']]+=1
                coverage['ordering_with_ties']+=len({r[spec['field']] for r in spec['rows']})<len(spec['rows'])
            if kind=='evidence_state':coverage['evidence:'+spec['state']]+=1
            if kind=='operation_state':coverage['operation_status:'+spec['status']]+=1
            if kind=='rule_version':coverage['active_version:'+spec['effective_version']]+=1
    core=read_json(state/'pilot/summary.json');faded=read_json(state/'faded-binding/summary.json')
    files=core['files']+faded['files']+read_json(private/'files.json')+read_json(private/'faded-binding/files.json')
    faded_specs={}
    for split in ['train','dev','calibration','sealed_test']:
        binding_roots=[r for r in read_json(private/(split+'-private-roots.json')) if r['semantic_family_id']=='binding']
        specs=read_json(private/'faded-binding'/(split+'-specifications.json'))
        assert len(binding_roots)==len(specs)
        faded_specs.update({r['root_task_id']:s for r,s in zip(binding_roots,specs)})
    ids=set();shards=[]
    for f in files:
        if '/records/' not in f['path']:continue
        assert sha_file(f['path'])==f['sha256']
        for row in pq.read_table(f['path']).to_pylist():
            rid=row['root_task_id'];root=roots[rid];split=row['split'];variant=row['variant']
            assert split==root['split'] and row['source_group_id']==root['source_group_id']
            assert row['record_id'] not in ids
            ids.add(row['record_id']);views[rid].add(variant)
            frame=row['surface_template_id']
            assert frame not in frame_owners or frame_owners[frame]==split
            frame_owners[frame]=split
            assert root['private_canary'] not in canonical(row)
            public=json.loads(row['public_episode_json'])
            actor_messages(public,forbidden_canaries=[root['private_canary']])
            spec=faded_specs[rid] if variant=='faded_binding' else root['variants'][variant]
            if variant=='faded_binding':
                assert spec['row_id'] not in public['task'] and spec['field'] not in public['task']
            env=json.loads(row['environment_json']);ledger=StateLedger()
            ledger.revision=env['state_ledger']['revision'];ledger.operations=copy.deepcopy(env['state_ledger']['operations'])
            runtime=Executor(tables=env['tables'],evidence=env['evidence'],skills=env['skills'],ledger=ledger)
            messages=json.loads(row['messages_json']);receipts=json.loads(row['execution_receipts_json'])
            calls=[call for m in messages for call in m.get('tool_calls',[])]
            observations=[m for m in messages if m['role']=='tool']
            assert len(calls)==len(receipts)==len(observations)
            for call,receipt,observation in zip(calls,receipts,observations):
                name=call['function']['name'];args=call['function']['arguments']
                if isinstance(args,str):args=json.loads(args)
                assert name==receipt['tool'] and canonical(args)==canonical(receipt['arguments'])
                result=runtime.call(name,args)
                assert canonical(result)==canonical(receipt['output'])==observation['content']
                assert digest(args)==receipt['arguments_sha256'] and digest(result)==receipt['output_sha256']
                counts['replayed_tool_calls']+=1
            if spec['kind']!='evidence_state':
                assert receipts and verify_execution(spec,receipts[-1]['output'])
                counts['independent_numeric_typed_or_state_oracle_checks']+=1
            else:
                # This is source/state consistency, not an independent language-quality judge.
                answer=messages[-1]['content'];kind=spec['state']
                if kind in {'present','retrievable'}:
                    assert spec['answer'] in answer and spec['document_id'] in answer
                    evidence=canonical(public['context']) if kind=='present' else canonical(receipts[-1]['output'])
                    assert spec['answer'] in evidence
                elif kind=='unavailable':assert receipts and not receipts[-1]['output']['ok']
                else:assert not receipts
                counts['source_and_state_consistency_checks']+=1
            counts['rows:'+split]+=1
            counts['input_tokens:'+split]+=row['pangolin_tokens']
            counts['supervised_tokens:'+split]+=row['pangolin_supervised_tokens']
            coverage['memory_placement:'+public['memory_placement']]+=1
        shards.append({'sha256':f['sha256'],'bytes':f['bytes'],'public':str(private) not in f['path']})
    for rid,root in roots.items():
        expected={'original','counterfactual','invariance'}
        if root['semantic_family_id']=='binding':expected.add('faded_binding')
        assert views[rid]==expected
    result={'counts':dict(counts),'coverage':dict(coverage),'record_shards':shards,
        'root_source_surface_splits_disjoint':True,'all_root_variant_sets_complete':True,
        'all_receipts_reexecuted_from_public_state':True,'private_canaries_absent':True,
        'faded_question_does_not_supply_tool_row_or_field':True,
        'program_graphs_disjoint':False,'unseen_composition_certification':False,
        'learning_transfer_retention':'NOT_RUN','driver_sha256':sha_file(Path(__file__))}
    save_new(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='record_shards'}))


if __name__=='__main__':audit(*sys.argv[1:])
