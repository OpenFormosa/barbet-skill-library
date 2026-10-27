"""First non-replayed lexical source diagnosis with exact independent targets."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts import package_memory_sft_v2 as package
from scripts import audit_memory_sft_v2 as reviewer
from scripts import audit_memory_sft_math_v2 as transport
from scripts import conceptnet_relation_adapter_v1 as adapter
from scripts.diagnose_wikidata_relations_v1 import independent_sql_target

STATE=adapter.INVENTORY.parent/'first-lexical-diagnosis-v1'


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE/'pilot-summary.json').exists():raise ValueError('first lexical diagnosis already frozen')
    inventory=json.loads((adapter.INVENTORY/'summary.json').read_text())
    if base.file_sha(adapter.INVENTORY/'lexical-index.sqlite')!=inventory['database_sha256']:
        raise ValueError('lexical inventory changed')
    original=json.loads((adapter.ORIGINAL/'summary.json').read_text())
    if base.file_sha(adapter.ORIGINAL/'english-assertions.sqlite')!=original['database_sha256']:
        raise ValueError('original assertion metadata changed')
    graph=adapter.SourceGraph();skills=SkillLibrary(base.REPO).render(adapter.SKILLS)
    anchors=list(graph.db.execute('SELECT base_uri,subject FROM anchors ORDER BY base_uri'))
    sample=random.Random(2026102707).sample(anchors,1024);pools=defaultdict(list);counts=Counter();held=[]
    for anchor,subject in sample:
        try:
            row=graph.make_row(anchor,subject,skills);memory=restore_relational_memory(row)
            if canonical(independent_sql_target(row['source_refs']['query'],memory.facts))!=row['messages'][-1]['content']:
                raise ValueError('independent SQL target disagrees')
            for call in row['source_refs']['actual_tool_calls']:
                if base.sha(canonical(memory.lookup(**call['arguments'])))!=call['observation_sha256']:
                    raise ValueError('portable actual observation differs')
            key=(row['source_refs']['query_schema_id'],row['memory_placement'])
            counts[':'.join(key)]+=1
            if len(pools[key])<2:pools[key].append(row)
        except Rejected as exc:held.append({'base_uri':anchor,'reason':str(exc)})
    selected=[r for key in sorted(pools) for r in pools[key]][:36]
    if len(selected)<18:raise ValueError('insufficient diagnostic coverage')
    base.init_tokenizer();package.init();budgets=[];proofs=[];jobs=[]
    for row in selected:
        record,_,_=base.compile_row(row);native=package.request_budget(record)
        job={'id':'sample-'+row['record_id'],'record_id':row['record_id'],'domain':row['domain'],
             'messages':row['messages'],'tools':row['tools'],'observation_origin':row['observation_origin']}
        body=transport.body_for(job)
        rendered=package.CHAT.render(messages=body['messages'],tools=[],add_generation_prompt=True,
            enable_thinking=body['chat_template_kwargs']['enable_thinking'],bos_token='',eos_token='<|im_end|>')
        tokens=len(package.QWEN.encode(rendered,add_special_tokens=False).ids)
        if tokens+body['max_tokens']>1000000:raise ValueError('complete review input exceeds actual context')
        budgets.append({'record_id':row['record_id'],'native':native,'audit_input_tokens':tokens,
            'audit_output_reserve':body['max_tokens'],'audit_request_sha256':reviewer.digest(body),
            'pangolin_tokens':record['pangolin_tokens'],'minicpm_tokens':record['minicpm_tokens']})
        proofs.append({'record_id':row['record_id'],'base_concept':row['source_task_key'],
            'independent_sql_target_equal':True,'snapshot_sha256':row['source_refs']['snapshot_sha256'],
            'actual_observations_restored':len(row['source_refs']['actual_tool_calls'])})
        jobs.append(job)
    original_plan=json.loads((base.WORK/'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
    plan={k:original_plan[k] for k in ['schema_sha256','prompt_sha256','model','revision','controls']}
    plan.update(jobs=jobs,first_non_accept_is_final_for_that_record=True,
        scope='First 1024 seeded distinct base-concept probes; up to two per query-schema/memory-placement stratum, max 36. No previous graph or lexical reviews replayed. Every target independently checked with SQLite. This diagnosis is not corpus accuracy or measured model improvement.')
    folder=STATE/'normalized';folder.mkdir(exist_ok=True)
    with (folder/'pilot-lexical.jsonl').open('w') as handle:
        for row in selected:handle.write(canonical(row)+'\n')
    reviewer.dump(STATE/'semantic-audit-v1/plan.json',plan)
    base.save(STATE/'budget-proof.json',budgets);base.save(STATE/'independent-sql-proofs.json',proofs)
    base.save(STATE/'pilot-summary.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'adapter_sha256':base.file_sha(Path(adapter.__file__)),'transport_sha256':base.file_sha(Path(transport.__file__)),
        'relational_memory_sha256':base.file_sha(base.REPO/'barbet_inference/relational_memory.py'),
        'independent_sql_sha256':base.file_sha(base.REPO/'scripts/diagnose_wikidata_relations_v1.py'),
        'plan_sha256':base.file_sha(STATE/'semantic-audit-v1/plan.json'),
        'source_inventory_sha256':base.file_sha(adapter.INVENTORY/'summary.json'),
        'original_inventory_sha256':base.file_sha(adapter.ORIGINAL/'summary.json'),
        'budget_proof_sha256':base.file_sha(STATE/'budget-proof.json'),
        'independent_sql_proof_sha256':base.file_sha(STATE/'independent-sql-proofs.json'),
        'bounded_anchor_probes':len(sample),'probe_counts':dict(counts),'structural_source_holds':held,
        'diagnostic_samples':len(selected),'distinct_base_concepts':len({r['source_task_key'] for r in selected}),
        'single_canonical_target_per_task':True,'same_question_alternatives_require_native_v26_ppl':True,
        'accepted_training_rows':0,'source_programs_executed':False,'model_weights_trained':False})
    print(json.dumps({'samples':len(selected),'bounded_anchors':len(sample),'counts':dict(counts),'structural_holds':len(held)}),flush=True)


def audit():
    binding=json.loads((STATE/'pilot-summary.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('adapter_sha256',Path(adapter.__file__)),
        ('transport_sha256',Path(transport.__file__)),('plan_sha256',STATE/'semantic-audit-v1/plan.json'),
        ('relational_memory_sha256',base.REPO/'barbet_inference/relational_memory.py'),
        ('independent_sql_sha256',base.REPO/'scripts/diagnose_wikidata_relations_v1.py')]:
        if base.file_sha(path)!=binding[key]:raise ValueError('first lexical diagnosis binding changed')
    reviewer.STATE=STATE;reviewer.review=transport.review;reviewer.main()
    summary=json.loads((STATE/'semantic-audit-v1/summary.json').read_text());bad=set(summary['excluded_record_ids'])
    holds=[{'record_id':r['record_id'],'query_hash':r['query_hash'],'base_uri':r['source_task_key']}
        for r in map(json.loads,(STATE/'normalized/pilot-lexical.jsonl').open()) if r['record_id'] in bad]
    if len(holds)!=len(bad):raise ValueError('first source hold coverage differs')
    base.save(STATE/'first-review-query-holds.json',{'holds':holds,'base_concepts':[h['base_uri'] for h in holds],
        'query_hashes':[h['query_hash'] for h in holds],'first_audit_sha256':base.file_sha(STATE/'semantic-audit-v1/summary.json'),
        'first_nonaccepts_not_replayed':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','audit']);args=parser.parse_args()
    globals()[args.phase]()
