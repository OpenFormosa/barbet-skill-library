"""First full-contract graph-task diagnosis, before any source admission."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3

from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from barbet_inference import relational_memory
from scripts import build_memory_sft_v2 as base
from scripts import package_memory_sft_v2 as package
from scripts import audit_memory_sft_v2 as reviewer
from scripts import audit_memory_sft_math_v2 as audit_transport
from scripts import wikidata_relation_adapter_v1 as adapter

STATE = adapter.INVENTORY.parent / 'first-relation-diagnosis-v1'


def independent_sql_target(query, facts):
    """Independent SQLite joins over public records, never source program text."""
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE facts(id TEXT PRIMARY KEY,subject TEXT,relation TEXT,object TEXT)')
    db.executemany('INSERT INTO facts VALUES(?,?,?,?)', [(f.id,f.subject,f.relation,f.object) for f in facts])
    branches = []; evidence = set()
    for branch in query['branches']:
        current = [branch['start']]
        for step in branch['path']:
            if not current: break
            source, target = ('subject','object') if step['direction']=='out' else ('object','subject')
            sql = 'SELECT id,'+target+' FROM facts WHERE relation=? AND '+source+' IN ('+','.join('?' for _ in current)+')'
            matches = db.execute(sql,[step['relation']]+current).fetchall()
            evidence.update(r[0] for r in matches); current = sorted({r[1] for r in matches})
        branches.append(current)
    for i, ids in enumerate(branches):
        db.execute('CREATE TABLE branch'+str(i)+'(id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO branch'+str(i)+' VALUES(?)', [(x,) for x in ids])
    op = query['operation']
    if op == 'count': target = {'count':db.execute('SELECT COUNT(*) FROM branch0').fetchone()[0]}
    else:
        sql = 'SELECT id FROM branch0'
        if len(branches)==2: sql += {'union':' UNION ','intersection':' INTERSECT ','difference':' EXCEPT '}[op]+'SELECT id FROM branch1'
        target = {'answer_ids':[r[0] for r in db.execute(sql+' ORDER BY id')]}
    db.close(); target['evidence_ids'] = sorted(evidence); return target


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('relation source diagnosis already frozen')
    inventory = json.loads((adapter.INVENTORY / 'summary.json').read_text())
    for item in inventory['files']:
        if base.file_sha(adapter.INVENTORY / item['path']) != item['sha256']:
            raise ValueError('source inventory changed before first task diagnosis')
    graph = adapter.SourceGraph(); skills = SkillLibrary(base.REPO).render(adapter.SKILLS)
    rng = random.Random(2026102703); indices = rng.sample(range(len(graph.heads)),2048)
    candidates = []; counts = Counter(); source_holds = []; pools = defaultdict(list)
    for index in indices:
        anchor = int(graph.heads[index])
        try:
            row = graph.make_row(anchor,skills)
            q = row['source_refs']['query']; key = (q['operation'],row['memory_placement'],len(q['branches'][0]['path']))
            if len(pools[key]) < 2: pools[key].append(row)
            counts[row['source_refs']['query_schema_id']+':'+row['memory_placement']] += 1
        except Rejected as exc:
            source_holds.append({'anchor':'Q'+str(anchor),'reason':str(exc)})
    # Distinct source entities throughout; two cases per available operator,
    # placement and hop-count stratum, up to 36 total. No retries of old cases.
    selected = [r for key in sorted(pools) for r in pools[key]][:36]
    base.init_tokenizer(); package.init(); budgets = []; exact_proofs = []; jobs = []
    for row in selected:
        refs = row['source_refs']; facts = graph.facts([i-1 for i in refs['original_training_source_line_numbers']])
        expected = independent_sql_target(refs['query'],facts)
        if canonical(expected) != row['messages'][-1]['content']:
            raise ValueError('independent SQL disagrees with procedural target')
        record,_,_ = base.compile_row(row); measured = package.request_budget(record)
        job = {'id':'sample-'+row['record_id'],'record_id':row['record_id'],'domain':row['domain'],
               'messages':row['messages'],'tools':row['tools'],'observation_origin':row['observation_origin']}
        body = audit_transport.body_for(job)
        rendered = package.CHAT.render(messages=body['messages'],tools=[],add_generation_prompt=True,
            enable_thinking=body['chat_template_kwargs']['enable_thinking'],bos_token='',eos_token='<|im_end|>')
        input_tokens = len(package.QWEN.encode(rendered,add_special_tokens=False).ids)
        if input_tokens + body['max_tokens'] > 1000000: raise ValueError('review input exceeds served budget')
        budgets.append({'record_id':row['record_id'],'native':measured,'audit_input_tokens':input_tokens,
            'audit_output_reserve':body['max_tokens'],'audit_request_sha256':reviewer.digest(body),
            'pangolin_tokens':record['pangolin_tokens'],'minicpm_tokens':record['minicpm_tokens']})
        exact_proofs.append({'record_id':row['record_id'],'anchor_id':refs['source_anchor_id'],
            'snapshot_sha256':refs['snapshot_sha256'],'target_sha256':base.sha(row['messages'][-1]['content']),
            'independent_sql_target_equal':True,'actual_callback_observations':len(refs['actual_tool_calls'])})
        jobs.append(job)
    original = json.loads((base.WORK / 'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
    plan = {k:original[k] for k in ['schema_sha256','prompt_sha256','model','revision','controls']}
    plan.update(jobs=jobs,first_non_accept_is_final_for_that_record=True,
        scope='First stratified diagnosis of deterministic finite-record relation tasks from 2048 seeded distinct training-entity anchors. Up to two per available operation/placement/hop stratum, at most 36. Exact SQL target comparisons are separate from semantic review. Not whole-corpus accuracy or demonstrated model improvement.')
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-relations.jsonl').open('w') as handle:
        for row in selected: handle.write(canonical(row)+'\n')
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json',plan)
    base.save(STATE / 'budget-proof.json',budgets); base.save(STATE / 'independent-sql-proofs.json',exact_proofs)
    base.save(STATE / 'pilot-summary.json', {'driver_sha256':base.file_sha(Path(__file__)),
        'adapter_sha256':base.file_sha(Path(adapter.__file__)),
        'relational_memory_sha256':base.file_sha(Path(relational_memory.__file__)),
        'transport_sha256':base.file_sha(Path(audit_transport.__file__)),
        'plan_sha256':base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'source_inventory_sha256':base.file_sha(adapter.INVENTORY / 'summary.json'),
        'budget_proof_sha256':base.file_sha(STATE / 'budget-proof.json'),
        'independent_sql_proof_sha256':base.file_sha(STATE / 'independent-sql-proofs.json'),
        'bounded_anchor_probes':len(indices),'probe_counts':dict(counts),'structural_source_holds':source_holds,
        'diagnostic_samples':len(selected),'source_entities':len({r['source_task_key'] for r in selected}),
        'single_canonical_target_per_task':True,'same_question_alternatives_would_require_v26_ppl':True,
        'model_weights_trained':False,'accepted_training_rows':0,
        'source_programs_executed':False,'exact_procedural_targets_not_teacher_generated_prose':True})
    print(json.dumps({'samples':len(selected),'bounded_anchors':len(indices),'counts':dict(counts),'structural_holds':len(source_holds)}),flush=True)


def audit():
    binding = json.loads((STATE / 'pilot-summary.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('adapter_sha256',Path(adapter.__file__)),
                     ('relational_memory_sha256',Path(relational_memory.__file__)),
                     ('transport_sha256',Path(audit_transport.__file__)),
                     ('plan_sha256',STATE/'semantic-audit-v1/plan.json')]:
        if base.file_sha(path) != binding[key]: raise ValueError('first graph diagnosis binding changed')
    reviewer.STATE = STATE; reviewer.review = audit_transport.review; reviewer.main()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','audit']);args=parser.parse_args()
    globals()[args.phase]()
