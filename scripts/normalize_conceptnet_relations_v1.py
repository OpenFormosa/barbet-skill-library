"""Bounded lexical-memory expansion with complete source and SQL verification."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

from barbet_inference import candidate_archive as archive
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.sft import Rejected,RESERVED
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts import conceptnet_relation_adapter_v1 as adapter
from scripts import diagnose_conceptnet_relations_v1 as diagnosis
from scripts.diagnose_wikidata_relations_v1 import independent_sql_target
from scripts.prepare_v26_answer_candidates import canonicalize
from scripts.normalize_ifm_math_memory_v1 import processed_bytes
from scripts.index_wikidata_relations_v1 import FROZEN as PRIOR_FROZEN,STATE as GRAPH_INDEX
from scripts.finalize_wikidata_singletons_v1 import STATE as GRAPH_SELECTED,OUT as GRAPH_OUT

STATE=adapter.INVENTORY.parent/'full-relations-v1'
RESOURCES=base.REPO/'configs/memory_sft_v2_scale_10m_resources_v3.json'
JOBS=88


def prepare():
    STATE.mkdir(exist_ok=True);(STATE/'normalized').mkdir(exist_ok=True)
    path=STATE/'normalization-policy.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value['driver_sha256']!=base.file_sha(Path(__file__)):raise ValueError('normalizer frozen implementation changed')
        return value
    first=json.loads((diagnosis.STATE/'semantic-audit-v1/summary.json').read_text())
    holds=json.loads((diagnosis.STATE/'first-review-query-holds.json').read_text())
    if len(first['excluded_record_ids'])!=len(holds['holds']):raise ValueError('first hold coverage differs')
    probe=json.loads((adapter.INVENTORY.parent/'export-probe-v1/summary.json').read_text())
    if not probe['all_three_configs_and_archive_exact_readback']:raise ValueError('portable lexical export not verified')
    graph=json.loads((GRAPH_SELECTED/'ppl-score-binding.json').read_text())
    if graph['competing_groups'] or graph['model_requests']:raise ValueError('prior graph singleton contract changed')
    resources=json.loads(RESOURCES.read_text())['resources'];used=processed_bytes();free=shutil.disk_usage(base.WORK).free
    # Respect the simultaneous graph export reservation, not just bytes already
    # written. No second public packager is started during this construction.
    graph_remaining=0
    if not (GRAPH_SELECTED/'compile-summary.json').exists():
        policy=json.loads((GRAPH_SELECTED/'framed-compile-policy.json').read_text())
        written=sum(p.stat().st_size for p in GRAPH_OUT.rglob('*.parquet'))
        graph_remaining=max(0,policy['reserved_compile_bytes']-written)
    reserve=8*1024**3;candidate_bound=6*1024**3
    if used+reserve+graph_remaining>resources['new_processed_cap_bytes'] or free<resources['minimum_free_disk_bytes']+reserve+graph_remaining:
        raise ValueError('combined in-flight lexical and graph reservations exceed free floor or processed cap')
    inventory=json.loads((adapter.INVENTORY/'summary.json').read_text());skills=SkillLibrary(base.REPO).render(adapter.SKILLS)
    policy={'driver_sha256':base.file_sha(Path(__file__)),
        'adapter_sha256':base.file_sha(Path(adapter.__file__)),
        'relational_memory_sha256':base.file_sha(base.REPO/'barbet_inference/relational_memory.py'),
        'restorer_sha256':base.file_sha(base.REPO/'barbet_inference/relational_dataset.py'),
        'independent_sql_sha256':base.file_sha(base.REPO/'scripts/diagnose_wikidata_relations_v1.py'),
        'archive_sha256':base.file_sha(Path(archive.__file__)),
        'source_inventory_sha256':base.file_sha(adapter.INVENTORY/'summary.json'),
        'source_database_sha256':inventory['database_sha256'],
        'original_source_database_sha256':json.loads((adapter.ORIGINAL/'summary.json').read_text())['database_sha256'],
        'first_audit_sha256':base.file_sha(diagnosis.STATE/'semantic-audit-v1/summary.json'),
        'first_holds_sha256':base.file_sha(diagnosis.STATE/'first-review-query-holds.json'),
        'export_probe_sha256':base.file_sha(adapter.INVENTORY.parent/'export-probe-v1/summary.json'),
        'prior_frozen_aliases_sha256':base.file_sha(PRIOR_FROZEN/'frozen-aliases.sqlite'),
        'prior_graph_candidates_sha256':base.file_sha(GRAPH_INDEX/'candidates.sqlite'),
        'prior_graph_selection_sha256':base.file_sha(GRAPH_SELECTED/'ppl-score-binding.json'),
        'benchmark_exclusions_sha256':base.file_sha(ni.STATE/'exclusions.private.json'),
        'resource_policy_sha256':base.file_sha(RESOURCES),'rendered_skills_sha256':base.sha(canonical(skills)),
        'source_anchor_count':inventory['distinct_base_concept_anchors'],'job_count':JOBS,
        'first_held_base_concepts':holds['base_concepts'],'first_held_query_hashes':holds['query_hashes'],
        'processed_bytes_before':used,'free_bytes_before':free,'reserved_stage_bytes':reserve,
        'concurrent_graph_reserved_remaining_bytes':graph_remaining,'candidate_bound_bytes':candidate_bound,
        'one_task_per_base_concept':True,'source_uri_senses_polarity_and_per_assertion_licenses_preserved':True,
        'all_row_independent_SQL_required':True,'all_actual_observations_restored':True,
        'all_first_holds_preserved':True,'model_requests':0,'accepted_training_rows':0,
        'any_actual_answer_alternatives_require_frozen_native_v26_ppl':True}
    jobs=[{'id':f'{i:05d}','first_position':i,'step':JOBS,'stop':policy['source_anchor_count'],
        'maximum_archive_bytes':(candidate_bound-128*1024**2)//JOBS} for i in range(JOBS)]
    base.save(STATE/'jobs.json',jobs);base.save(path,policy)
    print(json.dumps({'source_base_concepts':policy['source_anchor_count'],'jobs':JOBS,'first_holds':len(holds['holds']),
        'candidate_bound_bytes':candidate_bound,'graph_reserved_remaining':graph_remaining}),flush=True)
    return policy


def init():
    global GRAPH,ANCHORS,SKILLS,POLICY,HOLD_ANCHORS,HOLD_QUERIES,PRIOR_QUERIES
    POLICY=json.loads((STATE/'normalization-policy.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('adapter_sha256',Path(adapter.__file__)),
        ('relational_memory_sha256',base.REPO/'barbet_inference/relational_memory.py'),
        ('restorer_sha256',base.REPO/'barbet_inference/relational_dataset.py'),
        ('independent_sql_sha256',base.REPO/'scripts/diagnose_wikidata_relations_v1.py'),('archive_sha256',Path(archive.__file__))]:
        if base.file_sha(path)!=POLICY[key]:raise ValueError('bound lexical implementation changed: '+key)
    GRAPH=adapter.SourceGraph();ANCHORS=list(GRAPH.db.execute('SELECT base_uri,subject FROM anchors ORDER BY base_uri'))
    if len(ANCHORS)!=POLICY['source_anchor_count']:raise ValueError('base concept coverage differs')
    SKILLS=SkillLibrary(base.REPO).render(adapter.SKILLS)
    if base.sha(canonical(SKILLS))!=POLICY['rendered_skills_sha256']:raise ValueError('frozen skill render differs')
    HOLD_ANCHORS=set(POLICY['first_held_base_concepts']);HOLD_QUERIES=set(POLICY['first_held_query_hashes']);ni.init()
    old=sqlite3.connect(f'file:{PRIOR_FROZEN / "frozen-aliases.sqlite"}?mode=ro',uri=True)
    PRIOR_QUERIES={r[0][6:] for r in old.execute("SELECT node FROM aliases WHERE node >= 'query:' AND node < 'query;'")};old.close()
    graph=sqlite3.connect(f'file:{GRAPH_INDEX / "candidates.sqlite"}?mode=ro',uri=True)
    PRIOR_QUERIES.update(q for q, in graph.execute('SELECT query_hash FROM candidates'));graph.close()


def job(spec):
    path=STATE/'normalized'/('full-'+spec['id']+'.bca');receipt=path.with_suffix('.report.json')
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if base.file_sha(path)!=result['sha256'] or result['policy_sha256']!=base.file_sha(STATE/'normalization-policy.json'):
            raise ValueError('completed lexical shard differs')
        return result
    if path.exists():raise ValueError('partial source archive exists; preserve and diagnose')
    counts=Counter();schemas=Counter();held=[];raw_bytes=0
    with archive.ArchiveWriter(path) as writer:
        for pos in range(spec['first_position'],spec['stop'],spec['step']):
            anchor,subject=ANCHORS[pos];counts['source_base_concepts_seen']+=1
            try:
                if anchor in HOLD_ANCHORS:raise Rejected('first_source_nonaccept_base_concept')
                row=GRAPH.make_row(anchor,subject,SKILLS)
                if row['query_hash'] in HOLD_QUERIES:raise Rejected('first_source_nonaccept_query')
                if row['query_hash'] in PRIOR_QUERIES:raise Rejected('prior_question_alias')
                if RESERVED.search(canonical(row['messages'])+canonical(row['tools'])):raise Rejected('literal_reserved_delimiter')
                if any(ni.benchmark_overlap(m['content']) for m in row['messages'] if m['role']=='user'):
                    raise Rejected('local_evaluation_text_overlap')
                memory=restore_relational_memory(row)
                if canonical(independent_sql_target(row['source_refs']['query'],memory.facts))!=row['messages'][-1]['content']:
                    raise ValueError('independent lexical SQL disagreement')
                for call in row['source_refs']['actual_tool_calls']:
                    if base.sha(canonical(memory.lookup(**call['arguments'])))!=call['observation_sha256']:
                        raise ValueError('restored lexical observation differs')
                row=canonicalize(row)
            except Rejected as exc:counts['held:'+str(exc)]+=1;held.append({'base_uri':anchor,'source_subject':subject,'reason':str(exc)});continue
            raw=(canonical(row)+'\n').encode();writer.append(raw);raw_bytes+=len(raw)
            if writer.handle.tell()>spec['maximum_archive_bytes']:raise ValueError('fixed lexical shard allocation exhausted')
            counts['candidate_answers']+=1;counts['independent_SQL_targets_equal']+=1
            counts['actual_observations_restored']+=len(row['source_refs']['actual_tool_calls'])
            schemas[row['source_refs']['query_schema_id']]+=1
    excluded=path.with_suffix('.excluded.json');base.save(excluded,held)
    result={'job':spec['id'],'path':str(path),'sha256':base.file_sha(path),'bytes':path.stat().st_size,
        'source_json_payload_bytes':raw_bytes,'counts':dict(counts),'query_schema_counts':dict(schemas),
        'exclusions_sha256':base.file_sha(excluded),'policy_sha256':base.file_sha(STATE/'normalization-policy.json'),
        'accepted_training_rows':0,'source_programs_executed':False,'model_requests':0}
    base.save(receipt,result);return result


def main(workers):
    policy=prepare()
    for key,path in [('source_database_sha256',adapter.INVENTORY/'lexical-index.sqlite'),
        ('original_source_database_sha256',adapter.ORIGINAL/'english-assertions.sqlite'),
        ('prior_frozen_aliases_sha256',PRIOR_FROZEN/'frozen-aliases.sqlite'),
        ('prior_graph_candidates_sha256',GRAPH_INDEX/'candidates.sqlite')]:
        if base.file_sha(path)!=policy[key]:raise ValueError('bound source or prior identity changed')
    jobs=json.loads((STATE/'jobs.json').read_text());counts=Counter();schemas=Counter();parts=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,part in enumerate(pool.map(job,jobs),1):
            parts.append(part);counts.update(part['counts']);schemas.update(part['query_schema_counts'])
            print(json.dumps({'shards':n,'total':len(jobs),'candidate_answers':counts['candidate_answers']}),flush=True)
    if counts['source_base_concepts_seen']!=policy['source_anchor_count']:raise ValueError('full source coverage differs')
    size=sum(p['bytes'] for p in parts)
    if size>policy['candidate_bound_bytes']:raise ValueError('complete archive bound exceeded')
    result={'counts':dict(counts),'query_schema_counts':dict(schemas),'files':parts,'candidate_bytes':size,
        'source_json_payload_bytes':sum(p['source_json_payload_bytes'] for p in parts),
        'policy_sha256':base.file_sha(STATE/'normalization-policy.json'),'accepted_training_rows':0,
        'all_first_holds_preserved':True,'full_identity_selection_export_and_publication_still_required':True,
        'model_requests':0,'source_programs_executed':False,'model_weights_trained':False}
    base.save(STATE/'normalize-summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if not 1<=args.workers<=16:raise ValueError('CPU worker bound')
    main(args.workers)
