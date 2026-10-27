"""Full bounded relation candidate construction, one source anchor per task.

All final admission/identity/token/budget/publication gates remain downstream.
No teacher is substituted: targets are exact procedural finite-record queries,
independently recomputed with SQLite on every row. No model scores are invented.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.sft import Rejected, RESERVED
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts import wikidata_relation_adapter_v2 as adapter
from scripts import diagnose_wikidata_relations_v1 as diagnosis
from scripts.prepare_v26_answer_candidates import canonicalize
from scripts.normalize_ifm_math_memory_v1 import processed_bytes
from scripts.rank_ifm_code_v26_v1 import FROZEN as PRIOR_FOLDS, STATE as PRIOR_CODE
from scripts.publish_ifm_code_v26_v1 import STATE as PRIOR_PUBLIC

STATE=adapter.INVENTORY.parent/'full-relations-v1'
RESOURCES=base.REPO/'configs/memory_sft_v2_scale_10m_resources_v3.json'
JOBS=456


def prepare():
    STATE.mkdir(exist_ok=True);(STATE/'normalized').mkdir(exist_ok=True)
    path=STATE/'normalization-policy.json'
    if path.exists():
        policy=json.loads(path.read_text())
        if policy['driver_sha256']!=base.file_sha(Path(__file__)):raise ValueError('full relation normalizer changed')
        return policy
    proof=json.loads((PRIOR_PUBLIC/'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:raise ValueError('prior release not fully verified')
    first=json.loads((diagnosis.STATE/'semantic-audit-v1/summary.json').read_text())
    holds=json.loads((diagnosis.STATE/'first-review-query-holds.json').read_text())
    if len(holds['holds'])!=len(first['excluded_record_ids']):raise ValueError('first review hold coverage differs')
    probe=json.loads((adapter.INVENTORY.parent/'export-probe-v1/summary.json').read_text())
    if probe['prior_pilot_model_input_equivalence_rows']!=36 or probe['prior_actual_observations_restored_identically']!=32:
        raise ValueError('portable metadata equivalence not established')
    resources=json.loads(RESOURCES.read_text())['resources'];used=processed_bytes();free=shutil.disk_usage(base.WORK).free
    total_reserve=64*1024**3;candidate_bound=32*1024**3
    if used+total_reserve>resources['new_processed_cap_bytes'] or free<resources['minimum_free_disk_bytes']+total_reserve:
        raise ValueError('full source expansion reservation would exceed storage cap or floor')
    inventory=json.loads((adapter.INVENTORY/'summary.json').read_text())
    skills=SkillLibrary(base.REPO).render(adapter.SKILLS)
    policy={'driver_sha256':base.file_sha(Path(__file__)),
        'adapter_v2_sha256':base.file_sha(Path(adapter.__file__)),
        'adapter_v1_sha256':base.file_sha(Path(adapter.original.__file__)),
        'relational_memory_sha256':base.file_sha(base.REPO/'barbet_inference/relational_memory.py'),
        'restorer_sha256':base.file_sha(base.REPO/'barbet_inference/relational_dataset.py'),
        'independent_sql_driver_sha256':base.file_sha(Path(diagnosis.__file__)),
        'archive_sha256':base.file_sha(Path(archive.__file__)),
        'source_inventory_sha256':base.file_sha(adapter.INVENTORY/'summary.json'),
        'first_source_audit_sha256':base.file_sha(diagnosis.STATE/'semantic-audit-v1/summary.json'),
        'first_hold_proof_sha256':base.file_sha(diagnosis.STATE/'first-review-query-holds.json'),
        'portable_export_probe_sha256':base.file_sha(adapter.INVENTORY.parent/'export-probe-v1/summary.json'),
        'resource_config_sha256':base.file_sha(RESOURCES),
        'prior_frozen_aliases_sha256':base.file_sha(PRIOR_FOLDS/'frozen-aliases.sqlite'),
        'prior_code_candidates_sha256':base.file_sha(PRIOR_CODE/'candidates.sqlite'),
        'prior_publication_sha256':base.file_sha(PRIOR_PUBLIC/'publication-final-commit.json'),
        'benchmark_exclusions_sha256':base.file_sha(ni.STATE/'exclusions.private.json'),
        'rendered_skills_sha256':base.sha(canonical(skills)),
        'source_anchor_count':inventory['usable_training_subjects'],'job_count':JOBS,
        'first_held_anchor_ids':holds['anchor_ids'],'first_held_query_hashes':holds['query_hashes'],
        'processed_bytes_before':used,'free_disk_bytes_before':free,
        'full_stage_reserved_bytes':total_reserve,'candidate_stage_bound_bytes':candidate_bound,
        'source_task_contract':'One fixed relation task per labeled training anchor, using complete bounded adjacency; no memory-position or answer-style duplicates. Keep source direction, distinct endpoint semantics and all traversed evidence IDs.',
        'first_review_count_is_not_corpus_error_rate':True,'all_row_exact_sql_verification_required':True,
        'all_first_source_holds_preserved':True,'model_requests':0,'accepted_training_rows':0,
        'single_targets_require_later_global_identity_confirmation':True,
        'alternatives_if_discovered_require_frozen_v26_ppl_not_arbitrary_choice':True}
    jobs=[{'id':f'{i:05d}','first_position':i,'step':JOBS,'stop':policy['source_anchor_count'],
           'maximum_archive_bytes':(candidate_bound-1024**3)//JOBS} for i in range(JOBS)]
    base.save(STATE/'jobs.json',jobs);base.save(path,policy)
    print(json.dumps({'source_anchors':policy['source_anchor_count'],'jobs':JOBS,'reserved_bytes':total_reserve,'first_anchor_holds':len(holds['anchor_ids'])}),flush=True)
    return policy


def init():
    global GRAPH,SKILLS,POLICY,HOLD_ANCHORS,HOLD_QUERIES,PRIOR_QUERIES
    POLICY=json.loads((STATE/'normalization-policy.json').read_text())
    for key,path in [('driver_sha256',Path(__file__)),('adapter_v2_sha256',Path(adapter.__file__)),
        ('adapter_v1_sha256',Path(adapter.original.__file__)),('restorer_sha256',base.REPO/'barbet_inference/relational_dataset.py'),
        ('relational_memory_sha256',base.REPO/'barbet_inference/relational_memory.py'),
        ('independent_sql_driver_sha256',Path(diagnosis.__file__)),('archive_sha256',Path(archive.__file__))]:
        if base.file_sha(path)!=POLICY[key]:raise ValueError('frozen source implementation differs: '+key)
    GRAPH=adapter.SourceGraph();SKILLS=SkillLibrary(base.REPO).render(adapter.SKILLS)
    if base.sha(canonical(SKILLS))!=POLICY['rendered_skills_sha256']:raise ValueError('frozen skill render differs')
    HOLD_ANCHORS=set(POLICY['first_held_anchor_ids']);HOLD_QUERIES=set(POLICY['first_held_query_hashes']);ni.init()
    old=sqlite3.connect(f'file:{PRIOR_FOLDS / "frozen-aliases.sqlite"}?mode=ro',uri=True)
    PRIOR_QUERIES={r[0][6:] for r in old.execute("SELECT node FROM aliases WHERE node >= 'query:' AND node < 'query;'")};old.close()
    code=sqlite3.connect(f'file:{PRIOR_CODE / "candidates.sqlite"}?mode=ro',uri=True)
    PRIOR_QUERIES.update(q for q, in code.execute('SELECT query_hash FROM candidates'));code.close()


def job(spec):
    path=STATE/'normalized'/('full-'+spec['id']+'.bca');receipt=path.with_suffix('.report.json')
    if receipt.exists():
        report=json.loads(receipt.read_text())
        if base.file_sha(path)!=report['sha256']:raise ValueError('completed relation archive changed')
        return report
    if path.exists():raise ValueError('partial relation archive exists; preserve and inspect')
    counts=Counter();held=[];raw_bytes=0;schema_counts=Counter()
    with archive.ArchiveWriter(path) as writer:
        for pos in range(spec['first_position'],spec['stop'],spec['step']):
            anchor=int(GRAPH.heads[pos]);anchor_id='Q'+str(anchor);counts['source_anchors_seen']+=1
            try:
                if anchor_id in HOLD_ANCHORS:raise Rejected('first_source_nonaccept_anchor')
                row=GRAPH.make_row(anchor,SKILLS)
                if row['query_hash'] in HOLD_QUERIES:raise Rejected('first_source_nonaccept_query')
                if row['query_hash'] in PRIOR_QUERIES:raise Rejected('prior_question_alias')
                if RESERVED.search(canonical(row['messages'])+canonical(row['tools'])):raise Rejected('literal_reserved_delimiter')
                if any(ni.benchmark_overlap(m['content']) for m in row['messages'] if m['role']=='user'):
                    raise Rejected('local_evaluation_text_overlap')
                memory=restore_relational_memory(row)
                expected=diagnosis.independent_sql_target(row['source_refs']['query'],memory.facts)
                if canonical(expected)!=row['messages'][-1]['content']:raise ValueError('independent source SQL disagreement')
                for call in row['source_refs']['actual_tool_calls']:
                    if base.sha(canonical(memory.lookup(**call['arguments'])))!=call['observation_sha256']:
                        raise ValueError('restored tool observation differs')
                row=canonicalize(row)
            except Rejected as exc:
                counts['held:'+str(exc)]+=1;held.append({'anchor_id':anchor_id,'reason':str(exc)});continue
            raw=(canonical(row)+'\n').encode();writer.append(raw);raw_bytes+=len(raw)
            if writer.handle.tell()>spec['maximum_archive_bytes']:
                raise ValueError('fixed candidate-shard archive byte allocation exhausted')
            counts['candidate_answers']+=1;counts['independent_sql_targets_equal']+=1
            counts['actual_observations_restored']+=len(row['source_refs']['actual_tool_calls'])
            schema_counts[row['source_refs']['query_schema_id']]+=1
    excluded=path.with_suffix('.excluded.json');base.save(excluded,held)
    result={'job':spec['id'],'path':str(path),'sha256':base.file_sha(path),'bytes':path.stat().st_size,
        'source_json_payload_bytes':raw_bytes,'counts':dict(counts),'query_schema_counts':dict(schema_counts),
        'exclusions_sha256':base.file_sha(excluded),'policy_sha256':base.file_sha(STATE/'normalization-policy.json'),
        'accepted_training_rows':0,'source_programs_executed':False,'model_requests':0}
    base.save(receipt,result);return result


def main(workers):
    policy=prepare()
    # Verify large immutable identity sources once in the parent, before workers.
    for key,path in [('prior_frozen_aliases_sha256',PRIOR_FOLDS/'frozen-aliases.sqlite'),('prior_code_candidates_sha256',PRIOR_CODE/'candidates.sqlite')]:
        if base.file_sha(path)!=policy[key]:raise ValueError('prior identity evidence changed')
    jobs=json.loads((STATE/'jobs.json').read_text());totals=Counter();schemas=Counter();parts=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,part in enumerate(pool.map(job,jobs),1):
            totals.update(part['counts']);schemas.update(part['query_schema_counts']);parts.append(part)
            print(json.dumps({'shards':n,'total':len(jobs),'candidate_answers':totals['candidate_answers']}),flush=True)
    if totals['source_anchors_seen']!=policy['source_anchor_count']:raise ValueError('complete anchor coverage differs')
    result={'counts':dict(totals),'query_schema_counts':dict(schemas),'files':parts,
        'archive_bytes':sum(p['bytes'] for p in parts),'source_json_payload_bytes':sum(p['source_json_payload_bytes'] for p in parts),
        'policy_sha256':base.file_sha(STATE/'normalization-policy.json'),'accepted_training_rows':0,
        'needs_global_identity_fold_token_budget_and_publication_gates':True,'model_requests':0}
    base.save(STATE/'normalize-summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    if not 1<=args.workers<=24:raise ValueError('CPU worker bound')
    main(args.workers)
