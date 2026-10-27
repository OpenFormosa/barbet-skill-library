"""Apply the existing full export gates to exact, explicitly unscored tasks."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts import normalize_conceptnet_relations_v1 as normalizer
from scripts import finalize_conceptnet_singletons_v1 as selected
from scripts import package_memory_sft_v2 as package
from scripts import release_natural_instructions_v26_v1 as checks
from scripts.publish_wikidata_relations_v1 import STATE as PRIOR

STATE=selected.STATE
OUT=selected.OUT
SOURCE=normalizer.diagnosis.STATE
PUBLIC=base.WORK/'exports/memory-harness-v2-conceptnet-relations-public-v1'
WAVE='wave-0010'


def init():
    # Load the actual source diagnosis, without relabeling it as a new winner
    # audit. All records have one exact target, confirmed by the global index.
    package.STATE=SOURCE;package.init()
    package.STATE=STATE;package.OUT=OUT;package.PUBLIC=PUBLIC;package.WAVE=WAVE
    package.EXTRA_ROW_GATE=None
    holds=json.loads((SOURCE/'first-review-query-holds.json').read_text())
    package.HOLD_QUERIES.update(holds['query_hashes'])


def main(workers):
    if (STATE/'public-package-summary.json').exists():raise ValueError('lexical package already complete')
    proof=json.loads((PRIOR/'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:raise ValueError('prior release not verified')
    resources=json.loads(normalizer.RESOURCES.read_text())['resources'];reserve=8*1024**3
    used=normalizer.processed_bytes();free=shutil.disk_usage(base.WORK).free
    if used+reserve>resources['new_processed_cap_bytes'] or free<resources['minimum_free_disk_bytes']+reserve:
        raise ValueError('public budget/index stage resource bound')
    source=json.loads((SOURCE/'semantic-audit-v1/summary.json').read_text())
    compiled=json.loads((STATE/'compile-summary.json').read_text())
    selection=json.loads((STATE/'ppl-score-binding.json').read_text())
    if selection['competing_groups'] or selection['model_requests'] or selection['measured_ppl_rows']:
        raise ValueError('this package requires confirmed unscored singletons')
    policy={'driver_sha256':base.file_sha(Path(__file__)),
        'shared_full_array_budget_gate_sha256':base.file_sha(Path(package.__file__)),
        'shared_cross_release_identity_check_sha256':base.file_sha(Path(checks.__file__)),
        'source_audit_sha256':base.file_sha(SOURCE/'semantic-audit-v1/summary.json'),
        'source_holds_sha256':base.file_sha(SOURCE/'first-review-query-holds.json'),
        'framed_compile_policy_sha256':base.file_sha(STATE/'framed-compile-policy.json'),
        'compile_summary_sha256':base.file_sha(STATE/'compile-summary.json'),
        'ppl_binding_sha256':base.file_sha(STATE/'ppl-score-binding.json'),
        'prior_verified_publication_sha256':base.file_sha(PRIOR/'publication-final-commit.json'),
        'resource_policy_sha256':base.file_sha(normalizer.RESOURCES),'reserved_metadata_bytes':reserve,
        'processed_bytes_before':used,'free_bytes_before':free,
        'source_diagnosis_not_replayed_or_relabelled_as_winner_audit':True,
        'all_first_nonaccept_queries_held':True,'unscored_singletons_explicit':True,
        'no_world_truth_or_model_capability_claim':True}
    path=STATE/'package-policy-v1.json'
    if path.exists():
        saved=json.loads(path.read_text())
        for key in policy:
            if key not in ['processed_bytes_before','free_bytes_before'] and policy[key]!=saved[key]:
                raise ValueError('package binding changed')
    else:base.save(path,policy)
    paths=sorted((STATE/'compiled').glob('[0-9]*.json'))
    if len(paths)!=json.loads((STATE/'dedup-summary.json').read_text())['shards']:
        raise ValueError('compiled source coverage incomplete')
    counts=Counter();files=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,part in enumerate(pool.map(package.job,paths),1):
            counts.update(part['counts']);files.extend(part['files'])
            print(json.dumps({'packaged_shards':n,'total':len(paths),'accepted':counts['accepted']}),flush=True)
    if counts['seen']!=compiled['accepted']:raise ValueError('public gate coverage differs')
    index=STATE/'public-index.sqlite'
    if index.exists():raise ValueError('partial public index exists; preserve and diagnose')
    db=sqlite3.connect(index);db.execute('PRAGMA journal_mode=OFF')
    db.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)')
    for path in sorted((STATE/'public-package').glob('*.index.parquet')):
        rows=pq.read_table(path).to_pylist()
        db.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?)',[tuple(r[k] for k in
            ['record_id','family_id','query_hash','split','domain','source','upstream_source']) for r in rows]);db.commit()
    db.execute('ATTACH DATABASE ? AS previous',(str(PRIOR/'public-index.sqlite'),))
    unique=checks.check_cumulative_index(db)
    ids={r[0] for r in db.execute('SELECT record_id FROM records')};db.close();budget_ids=set()
    for item in files:
        if base.file_sha(PUBLIC/item['path'])!=item['sha256']:raise ValueError('public output changed')
        if item['config']=='budget-proof':
            values=pq.read_table(PUBLIC/item['path'],columns=['record_id']).column('record_id').to_pylist()
            if len(set(values))!=len(values) or budget_ids.intersection(values):raise ValueError('duplicated request proof')
            budget_ids.update(values)
        elif item['config']=='records':
            for raw in pq.read_table(PUBLIC/item['path'],columns=['source_refs_json']).column(0).to_pylist():
                refs=json.loads(raw);answer=refs['answer_selection']
                if answer!={'method':'only_eligible_answer','competitors':1,'ppl_measured':False}:
                    raise ValueError('unmeasured singleton label changed')
                counts['selection:only_eligible_answer']+=1
                counts['query_schema:'+refs['query_schema_id']]+=1
    if ids!=budget_ids or len(ids)!=counts['accepted']:raise ValueError('record/request proof coverage differs')
    if counts['accepted']!=unique['family_id']:raise ValueError('one eligible source base concept per independent family required')
    base.save(STATE/'public-package-summary.json',{
        'wave':WAVE,'status':'LOCAL_VERIFIED_NOT_YET_PUBLISHED','counts':dict(counts),'unique_counts':unique,'files':files,
        'source_audit':source['counts'],'source_audit_samples':len(source['results']),
        'winner_audit_performed':False,'winner_audit_reason':'One deterministic target per complete input; the original source diagnosis is retained.',
        'all_records_exact_finite_query_SQL_verified':True,'all_records_semantically_verified':False,
        'all_three_configs_exact_readback':True,'all_answer_free_qwen_chat_and_completion_budgets_measured':True,
        'all_native_budgets_include_4096_output_reserve':True,'all_previous_folds_preserved':True,
        'no_published_question_overlap':True,'first_nonaccept_query_holds_preserved':True,'all_budget_proof_identities_equal':True,
        'legacy_active_rows':0,'accepted_episodes_target':10000000,'full_goal_complete':False,
        'ppl_binding_sha256':base.file_sha(STATE/'ppl-score-binding.json'),
        'model_weights_trained':False,'model_efficacy_measured':False,
        'new_SFT_splits_entity_disjoint':False,
        'upstream_evaluation_split_not_claimed':True})
    print(json.dumps({'accepted':counts['accepted'],'unique':unique,'files':len(files)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    if not 1<=args.workers<=24:raise ValueError('CPU worker bound')
    main(args.workers)
