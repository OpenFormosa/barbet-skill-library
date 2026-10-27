"""Bounded full anonymous readback and final new-only dataset target receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi,CommitOperationAdd
from scripts import build_memory_sft_v2 as base
from scripts import resume_v26_recovery_readback_v2 as bounded
from scripts.publish_memory_cumulative_v2 import DATASET


def main(state,dest):
    state=state.resolve();dest=dest.resolve()
    if not state.is_relative_to(base.WORK/'reports') or not dest.is_relative_to(base.WORK/'exports'):
        raise ValueError('release verification scope outside dataset work')
    plan=json.loads((state/'publication-plan.json').read_text());data=json.loads((state/'publication-data-commit.json').read_text())
    if base.file_sha(state/'publication-plan.json')!=data['plan_sha256']:raise ValueError('committed manifest changed')
    summary=json.loads((state/'summary.json').read_text());revision=data['commit']
    if summary['counts']['accepted']!=plan['accepted'] or summary['unique_counts']['record_id']!=plan['accepted']:
        raise ValueError('complete cumulative counts differ')
    for key in ['all_cumulative_folds_preserved','all_previous_source_contract_query_holds_preserved','prior_rows_preserved_byte_for_byte']:
        if not summary[key]:raise ValueError('cumulative release gate missing: '+key)
    for key in ['all_three_configs_exact_readback','all_answer_free_qwen_chat_and_completion_budgets_measured',
        'all_native_budgets_include_4096_output_reserve','all_previous_folds_preserved','no_published_question_overlap',
        'first_nonaccept_query_holds_preserved','all_budget_proof_identities_equal']:
        if not summary['new_wave'][key]:raise ValueError('new-wave release gate missing: '+key)
    protocol={'driver_sha256':base.file_sha(Path(__file__)),'bounded_transport_sha256':base.file_sha(Path(bounded.__file__)),
        'data_commit':revision,'plan_sha256':data['plan_sha256'],'http_workers':16,'max_attempts_per_file':3,
        'per_attempt_wall_limit_seconds':180,'socket_read_timeout_seconds':30,'chunk_bytes':65536,
        'anonymous':True,'all_manifest_files_downloaded_and_SHA256_checked':True,
        'no_prior_revision_receipts_substituted':True,'model_requests':0}
    pp=state/'anonymous-readback/protocol-v4.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('readback protocol changed')
    if not pp.exists():base.save(pp,protocol)
    expected_tree={f['path'] for f in plan['manifest']}|{'.gitattributes'}
    if set(HfApi(token=False).list_repo_files(DATASET,repo_type='dataset',revision=revision))!=expected_tree:
        raise ValueError('public tree differs before full byte readback')
    folder=state/'anonymous-readback/files';folder.mkdir(parents=True,exist_ok=True);bounded.STATE=state
    def verify_item(item,rev,receipts):
        bounded.verify(item,rev,receipts)
        result=json.loads((receipts/(hashlib.sha256(item['path'].encode()).hexdigest()+'.json')).read_text())
        if result!={**item,'revision':rev,'anonymous':True}:raise ValueError('full readback receipt differs')
        return result
    checks=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        tasks=[pool.submit(verify_item,item,revision,folder) for item in plan['manifest']]
        for n,future in enumerate(as_completed(tasks),1):
            checks.append(future.result())
            if n%100==0 or n==len(tasks):print(json.dumps({'verified_files':n,'total':len(tasks)}),flush=True)
    if set(HfApi(token=False).list_repo_files(DATASET,repo_type='dataset',revision=revision))!=expected_tree:
        raise ValueError('public tree differs after full byte readback')
    base.save(state/'anonymous-readback/summary.json',{'commit':revision,'checks':sorted(checks,key=lambda r:r['path']),
        'all_files_verified':True,'readback_protocol_sha256':base.file_sha(pp)})
    complete=plan['accepted']>=10000000
    report={'dataset':DATASET,'data_commit':revision,'parent':data['parent'],'active_new_episodes':plan['accepted'],
        'active_legacy_rows':0,'accepted_episode_target':10000000,'goal_complete':complete,
        'unique_normalized_first_queries':summary['unique_counts']['query_hash'],
        'unique_task_families':summary['unique_counts']['family_id'],
        'anonymous_full_file_sha256_readback':True,'verified_files':len(checks),'verified_bytes':sum(r['bytes'] for r in checks),
        'library_commit':plan['library_commit'],'readback_sha256':base.file_sha(state/'anonymous-readback/summary.json'),
        'readback_protocol_sha256':base.file_sha(pp),'answer_selection':plan['answer_selection'],
        'model_weights_trained':False,'model_efficacy_measured':False,
        'completion_scope':'Dataset engineering and publication count target only; no new trained-model capability or 9B-equivalence claim.',
        'new_declared_argument_holds_removed':summary['counts'].get('excluded:new_declared_argument_guard',0)}
    base.save(dest/'PUBLICATION.json',report);api=HfApi();receipt_path=state/'publication-metadata-commit.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['data_commit']!=revision or receipt['publication_sha256']!=base.file_sha(dest/'PUBLICATION.json'):
            raise ValueError('metadata resume binding changed')
        final=receipt['final_head']
    else:
        if api.repo_info(DATASET,repo_type='dataset').sha!=revision:raise ValueError('remote head changed before proof publication')
        commit=api.create_commit(DATASET,repo_type='dataset',operations=[CommitOperationAdd('PUBLICATION.json',dest/'PUBLICATION.json')],
            parent_commit=revision,commit_message='Record full anonymous byte verification and actual new-only dataset count target status')
        final=commit.oid;base.save(receipt_path,{'data_commit':revision,'final_head':final,'publication_sha256':base.file_sha(dest/'PUBLICATION.json')})
    post=state/'anonymous-readback/metadata';post.mkdir(exist_ok=True)
    verify_item({'path':'PUBLICATION.json','bytes':(dest/'PUBLICATION.json').stat().st_size,'sha256':base.file_sha(dest/'PUBLICATION.json')},final,post)
    base.save(state/'publication-final-commit.json',{**report,'final_head':final,'publication_receipt_anonymous_verified':True})
    base.save(base.WORK/'reports/memory-harness-v2/GOAL_PROGRESS.json',{'target':10000000,
        'published_eligible_new_episodes':plan['accepted'],'phase':plan['goal_phase'],'goal_complete':complete,'final_head':final})
    print(json.dumps({'head':final,'accepted':plan['accepted'],'dataset_target_complete':complete}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path,required=True);parser.add_argument('--dest',type=Path,required=True);args=parser.parse_args()
    with (base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(args.state,args.dest)
