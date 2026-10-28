"""Verify HF's exact automatic large-JSON LFS rule without changing payloads.

The frozen v1 verifier and first failure remain intact. The one extra transport
metadata change is checked byte for byte, bound to its Git blob, and added to
the anonymous verification manifest. It is never called an unchanged object.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import requests
from huggingface_hub import HfApi,CommitOperationAdd
from scripts import publish_value_alignment_v1 as release


def fetch_attributes(revision,tree):
    session=requests.Session();session.trust_env=False
    response=session.get(f'https://huggingface.co/datasets/{release.DATASET}/resolve/{revision}/.gitattributes',timeout=60)
    response.raise_for_status();raw=response.content
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    info=tree['.gitattributes']
    if info['blob_id']!=blob or info['bytes']!=len(raw):raise ValueError('Anonymous attributes differ from the pinned Git object')
    return raw


def main():
    state=release.STATE;plan=release.read(state/'publication-plan.json');manifest=release.read(state/'publication-manifest.json');data=release.read(state/'publication-data-commit.json')
    if release.sha(state/'publication-manifest.json')!=data['manifest_sha256']:raise ValueError('Frozen original manifest changed')
    if release.sha(Path(release.__file__))!=plan['driver_sha256']:raise ValueError('Frozen v1 implementation changed')
    if (state/'publication-final-commit.json').exists():raise ValueError('Final proof already exists')
    api=HfApi();head=data['commit']
    if api.repo_info(release.DATASET,repo_type='dataset').sha!=head:raise ValueError('Main changed')
    tree=release.prior.prior.parent.tree(head);old=release.read(state/'parent-tree.json');stage=release.prior.prior.parent.tree(manifest['staging_head'])
    parent_raw=fetch_attributes(plan['parent'],old);current_raw=fetch_attributes(head,tree)
    rule=b'reports/value-alignment-v1/held-queries.json filter=lfs diff=lfs merge=lfs -text\n'
    if not parent_raw.endswith(b'\n') or current_raw!=parent_raw+rule:raise ValueError('Attributes changed beyond the single authorized large held-query JSON path')
    source=next(f for f in manifest['uploaded'] if f['path']=='reports/value-alignment-v1/held-queries.json')
    if source['bytes']<=10_000_000:raise ValueError('Unexpected large-file trigger')
    attributes={'path':'.gitattributes','bytes':len(current_raw),'sha256':hashlib.sha256(current_raw).hexdigest()}
    public_files=manifest['files']+[attributes]
    if len({f['path'] for f in public_files})!=len(public_files):raise ValueError('Duplicate manifest path')
    uploaded={f['path'] for f in manifest['uploaded']};changed={f['path'] for f in manifest['changed']};expected={f['path'] for f in public_files}
    if set(tree)!=expected or any('minicpm5_tokenized' in p for p in tree):raise ValueError('Unexpected public tree')
    unchanged=expected-uploaded-changed-{'.gitattributes'}
    for p in expected-uploaded-{'.gitattributes'}:
        if tree[p]!=(stage if p in changed else old)[p]:raise ValueError('Payload object identity mismatch: '+p)
    receipt={'transport_driver_sha256':release.sha(Path(__file__)),'transport_library_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=release.REPO,text=True).strip(),
        'frozen_v1_driver_sha256':plan['driver_sha256'],'original_manifest_sha256':data['manifest_sha256'],
        'first_v1_failure_log_sha256':release.sha(state/'publication-pipeline.log'),'data_commit':head,
        'parent_attributes_sha256':hashlib.sha256(parent_raw).hexdigest(),'attributes':attributes,
        'exact_automatic_added_rule':rule.decode().strip(),'only_this_transport_metadata_difference_allowed':True,
        'training_payload_target_split_score_or_review_changes':0}
    release.save(state/'transport-v2-attributes-verification.json',receipt)
    release.save(state/'publication-manifest-v2.json',{'files':public_files,'original_manifest_sha256':data['manifest_sha256'],'additional_transport_metadata':[attributes]})
    folder=state/'main-new-file-readback-v2';folder.mkdir(exist_ok=True);release.prior.prior.bounded.STATE=state
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda f:release.prior.prior.bounded.verify(f,head,folder),manifest['uploaded']+[attributes]))
    summary=release.read(state/'summary.json');admission=release.read(state/'admission.json');previous=release.read(release.prior.STATE/'publication-final-commit.json');counts=summary['counts']
    proof={**previous,'parent':plan['parent'],'data_commit':head,'active_new_episodes':counts['accepted'],
        'unique_normalized_first_queries':summary['unique_counts']['query_hash'],'unique_task_families':summary['unique_counts']['family_id'],
        'train_episodes':counts['split:train'],'validation_episodes':counts['split:validation'],
        'held_queries':summary['cumulative_held_queries'],'removed_episodes':summary['cumulative_removed_rows'],
        'new_civic_episodes':summary['retained_civic_episodes'],'new_value_gate_held_queries':summary['new_held_queries'],'new_value_gate_held_rows':summary['new_held_rows'],
        'value_policy':'VALUE_ALIGNMENT_POLICY.json','value_gate_report':release.FOLDER+'/report.html',
        'parent_active_rows_topic_screened':10321901,'sensitive_candidates':admission['candidate_rows'],
        'semantic_first_reviews':admission['semantic_first_reviews'],'semantic_counts':admission['semantic_counts'],
        'values_exhaustive_semantic_certification':False,'model_efficacy_measured':False,'staging_head':manifest['staging_head'],
        'changed_payload_files_anonymously_verified':len(manifest['changed']),'retained_unchanged_objects_verified':len(unchanged),
        'verified_files':len(public_files),'verified_bytes':sum(f['bytes'] for f in public_files),'library_commit':plan['library_commit'],
        'goal_complete':summary['quantity_target_met'],'automatic_attributes_change_verification':receipt,
        'verification_method':'Prior verified unchanged objects; exact retained-row subset staging anonymous SHA256 and main-copy identities; full new metadata plus one byte-verified automatic LFS attribute rule',
        'frozen_v1_first_attributes_mismatch_preserved':True}
    proof.pop('final_head',None);proof.pop('publication_receipt_anonymous_verified',None)
    release.save(release.DEST/'PUBLICATION.json',proof)
    if api.repo_info(release.DATASET,repo_type='dataset').sha!=head:raise ValueError('Main changed before proof publication')
    commit=api.create_commit(release.DATASET,repo_type='dataset',parent_commit=head,
        operations=[CommitOperationAdd('PUBLICATION.json',release.DEST/'PUBLICATION.json')],
        commit_message='Verify values release and the exact HF-added LFS rule; preserve first transport mismatch')
    item={'path':'PUBLICATION.json','bytes':(release.DEST/'PUBLICATION.json').stat().st_size,'sha256':release.sha(release.DEST/'PUBLICATION.json')}
    folder=state/'proof-readback-v2';folder.mkdir(exist_ok=True);release.prior.prior.bounded.verify(item,commit.oid,folder)
    final=release.prior.prior.parent.tree(commit.oid)
    if set(final)!=expected|{'PUBLICATION.json'} or any(final[p]!=tree[p] for p in tree):raise ValueError('Proof publication changed data or metadata')
    proof.update(final_head=commit.oid,publication_receipt_anonymous_verified=True)
    release.save(state/'publication-final-commit.json',proof);release.save(release.prior.prior.parent.base.WORK/'reports/memory-harness-v2/CURRENT_PUBLICATION.json',proof)
    print(json.dumps({'final_head':commit.oid,'active':counts['accepted'],'verified_files':len(public_files),'automatic_attributes_rule_verified':True}),flush=True)


if __name__=='__main__':
    with (release.prior.prior.parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
