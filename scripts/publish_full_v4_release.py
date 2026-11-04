"""Finish the complete corpus branch, keeping a staging marker through checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from barbet_pipeline.common import read_json, save_new, sha_file
from barbet_pipeline.full_corpus import STATE, REPO as CODE
from barbet_pipeline.full_publication import BRANCH, MAIN, REPO
from barbet_pipeline.full_release import collect_complete_release
from barbet_pipeline.publication import verify_remote


def field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key)


def verify_tree(api, head, items):
    tree={f.path:f for f in api.list_repo_tree(REPO, repo_type='dataset',revision=head,path_in_repo='full-v4',recursive=True)
          if hasattr(f,'size')}
    card=api.get_paths_info(REPO,['README.md'],repo_type='dataset',revision=head)[0]
    tree['README.md']=card
    expected={i['remote']:i for i in items}
    if set(tree)!=set(expected):
        raise ValueError('Final v4 tree differs from the explicit file manifest')
    proofs=[]
    for path,item in expected.items():
        remote=tree[path]
        if remote.size!=item['bytes']:raise ValueError('Final tree byte size mismatch: '+path)
        if remote.lfs:
            if field(remote.lfs,'sha256')!=item['sha256']:raise ValueError('Final tree LFS identity mismatch: '+path)
            method='sha256_lfs'
        else:
            local=Path(item['local'])
            if sha_file(local)!=item['sha256']:raise ValueError('Small final metadata changed: '+path)
            content=local.read_bytes();git_hash=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if remote.blob_id!=git_hash:raise ValueError('Final tree Git identity mismatch: '+path)
            method='git_blob_sha1_plus_anonymous_sha256'
        proofs.append({'path':path,'bytes':item['bytes'],'sha256':item['sha256'],'tree_method':method})
    return proofs


def main():
    summary,statistics,data=collect_complete_release(STATE)
    package=STATE/'release-package';metadata=read_json(package/'upload-plan.json')['items']
    manifest=read_json(package/'release-manifest.json')
    if manifest['source_active_rows']!=10018967 or not manifest['all_sources_processed']:
        raise ValueError('Full processing is required before final publication')
    ledger=STATE/'publication/final';api=HfApi()
    if api.repo_info(REPO,repo_type='dataset').sha!=MAIN:raise ValueError('Preserved main has moved')
    uploaded=ledger/'metadata-uploaded.json'
    if uploaded.exists():head=read_json(uploaded)['head']
    else:
        for f in metadata:
            if sha_file(f['local'])!=f['sha256']:raise ValueError('Final metadata changed')
        parent=api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha
        result=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=parent,num_threads=4,
            operations=[CommitOperationAdd(f['remote'],f['local']) for f in metadata],
            commit_message='Document all 10,018,967 processed parents and explicit quality scopes')
        head=result.oid;save_new(uploaded,{'parent':parent,'head':head})
    completion_uploaded=ledger/'completion-uploaded.json'
    expected_head=read_json(completion_uploaded)['head'] if completion_uploaded.exists() else head
    if api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha!=expected_head:
        raise ValueError('Final staging branch moved unexpectedly')
    proof=ledger/'metadata-readback.json'
    if not proof.exists():
        with ThreadPoolExecutor(8) as pool:verified=list(pool.map(lambda f:verify_remote(f,head),metadata))
        save_new(proof,{'head':head,'verified':verified})
    proof=ledger/'complete-tree-check.json'
    if not proof.exists():save_new(proof,{'head':head,'files':verify_tree(api,head,data+metadata)})
    smoke=ledger/'anonymous-config-loads.json'
    if not smoke.exists():
        subprocess.run(['/root/barbet-context-sft-data/.venv/bin/python',str(CODE/'scripts/smoke_full_v4_hub.py'),head,str(smoke)],check=True)
    if not read_json(smoke)['all_configs_and_splits_loaded']:raise ValueError('Hub configuration smoke incomplete')
    complete=package/'publication-complete.json'
    if not complete.exists():
        save_new(complete,{'status':'FULL_CORPUS_PROCESSING_AND_PUBLICATION_COMPLETE',
            'verified_data_revision':head,'source_active_rows':10018967,'source_shards':summary['source_shards'],
            'exported_rows':summary['counts']['exported'],'quarantined_rows':summary['counts']['quarantined'],
            'all_new_data_anonymously_byte_verified':True,'all_final_tree_paths_and_identities_verified':True,
            'all_configurations_and_splits_anonymously_load':True,'implementation_commit':manifest['implementation_commit'],
            'manifest_sha256':sha_file(package/'release-manifest.json'),'preserved_main':MAIN,
            'private_oracles_and_heldout_not_published':True,'training_ready':False,'model_training':'NOT_RUN',
            'source_processing_complete_is_not_model_capability_success':True})
    if completion_uploaded.exists():final=read_json(completion_uploaded)['head']
    else:
        result=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=head,
            operations=[CommitOperationAdd('full-v4/publication-complete.json',str(complete)),
                        CommitOperationDelete('FULL_V4_STAGING_INCOMPLETE.txt')],
            commit_message='Complete exhaustive corpus publication after anonymous data and config checks')
        final=result.oid;save_new(completion_uploaded,{'parent':head,'head':final})
    item={'remote':'full-v4/publication-complete.json','local':str(complete),'sha256':sha_file(complete),'bytes':complete.stat().st_size}
    checked=verify_remote(item,final)
    if api.file_exists(REPO,'FULL_V4_STAGING_INCOMPLETE.txt',repo_type='dataset',revision=final):
        raise ValueError('Incomplete staging marker still exists')
    if api.repo_info(REPO,repo_type='dataset').sha!=MAIN:raise ValueError('Preserved main changed')
    receipt={'final_revision':final,'branch':BRANCH,'parent_metadata_revision':head,'main_preserved':MAIN,
             'source_rows':10018967,'exported_rows':summary['counts']['exported'],
             'quarantined_rows':summary['counts']['quarantined'],'complete_receipt_readback':checked}
    save_new(ledger/'final-receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
