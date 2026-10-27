"""Resumable anonymous full-byte readback with bounded persistent HTTP workers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import threading
import time
from urllib.parse import quote
import fcntl
import requests
from huggingface_hub import HfApi, CommitOperationAdd
from scripts.release_memory_sft_v2 import STATE, PUBLIC, DATASET, OLD, save, file_sha

LOCAL=threading.local()


def verify_file(item, revision, receipt_root):
    receipt=receipt_root/(hashlib.sha256(item['path'].encode()).hexdigest()+'.json')
    expected={**item,'revision':revision,'anonymous':True}
    if receipt.exists():
        if json.loads(receipt.read_text())!=expected:raise ValueError('readback receipt conflict')
        return expected
    if not hasattr(LOCAL,'session'):
        LOCAL.session=requests.Session();LOCAL.session.trust_env=False
    url=f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/'+quote(item['path'],safe='/')
    for attempt in range(6):
        try:
            count=0;h=hashlib.sha256()
            with LOCAL.session.get(url,stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(4*1024**2):count+=len(chunk);h.update(chunk)
            if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('public byte mismatch '+item['path'])
            part=receipt.with_suffix('.partial');part.write_text(json.dumps(expected,sort_keys=True)+'\n');os.replace(part,receipt)
            return expected
        except (requests.RequestException,ValueError):
            if attempt==5:raise
            time.sleep(min(2**attempt,30))


def main():
    data=json.loads((STATE/'publication-data-commit.json').read_text());revision=data['commit']
    manifest=json.loads((STATE/'publication-manifest-v1.json').read_text())
    if file_sha(STATE/'publication-manifest-v1.json')!=data['manifest_sha256']:raise ValueError('manifest changed')
    folder=STATE/'anonymous-readback-v2';receipts=folder/'files';receipts.mkdir(parents=True,exist_ok=True)
    save(folder/'policy.json',{'data_commit':revision,'http_workers':16,'trust_environment_credentials':False,
        'persistent_session_per_worker':True,'resumable_full_sha256_receipts':True,
        'supersedes':'v1 readback stopped for transport throughput; no semantic review repeated',
        'no_model_requests':True,'no_source_mutation':True})
    checks=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(verify_file,f,revision,receipts) for f in manifest]
        for n,future in enumerate(as_completed(futures),1):
            checks.append(future.result())
            if n%100==0 or n==len(manifest):print(json.dumps({'verified_files':n,'total':len(manifest)}),flush=True)
    api=HfApi(token=False);actual=set(api.list_repo_files(DATASET,repo_type='dataset',revision=revision))
    if actual!=({f['path'] for f in manifest}|{'.gitattributes'}):raise ValueError('public tree differs from manifest')
    full=folder/'summary.json';save(full,{'commit':revision,'all_files_verified':True,'checks':sorted(checks,key=lambda x:x['path'])})
    summary=json.loads((STATE/'public-package-summary.json').read_text())
    report={'dataset':DATASET,'data_commit':revision,'previous_release':OLD,'anonymous_full_file_sha256_readback':True,
        'verified_files':len(checks),'verified_bytes':sum(f['bytes'] for f in checks),'active_new_episodes':summary['counts']['accepted'],
        'active_legacy_rows':0,'target_accepted_episodes':10000000,'target_complete':False,
        'manifest_sha256':data['manifest_sha256'],'readback_receipt_sha256':file_sha(full),
        'readback_protocol':'resumable-anonymous-v2','model_efficacy_measured':False}
    quality=STATE/'postpublication-tool-interface-audit-v2.json'
    if quality.exists():
        q=json.loads(quality.read_text());count=q['counts']['unconfirmed_root_arguments']
        report['new_quality_guard_pending_next_data_release']={'flagged_rows':count,
            'currently_eligible_rows':summary['counts']['accepted']-count,
            'reason':'Arguments absent from enumerated function parameters despite permissive JSON Schema defaults',
            'old_payload_unchanged_in_this_metadata_commit':True}
    save(PUBLIC/'PUBLICATION.json',report)
    writer=HfApi()
    if writer.repo_info(DATASET,repo_type='dataset').sha!=revision:raise ValueError('head moved; reconcile before metadata commit')
    commit=writer.create_commit(DATASET,repo_type='dataset',
        operations=[CommitOperationAdd('PUBLICATION.json',PUBLIC/'PUBLICATION.json')],
        commit_message='Record complete anonymous byte verification of new v2 data',parent_commit=revision)
    f={'path':'PUBLICATION.json','bytes':(PUBLIC/'PUBLICATION.json').stat().st_size,'sha256':file_sha(PUBLIC/'PUBLICATION.json')}
    post=folder/'metadata';post.mkdir(exist_ok=True);verify_file(f,commit.oid,post)
    final={**report,'final_head':commit.oid,'publication_receipt_anonymous_verified':True}
    save(STATE/'publication-final-commit.json',final)
    save(STATE/'GOAL_PROGRESS.json',{'target':10000000,'published_new_episodes':summary['counts']['accepted'],
        'currently_eligible_after_new_guard':report.get('new_quality_guard_pending_next_data_release',{}).get('currently_eligible_rows',summary['counts']['accepted']),
        'phase':'wave_1_published_continuing_expansion_and_new_guard','goal_complete':False,'final_head':commit.oid})
    print(json.dumps({'head':commit.oid,'all_files_verified':True,'target_complete':False}),flush=True)


if __name__=='__main__':
    with (STATE/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
