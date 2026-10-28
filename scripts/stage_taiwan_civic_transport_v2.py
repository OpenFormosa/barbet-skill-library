"""Larger bounded transport batches; original frozen row rewrite is unchanged."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from scripts import publish_taiwan_civic_v1 as release


def main():
    s=release.STATE;plan=release.read(s/'publication-plan.json')
    if release.sha(Path(release.__file__))!=plan['driver_sha256']:raise ValueError('Frozen payload driver changed')
    boundary=release.read(s/'transport-v1-controlled-boundary.json')
    receipts=sorted((s/'staging-batches').glob('*.json'))
    if len(receipts)!=boundary['verified_batches']:raise ValueError('Boundary receipt count changed')
    head=release.read(s/'staging-created.json')['head'];covered=[];all_files=[]
    for p in receipts:
        r=release.read(p)
        if r['parent']!=head or not r['all_changed_bytes_anonymously_verified']:raise ValueError('Old receipt chain')
        head=r['head'];covered.extend(r['paths']);all_files.extend(r['files'])
    if head!=boundary['head'] or covered!=[j['path'] for j in plan['jobs'][:len(covered)]]:raise ValueError('Prefix differs')
    api=HfApi();base_head=head
    v2=s/'staging-batches-v2';v2.mkdir(exist_ok=True)
    original_cache=release.CACHE;release.CACHE=s/'verified-staging-cache-v2';release.CACHE.mkdir(exist_ok=True)
    remaining=plan['jobs'][len(covered):];batches=[];current=[];size=0
    for job in remaining:
        if current and size+job['input_bytes']>1536*1024**2:batches.append(current);current=[];size=0
        current.append(job);size+=job['input_bytes']
    if current:batches.append(current)
    policy={'transport_driver_sha256':release.sha(Path(__file__)),'frozen_payload_driver_sha256':plan['driver_sha256'],
        'base_head':base_head,'verified_prefix_record_shards':len(covered),'batch_input_bound':1536*1024**2,
        'total_cache_bound':2*1024**3,'minimum_free_bytes':100*1024**3,'rewrite_workers':4,'upload_workers':8,'readback_workers':16}
    policy_path=s/'transport-v2-policy.json'
    if policy_path.exists() and release.read(policy_path)!=policy:raise ValueError('Transport policy changed')
    if not policy_path.exists():release.save(policy_path,policy)
    for i,batch in enumerate(batches):
        path=v2/f'{i:04d}.json'
        if path.exists():
            r=release.read(path)
            if r['parent']!=head or r['paths']!=[j['path'] for j in batch]:raise ValueError('V2 resume mismatch')
            head=r['head'];all_files.extend(r['files']);continue
        if api.repo_info(release.DATASET,repo_type='dataset',revision=release.BRANCH).sha!=head:raise ValueError('Branch changed')
        if shutil.disk_usage(release.DEST).free<100*1024**3+1536*1024**2:raise ValueError('Disk floor')
        with ThreadPoolExecutor(max_workers=4) as pool:files=[f for fs in pool.map(release.rewrite,batch) for f in fs]
        actual_cache=sum(p.stat().st_size for root in [original_cache,release.CACHE] for p in root.rglob('*') if p.is_file())
        if actual_cache>2*1024**3:raise ValueError('Combined cache exceeds bound')
        ops=[CommitOperationDelete(f['path']) if f.get('delete') else CommitOperationAdd(f['path'],release.CACHE/f['path']) for f in files]
        result=api.create_commit(release.DATASET,repo_type='dataset',revision=release.BRANCH,parent_commit=head,
            operations=ops,num_threads=8,commit_message=f'Civic quarantine transport v2 batch {i+1}/{len(batches)}')
        folder=s/'staging-readback-v2'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True);release.prior.bounded.STATE=s
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda f:release.prior.bounded.verify({k:f[k] for k in ['path','bytes','sha256']},result.oid,folder),[f for f in files if not f.get('delete')]))
        release.save(path,{'parent':head,'head':result.oid,'paths':[j['path'] for j in batch],'files':files,
            'all_changed_bytes_anonymously_verified':True,'transport_v2':True})
        head=result.oid;all_files.extend(files)
        for f in files:
            if not f.get('delete'):(release.CACHE/f['path']).unlink()
        print(json.dumps({'v2_batch':i+1,'total':len(batches),'verified_record_shards':len(covered)+sum(len(b) for b in batches[:i+1]),'total_record_shards':len(plan['jobs'])}),flush=True)
    release.save(s/'staging-complete.json',{'head':head,'files':all_files,'all_retained_values_and_arrays_equal':True,
        'transport_v2':True,'transport_policy_sha256':release.sha(policy_path)})


if __name__=='__main__':
    with (release.prior.parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
