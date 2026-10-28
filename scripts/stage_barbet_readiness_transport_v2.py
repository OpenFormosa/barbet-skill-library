"""Continue verified staging with larger transfer batches, same row/byte gates."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from pathlib import Path
import json
import shutil

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from scripts import publish_barbet_readiness_v1 as stage
from scripts import publish_barbet_only_v1 as parent
from scripts import resume_v26_recovery_readback_v2 as bounded


def main():
    state = stage.STATE; plan = stage.read(state / 'plan.json'); api = HfApi()
    policy_path = state / 'transport-v2-policy.json'
    policy = {'driver_sha256': stage.sha(Path(__file__)),
              'unchanged_row_subset_rewriter_sha256': stage.sha(Path(stage.__file__)),
              'initial_verified_boundary_sha256': stage.sha(state / 'transport-boundary-pause-v1.json'),
              'max_cache_bytes': 2 * 1024**3, 'minimum_free_bytes': 100 * 1024**3,
              'rewrite_workers': 4, 'upload_workers': 8, 'byte_readback_workers': 16,
              'max_source_bytes_per_batch': 7 * 1024**3 // 4, 'max_shard_pairs_per_batch': 8,
              'no_data_quality_or_verification_gate_change': True}
    if policy_path.exists() and stage.read(policy_path) != policy:
        raise ValueError('Transport v2 policy changed')
    if not policy_path.exists(): stage.save(policy_path, policy)
    head = stage.read(state / 'staging-initial.json')['head']; completed = []
    receipts = sorted((state / 'staging-batches').glob('*.json'))
    for receipt in receipts:
        saved = stage.read(receipt)
        if saved['parent'] != head: raise ValueError('Verified stage chain differs')
        for report in saved['reports']:
            for item in report['files']:
                if item.get('delete'): continue
                path = state / 'staged-readback' / saved['head'] / (hashlib.sha256(item['path'].encode()).hexdigest() + '.json')
                expected = {k: item[k] for k in ['path','bytes','sha256']} | {'revision':saved['head'],'anonymous':True}
                if stage.read(path) != expected: raise ValueError('Prior byte receipt differs')
                local = stage.CACHE / item['path']
                if local.exists():
                    if stage.sha(local) != item['sha256']: raise ValueError('Verified cache changed')
                    local.unlink()
        completed.extend(saved['source_paths']); head = saved['head']
    if completed != [j['path'] for j in plan['jobs'][:len(completed)]]:
        raise ValueError('Previously staged paths are not the frozen prefix')
    if api.repo_info(parent.DATASET, repo_type='dataset', revision=stage.BRANCH).sha != head:
        raise ValueError('Branch changed after the verified boundary')
    if any(p.is_file() for p in stage.CACHE.rglob('*')):
        raise ValueError('Unexpected unpublished cache files; preserve and inspect')
    batches = []; pending = []; size = 0
    for job in plan['jobs'][len(completed):]:
        if pending and (len(pending) >= 8 or size + job['input_bytes'] > policy['max_source_bytes_per_batch']):
            batches.append(pending); pending = []; size = 0
        pending.append(job); size += job['input_bytes']
    if pending: batches.append(pending)
    start = len(receipts); bounded.STATE = state
    for n, batch in enumerate(batches):
        if shutil.disk_usage(parent.base.WORK).free < policy['minimum_free_bytes'] + policy['max_cache_bytes']:
            raise ValueError('Bounded cache reserve unavailable')
        if api.repo_info(parent.DATASET, repo_type='dataset', revision=stage.BRANCH).sha != head:
            raise ValueError('Branch changed before next upload')
        with ThreadPoolExecutor(max_workers=4) as pool: reports = list(pool.map(stage.rewrite, batch))
        files = [f for report in reports for f in report['files']]
        size = sum(f.get('bytes',0) for f in files)
        if size > policy['max_cache_bytes']: raise ValueError('Generated cache exceeds unchanged bound')
        operations = [CommitOperationDelete(f['path']) if f.get('delete') else
                      CommitOperationAdd(f['path'], stage.CACHE / f['path']) for f in files]
        result = api.create_commit(parent.DATASET, repo_type='dataset', revision=stage.BRANCH,
            parent_commit=head, operations=operations, num_threads=8,
            commit_message=f'Continue identical row-subset quality correction, transport batch {start+n}')
        revision = result.oid; folder = state / 'staged-readback' / revision; folder.mkdir(parents=True,exist_ok=True)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda f: bounded.verify({k:f[k] for k in ['path','bytes','sha256']},revision,folder),
                          [f for f in files if not f.get('delete')]))
        record = {'head':revision,'parent':head,'source_paths':[j['path'] for j in batch],
                  'reports':reports,'all_new_payload_bytes_anonymously_verified':True,
                  'cache_bytes_reclaimed_after_verification':size,'transport_v2_policy_sha256':stage.sha(policy_path)}
        stage.save(state/'staging-batches'/f'{start+n:04d}.json',record)
        for item in files:
            if item.get('delete'): continue
            local=stage.CACHE/item['path']
            if not local.resolve().is_relative_to(stage.CACHE.resolve()) or stage.sha(local)!=item['sha256']:
                raise ValueError('Cache reclaim identity differs')
            local.unlink()
        completed.extend(record['source_paths']);head=revision
        print(json.dumps({'staged_shard_pairs':len(completed),'total_pairs':len(plan['jobs']),
                          'transport_v2_batches_done':n+1,'total_transport_v2_batches':len(batches),
                          'verified_batch_bytes':size}),flush=True)
    if completed != [j['path'] for j in plan['jobs']]: raise ValueError('Final changed-shard coverage differs')
    stage.save(state/'staging-complete.json',{'head':head,'batches':start+len(batches),
                                            'all_changed_shards_verified':True,'transport_v2':True})


if __name__ == '__main__':
    with (parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
