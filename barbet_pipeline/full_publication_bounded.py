"""One remote publisher; discard only byte-verified task-created Parquet files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from huggingface_hub import HfApi,CommitOperationAdd
from .common import read_json,save_new,sha_file
from .publication import REPO
from .anonymous_readback import verify_remote
from .full_corpus import STATE,OUTPUT

MAIN='b88f4ff7171bb6fb8c1b775ae21de814ea2123fb'
BASE='4b38655b336aa2d0c01f6f0047701e7eeec23ed6'
BRANCH='full-corpus-v4-20261104'

def publish_group(number,results):
    state=STATE/'publication';api=HfApi();path=state/'groups'/f'{number:04d}.json'
    if path.exists():return read_json(path)
    assert api.repo_info(REPO,repo_type='dataset').sha==MAIN
    if not (state/'branch-created.json').exists():
        api.create_branch(REPO,repo_type='dataset',branch=BRANCH,revision=BASE,exist_ok=False)
        commit=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=BASE,
            operations=[CommitOperationAdd('FULL_V4_STAGING_INCOMPLETE.txt',b'Full corpus reprocessing in progress. No final v4 training admission or complete release is claimed.\n')],
            commit_message='Stage exhaustive full-corpus v4 reprocessing; preserve prior releases')
        save_new(state/'branch-created.json',{'base':BASE,'head':commit.oid,'branch':BRANCH})
    files=[f for r in results for f in r['files']]
    plan=state/'plans'/f'{number:04d}.json'
    if not plan.exists():save_new(plan,{'files':files,'source_paths':[r['source_path'] for r in results]})
    else:assert read_json(plan)['files']==files
    uploaded=state/'uploaded'/f'{number:04d}.json'
    if not uploaded.exists():
        for f in files:
            assert Path(f['local']).stat().st_size==f['bytes'] and sha_file(f['local'])==f['sha256']
        head=api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha
        commit=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=head,
            operations=[CommitOperationAdd(f['remote'],f['local']) for f in files],num_threads=8,
            commit_message=f'Full-corpus v4 verified source group {number+1}')
        save_new(uploaded,{'head':commit.oid,'parent':head})
    commit=read_json(uploaded)['head']
    verified=[]
    for start in range(0,len(files),32):
        proof=state/'readback'/f'{number:04d}-{start:05d}.json'
        if proof.exists():batch=read_json(proof)['verified']
        else:
            with ThreadPoolExecutor(8) as pool:batch=list(pool.map(lambda item:verify_remote(item,commit),files[start:start+32]))
            save_new(proof,{'head':commit,'verified':batch})
        verified.extend(batch)
    result={'head':commit,'files':files,'verified_count':len(verified),'all_anonymous_hashes_match':True,
        'processed_rows':sum(r['counts']['active_rows_processed'] for r in results)}
    save_new(path,result)
    evicted=[]
    for f in files:
        p=Path(f['local']).resolve()
        if not p.is_relative_to(OUTPUT.resolve()) or p.suffix!='.parquet':raise ValueError('Refuse deletion outside generated outputs')
        if p.exists():
            assert sha_file(p)==f['sha256'];p.unlink();evicted.append(f['remote'])
    save_new(state/'offloaded'/f'{number:04d}.json',{'verified_commit':commit,'paths':evicted,
        'original_sources_and_receipts_untouched':True,'remote_bytes_verified_before_local_removal':True})
    return result
