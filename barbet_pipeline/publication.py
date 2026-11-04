"""Append-only candidate-branch publication with anonymous byte verification.

The parent main branch and its training shards are never changed. A staging
marker remains until every newly published file is verified and documented.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib import request,parse

from .common import read_json,save_new,sha_file

REPO='OpenFormosa/barbet-long-context-sft'
PARENT='b88f4ff7171bb6fb8c1b775ae21de814ea2123fb'
BRANCH='semantic-curriculum-v3-20261104'


def verify_remote(item,commit):
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{commit}/'+parse.quote(item['remote'],safe='/')
    for attempt in range(3):
        try:
            h=hashlib.sha256();size=0
            req=request.Request(url,headers={'User-Agent':'barbet-data-readback/3'})
            with request.urlopen(req,timeout=180) as stream:
                while True:
                    chunk=stream.read(1024*1024)
                    if not chunk:break
                    size+=len(chunk);h.update(chunk)
            if size!=item['bytes'] or h.hexdigest()!=item['sha256']:
                raise ValueError('Anonymous published byte hash differs')
            return {'path':item['remote'],'sha256':h.hexdigest(),'bytes':size,'anonymous':True,'transport_attempts':attempt+1}
        except ValueError:
            raise
        except Exception:
            if attempt==2:raise
            time.sleep(2+attempt)


def stage_files(state,items,*,label,batch_size=64):
    from huggingface_hub import HfApi,CommitOperationAdd
    state=Path(state);api=HfApi()
    # Authenticated local credentials are used only by the official Hub client;
    # verification above never sends them to public/CDN endpoints.
    if api.repo_info(REPO,repo_type='dataset').sha!=PARENT:
        raise ValueError('Published main moved; no overwrite or automatic rebase')
    created=state/'branch-created.json'
    if not created.exists():
        api.create_branch(REPO,repo_type='dataset',branch=BRANCH,revision=PARENT,exist_ok=False)
        commit=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=PARENT,
            operations=[CommitOperationAdd('SEMANTIC_V3_STAGING_INCOMPLETE.txt',
                b'Candidate branch under construction. Existing main remains unchanged. Do not train from this branch until the final publication manifest is present.\n')],
            commit_message='Stage semantic curriculum v3 without changing the published release')
        save_new(created,{'parent':PARENT,'head':commit.oid,'branch':BRANCH})
    ledger=state/(label+'-plan.json')
    if not ledger.exists():save_new(ledger,{'items':items,'batch_size':batch_size,'parent_main':PARENT})
    elif read_json(ledger)['items']!=items:raise ValueError('Publication input changed')
    for i in range(0,len(items),batch_size):
        batch=items[i:i+batch_size];number=i//batch_size
        path=state/(label+'-batches')/f'{number:04d}.json'
        uploaded=state/(label+'-uploaded')/f'{number:04d}.json'
        if path.exists():
            r=read_json(path)
            if [x['remote'] for x in batch]!=r['paths']:raise ValueError('Readback ledger changed')
            continue
        for item in batch:
            if Path(item['local']).stat().st_size!=item['bytes'] or sha_file(item['local'])!=item['sha256']:
                raise ValueError('Local publication bytes changed')
        if uploaded.exists():
            commit_sha=read_json(uploaded)['head']
            if api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha!=commit_sha:
                raise ValueError('Unverified staging commit moved')
        else:
            head=api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha
            commit=api.create_commit(REPO,repo_type='dataset',revision=BRANCH,parent_commit=head,
                operations=[CommitOperationAdd(x['remote'],x['local']) for x in batch],num_threads=4,
                commit_message=f'Semantic v3 {label} {number+1}/{(len(items)+batch_size-1)//batch_size}')
            commit_sha=commit.oid
            save_new(uploaded,{'parent':head,'head':commit_sha,'paths':[x['remote'] for x in batch]})
        with ThreadPoolExecutor(8) as pool:
            verified=list(pool.map(lambda x:verify_remote(x,commit_sha),batch))
        save_new(path,{'head':commit_sha,'paths':[x['remote'] for x in batch],'verified':verified})
        print(json.dumps({'published':label,'batch':number+1,'files_verified':i+len(batch),'total':len(items)}),flush=True)
    return api.repo_info(REPO,repo_type='dataset',revision=BRANCH).sha


def index_items(corpus):
    summary=read_json(Path(corpus)/'summary.json')
    return [{'local':x['path'] if 'path' in x else x['output_path'],
             'remote':'semantic-v3/source-index/'+Path(x['path'] if 'path' in x else x['output_path']).name,
             'bytes':x['bytes'],'sha256':x['sha256']} for x in summary['files']]
