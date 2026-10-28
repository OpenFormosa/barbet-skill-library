"""Verified bounded-cache subset publication of the explicit values gate."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationCopy,CommitOperationDelete
from scripts import publish_taiwan_civic_v1 as prior

STATE=prior.prior.parent.base.WORK/'reports/value-alignment-v1'
DEST=prior.prior.parent.base.WORK/'exports/value-alignment-v1'
CACHE=STATE/'verified-staging-cache'
FOLDER='reports/value-alignment-v1'
BRANCH='value-alignment-20261028-v1'
DATASET=prior.DATASET
read,save,sha=prior.read,prior.save,prior.sha
RAW=prior.prior.parent.parent.DEST
REPO=prior.prior.parent.base.REPO
BOUND=1536*1024**2


def scan_new(item,holds):
    table=pq.read_table(prior.DEST/item['path']);selected=pc.is_in(table['query_hash'],value_set=pa.array(holds));rows=table.filter(selected).to_pylist();counts=Counter()
    for row in rows:counts.update(prior.prior.removed_stats(row))
    return {'path':item['path'],'excluded':len(rows),'exclude_indices':np.flatnonzero(np.asarray(selected)).tolist(),'counts':dict(counts),'removed_ids':[r['record_id'] for r in rows]}


def prepare():
    DEST.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (STATE/'publication-plan.json').exists():raise ValueError('Publication plan already frozen')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit code before freezing')
    previous=read(prior.STATE/'publication-final-commit.json');parent=previous['final_head']
    if HfApi(token=False).repo_info(DATASET,repo_type='dataset').sha!=parent:raise ValueError('Parent head changed')
    admission=read(STATE/'admission.json');holds=admission['query_hashes'];old=read(prior.prior.STATE/'plan.json')
    previous_jobs={j['path']:j for j in old['jobs']}
    previous_jobs.update({j['path']:j for j in read(prior.STATE/'publication-plan.json')['jobs']})
    raw_manifest={f['path']:f for f in old['base_manifest']};jobs=[];removed=Counter()
    records=[f for f in old['base_manifest'] if '/records/' in f['path']]
    with ThreadPoolExecutor(8) as pool:
        for n,part in enumerate(pool.map(lambda item:prior.prior.scan(item,holds),records),1):
            if part['excluded']:
                part.pop('groups',None);removed.update(part['counts']);part['new_exclude_indices']=part['exclude_indices']
                part['exclude_indices']=sorted(set(part['exclude_indices'])|set(previous_jobs.get(part['path'],{}).get('exclude_indices',[])))
                part['root']=str(RAW);part['inputs']=[raw_manifest[p] for p in [part['path'],part['path'].replace('/records/','/tokenized/')]]
                part['input_bytes']=sum(f['bytes'] for f in part['inputs']);jobs.append(part)
            if n%200==0:print(json.dumps({'prepared_shards':n,'removed_rows':removed['accepted']}),flush=True)
    cohort=read(prior.STATE/'new-cohort-summary.json');new_manifest={f['path']:f for f in cohort['files']}
    civic_removed=0
    for item in cohort['files']:
        if '/records/' not in item['path']:continue
        part=scan_new(item,holds)
        if part['excluded']:
            civic_removed+=part['excluded'];removed.update(part['counts']);part['new_exclude_indices']=part['exclude_indices']
            part['root']=str(prior.DEST);part['inputs']=[new_manifest[p] for p in [part['path'],part['path'].replace('/records/','/tokenized/')]]
            part['input_bytes']=sum(f['bytes'] for f in part['inputs']);jobs.append(part)
    counts=Counter(read(prior.STATE/'summary.json')['counts']);counts.subtract(removed)
    if any(v<0 for v in counts.values()):raise ValueError('Negative count')
    db=sqlite3.connect(f'file:{prior.prior.parent.parent.STATE/"public-index.sqlite"}?mode=ro',uri=True);db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TEMP TABLE newheld(query_hash TEXT PRIMARY KEY)');db.executemany('INSERT INTO newheld VALUES(?)',[(h,) for h in holds])
    old_holds=set(read(prior.prior.audit.STATE/'new-query-holds-v1.json')['query_hashes'])|set(read(prior.STATE/'existing-topic-triage.json')['held_query_hashes'])
    if old_holds&set(holds):raise ValueError('Old holds reentered')
    db.execute('CREATE TEMP TABLE allheld(query_hash TEXT PRIMARY KEY)');db.executemany('INSERT INTO allheld VALUES(?)',[(h,) for h in old_holds|set(holds)])
    n=db.execute('SELECT COUNT(*) FROM records WHERE query_hash IN (SELECT query_hash FROM newheld)').fetchone()[0]
    if n+civic_removed!=removed['accepted']:raise ValueError('Whole-query count mismatch')
    found={q for q, in db.execute('SELECT DISTINCT query_hash FROM records WHERE query_hash IN (SELECT query_hash FROM newheld)')}
    for item in cohort['files']:
        if '/records/' in item['path']:
            found.update(q for q in pq.read_table(prior.DEST/item['path'],columns=['query_hash'])['query_hash'].to_pylist() if q in set(holds))
    if found!=set(holds):raise ValueError('An admitted query hold is absent from the parent')
    family_count=db.execute('SELECT COUNT(DISTINCT family_id) FROM records WHERE query_hash NOT IN (SELECT query_hash FROM allheld)').fetchone()[0];db.close()
    retained_civic=[]
    for item in cohort['files']:
        if '/records/' in item['path']:
            retained_civic += [r for r in pq.read_table(prior.DEST/item['path'],columns=['record_id','query_hash','family_id']).to_pylist() if r['query_hash'] not in set(holds)]
    unique={'record_id':counts['accepted'],'query_hash':read(prior.STATE/'summary.json')['unique_counts']['query_hash']-len(holds),'family_id':family_count+len({r['family_id'] for r in retained_civic})}
    for prefix in ['source:','domain:','length_bucket:','selection:']:
        if sum(v for k,v in counts.items() if k.startswith(prefix))!=counts['accepted']:raise ValueError('Aggregate partition differs')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    manifest=read(prior.STATE/'publication-manifest.json')['files']
    save(STATE/'publication-plan.json',{'parent':parent,'manifest':manifest,'jobs':jobs,'driver_sha256':sha(Path(__file__)),'metadata_driver_sha256':sha(REPO/'scripts/value_alignment_metadata_v1.py'),'admission_sha256':sha(STATE/'admission.json'),'library_commit':code,'minimum_free_bytes':100*1024**3,'cache_bound':2*1024**3})
    save(STATE/'summary.json',{'counts':dict(counts),'unique_counts':unique,'removed_counts':dict(removed),'new_held_queries':len(holds),'new_held_rows':removed['accepted'],'cumulative_held_queries':previous['held_queries']+len(holds),'cumulative_removed_rows':previous['removed_episodes']+removed['accepted'],'retained_civic_episodes':len(retained_civic),'new_training_rows':0,'all_retained_targets_scores_splits_and_arrays_unchanged':True,'no_runner_up_substitution':True,'quantity_target_met':counts['accepted']>=10_000_000})
    save(STATE/'parent-tree.json',prior.prior.parent.tree(parent))
    print(json.dumps({'prepared':True,'changed_record_shards':len(jobs),'removed':removed['accepted'],'active':counts['accepted']}),flush=True)


def rewrite(job):
    output=[]
    for item in job['inputs']:
        source=Path(job['root'])/item['path']
        if sha(source)!=item['sha256']:raise ValueError('Frozen input hash changed')
        table=pq.read_table(source);keep=np.ones(len(table),dtype=bool);keep[job['exclude_indices']]=False;expected=table.filter(pa.array(keep))
        target=CACHE/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if not len(expected):output.append({'path':item['path'],'delete':True});continue
        pq.write_table(expected,target,compression='zstd',compression_level=3)
        if not pq.read_table(target).equals(expected):raise ValueError('Retained row values or arrays changed')
        output.append({'path':item['path'],'bytes':target.stat().st_size,'sha256':sha(target),'rows':len(expected)})
    return output


def stage():
    plan=read(STATE/'publication-plan.json')
    if plan['driver_sha256']!=sha(Path(__file__)) or plan['admission_sha256']!=sha(STATE/'admission.json'):raise ValueError('Frozen publisher/admission changed')
    api=HfApi();receipts=STATE/'staging-batches';receipts.mkdir(exist_ok=True)
    if not (STATE/'staging-created.json').exists():
        if api.repo_info(DATASET,repo_type='dataset').sha!=plan['parent']:raise ValueError('Main changed')
        api.create_branch(DATASET,repo_type='dataset',branch=BRANCH,revision=plan['parent'],exist_ok=False)
        commit=api.create_commit(DATASET,repo_type='dataset',revision=BRANCH,parent_commit=plan['parent'],operations=[CommitOperationDelete('PUBLICATION.json'),CommitOperationAdd('STAGING_INCOMPLETE.txt',b'Incomplete values-filtered staging branch. Use main until final publication.\n')],commit_message='Mark values revision staging incomplete')
        save(STATE/'staging-created.json',{'head':commit.oid})
    batches=[];batch=[];size=0
    for job in plan['jobs']:
        if job['input_bytes']>BOUND:raise ValueError('Single input exceeds batch bound')
        if batch and size+job['input_bytes']>BOUND:batches.append(batch);batch=[];size=0
        batch.append(job);size+=job['input_bytes']
    if batch:batches.append(batch)
    head=read(STATE/'staging-created.json')['head'];all_files=[]
    for i,batch in enumerate(batches):
        receipt=receipts/f'{i:04d}.json'
        if receipt.exists():
            r=read(receipt)
            if r['parent']!=head or r['paths']!=[j['path'] for j in batch] or not r['all_changed_bytes_anonymously_verified']:raise ValueError('Staging receipt chain differs')
            # An interruption after writing a verified receipt may leave this
            # task's reclaimable cache. Never touch an unverified source file.
            for f in r['files']:
                cached=CACHE/f['path']
                if not f.get('delete') and cached.exists():
                    if sha(cached)!=f['sha256']:raise ValueError('Verified cache changed')
                    cached.unlink()
            head=r['head'];all_files.extend(r['files']);continue
        if api.repo_info(DATASET,repo_type='dataset',revision=BRANCH).sha!=head:raise ValueError('Staging changed')
        if shutil.disk_usage(DEST).free<100*1024**3+BOUND:raise ValueError('Disk floor')
        with ThreadPoolExecutor(4) as pool:files=[f for part in pool.map(rewrite,batch) for f in part]
        if sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())>2*1024**3:raise ValueError('Cache cap')
        ops=[CommitOperationDelete(f['path']) if f.get('delete') else CommitOperationAdd(f['path'],CACHE/f['path']) for f in files]
        commit=api.create_commit(DATASET,repo_type='dataset',revision=BRANCH,parent_commit=head,operations=ops,num_threads=8,commit_message=f'Values admission subset batch {i+1}/{len(batches)}')
        folder=STATE/'staging-readback'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True);prior.prior.bounded.STATE=STATE
        with ThreadPoolExecutor(16) as pool:list(pool.map(lambda f:prior.prior.bounded.verify({k:f[k] for k in ['path','bytes','sha256']},commit.oid,folder),[f for f in files if not f.get('delete')]))
        save(receipt,{'parent':head,'head':commit.oid,'paths':[j['path'] for j in batch],'files':files,'all_changed_bytes_anonymously_verified':True})
        head=commit.oid;all_files.extend(files)
        for f in files:
            if not f.get('delete'):(CACHE/f['path']).unlink()
        print(json.dumps({'batch':i+1,'total_batches':len(batches),'files':len(all_files),'verified':True}),flush=True)
    save(STATE/'staging-complete.json',{'head':head,'files':all_files,'all_retained_values_and_arrays_equal':True})


def publish():
    from scripts.value_alignment_metadata_v1 import metadata
    if (STATE/'publication-data-commit.json').exists():raise ValueError('Already published')
    plan=read(STATE/'publication-plan.json');stage=read(STATE/'staging-complete.json');api=HfApi()
    if sha(Path(__file__))!=plan['driver_sha256'] or sha(REPO/'scripts/value_alignment_metadata_v1.py')!=plan['metadata_driver_sha256']:raise ValueError('Frozen implementation changed')
    if api.repo_info(DATASET,repo_type='dataset').sha!=plan['parent'] or api.repo_info(DATASET,repo_type='dataset',revision=BRANCH).sha!=stage['head']:raise ValueError('Remote head changed')
    metadata(plan);manifest={f['path']:f for f in plan['manifest']};ops=[]
    for f in stage['files']:
        if f.get('delete'):manifest.pop(f['path']);ops.append(CommitOperationDelete(f['path']))
        else:manifest[f['path']]={k:f[k] for k in ['path','bytes','sha256']};ops.append(CommitOperationCopy(f['path'],f['path'],src_revision=stage['head']))
    uploaded=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        item={'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':sha(p)};manifest[item['path']]=item;uploaded.append(item);ops.append(CommitOperationAdd(item['path'],p))
    ops.append(CommitOperationDelete('PUBLICATION.json'))
    save(STATE/'publication-manifest.json',{'files':sorted(manifest.values(),key=lambda f:f['path']),'uploaded':uploaded,'changed':stage['files'],'staging_head':stage['head']})
    commit=api.create_commit(DATASET,repo_type='dataset',parent_commit=plan['parent'],operations=ops,num_threads=8,commit_message='Apply all-source values admission: verified bounded queries, contextual reviews and whole-query quarantine')
    save(STATE/'publication-data-commit.json',{'commit':commit.oid,'manifest_sha256':sha(STATE/'publication-manifest.json')});print(json.dumps({'data_commit':commit.oid}),flush=True)


def verify():
    plan=read(STATE/'publication-plan.json');manifest=read(STATE/'publication-manifest.json');data=read(STATE/'publication-data-commit.json')
    if data['manifest_sha256']!=sha(STATE/'publication-manifest.json'):raise ValueError('Manifest changed')
    head=data['commit'];tree=prior.prior.parent.tree(head);old=read(STATE/'parent-tree.json');stage=prior.prior.parent.tree(manifest['staging_head'])
    uploaded={f['path'] for f in manifest['uploaded']};changed={f['path'] for f in manifest['changed']};expected={f['path'] for f in manifest['files']}|{'.gitattributes'}
    if set(tree)!=expected or any('minicpm5_tokenized' in p for p in tree):raise ValueError('Unexpected published tree')
    for p in expected-uploaded:
        if tree[p]!=(stage if p in changed else old)[p]:raise ValueError('Object identity mismatch: '+p)
    folder=STATE/'main-new-file-readback';folder.mkdir(exist_ok=True);prior.prior.bounded.STATE=STATE
    with ThreadPoolExecutor(16) as pool:list(pool.map(lambda f:prior.prior.bounded.verify(f,head,folder),manifest['uploaded']))
    s=read(STATE/'summary.json');a=read(STATE/'admission.json');previous=read(prior.STATE/'publication-final-commit.json')
    proof={**previous,'parent':plan['parent'],'data_commit':head,'active_new_episodes':s['counts']['accepted'],'unique_normalized_first_queries':s['unique_counts']['query_hash'],'unique_task_families':s['unique_counts']['family_id'],'train_episodes':s['counts']['split:train'],'validation_episodes':s['counts']['split:validation'],'held_queries':s['cumulative_held_queries'],'removed_episodes':s['cumulative_removed_rows'],'new_civic_episodes':s['retained_civic_episodes'],'new_value_gate_held_queries':s['new_held_queries'],'new_value_gate_held_rows':s['new_held_rows'],'value_policy':'VALUE_ALIGNMENT_POLICY.json','value_gate_report':FOLDER+'/report.html','all_active_rows_topic_screened':10321901,'sensitive_candidates':a['candidate_rows'],'semantic_first_reviews':a['semantic_first_reviews'],'semantic_counts':a['semantic_counts'],'values_exhaustive_semantic_certification':False,'model_efficacy_measured':False,'staging_head':manifest['staging_head'],'changed_payload_files_anonymously_verified':len(manifest['changed']),'retained_unchanged_objects_verified':len(expected-uploaded-changed),'verified_files':len(manifest['files']),'verified_bytes':sum(f['bytes'] for f in manifest['files']),'library_commit':plan['library_commit'],'goal_complete':s['quantity_target_met']}
    proof.pop('final_head',None);proof.pop('publication_receipt_anonymous_verified',None)
    save(DEST/'PUBLICATION.json',proof);api=HfApi()
    if api.repo_info(DATASET,repo_type='dataset').sha!=head:raise ValueError('Main changed before proof')
    commit=api.create_commit(DATASET,repo_type='dataset',parent_commit=head,operations=[CommitOperationAdd('PUBLICATION.json',DEST/'PUBLICATION.json')],commit_message='Record verified values-filtered release and explicitly bounded audit scope')
    folder=STATE/'proof-readback';folder.mkdir(exist_ok=True);prior.prior.bounded.verify({'path':'PUBLICATION.json','bytes':(DEST/'PUBLICATION.json').stat().st_size,'sha256':sha(DEST/'PUBLICATION.json')},commit.oid,folder)
    final=prior.prior.parent.tree(commit.oid)
    if set(final)!=expected|{'PUBLICATION.json'} or any(final[p]!=tree[p] for p in tree):raise ValueError('Proof commit changed payload')
    proof.update(final_head=commit.oid,publication_receipt_anonymous_verified=True)
    save(STATE/'publication-final-commit.json',proof);save(prior.prior.parent.base.WORK/'reports/memory-harness-v2/CURRENT_PUBLICATION.json',proof)
    print(json.dumps({'final_head':commit.oid,'active':s['counts']['accepted'],'new_held':s['new_held_rows']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','stage','publish','verify']);phase=parser.parse_args().phase
    with (prior.prior.parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[phase]()
