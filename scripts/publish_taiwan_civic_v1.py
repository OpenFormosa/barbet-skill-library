"""Versioned political-topic quarantine and source-attributed civic supplement."""
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
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, DatasetCard

from scripts import publish_barbet_readiness_v1 as prior
from scripts.build_taiwan_civic_v1 import STATE, DEST, RAW

CACHE = STATE / 'verified-staging-cache'
BRANCH = 'taiwan-civic-20261028-v1'
DATASET = prior.parent.DATASET
FOLDER = 'reports/taiwan-civic-v1'
read, save, sha = prior.read, prior.save, prior.sha


def prepare():
    if (STATE / 'publication-plan.json').exists(): raise ValueError('Civic plan already frozen')
    if subprocess.check_output(['git','status','--porcelain'], cwd=prior.parent.base.REPO, text=True).strip():
        raise ValueError('Commit implementation before freezing')
    parent = read(prior.STATE / 'publication-final-commit.json')['final_head']
    if HfApi(token=False).repo_info(DATASET, repo_type='dataset').sha != parent: raise ValueError('Parent head changed')
    triage = read(STATE / 'existing-topic-triage.json'); holds = triage['held_query_hashes']
    old = read(prior.STATE / 'plan.json')
    previous = {j['path']: j for j in old['jobs']}
    manifest = read(prior.STATE / 'publication-plan.json')['manifest']
    raw_manifest = {x['path']: x for x in old['base_manifest']}
    jobs = []; removed = Counter()
    sources = [x for x in old['base_manifest'] if '/records/' in x['path']]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, part in enumerate(pool.map(lambda item: prior.scan(item, holds), sources), 1):
            if part['excluded']:
                removed.update(part['counts'])
                part['new_exclude_indices'] = part['exclude_indices']
                part['exclude_indices'] = sorted(set(part['exclude_indices']) | set(previous.get(part['path'], {}).get('exclude_indices', [])))
                part['inputs'] = [raw_manifest[p] for p in [part['path'], part['path'].replace('/records/','/tokenized/')]]
                part['input_bytes'] = sum(x['bytes'] for x in part['inputs']); jobs.append(part)
            if n % 200 == 0: print(json.dumps({'prepared_shards':n,'held_rows':removed['accepted']}),flush=True)
    new = read(STATE / 'new-cohort-summary.json'); added = Counter()
    for item in new['files']:
        if '/records/' in item['path']:
            for row in pq.read_table(DEST / item['path']).to_pylist(): added.update(prior.removed_stats(row))
    counts = Counter(read(prior.STATE / 'summary.json')['counts']); counts.subtract(removed); counts.update(added)
    if any(v < 0 for v in counts.values()): raise ValueError('Negative metric')
    db = sqlite3.connect(f'file:{prior.parent.parent.STATE / "public-index.sqlite"}?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=MEMORY'); db.execute('CREATE TEMP TABLE held(query_hash TEXT PRIMARY KEY)')
    old_holds = read(prior.audit.STATE / 'new-query-holds-v1.json')['query_hashes']
    if set(old_holds) & set(holds): raise ValueError('New holds overlap previous')
    db.executemany('INSERT INTO held VALUES(?)',[(h,) for h in old_holds + holds])
    family_count = db.execute('SELECT COUNT(DISTINCT family_id) FROM records WHERE query_hash NOT IN (SELECT query_hash FROM held)').fetchone()[0]
    total_held = db.execute('SELECT COUNT(*) FROM records WHERE query_hash IN (SELECT query_hash FROM held)').fetchone()[0]
    if total_held != removed['accepted'] + read(prior.STATE / 'summary.json')['held_rows']: raise ValueError('Query-wide hold mismatch')
    old_unique = read(prior.STATE / 'summary.json')['unique_counts']
    unique = {'record_id': counts['accepted'], 'query_hash': old_unique['query_hash'] - len(holds) + new['unique_queries'],
              'family_id': family_count + new['unique_families']}
    db.close()
    for prefix in ['source:', 'domain:', 'length_bucket:', 'selection:']:
        if sum(v for k,v in counts.items() if k.startswith(prefix)) != counts['accepted']: raise ValueError('Cumulative count mismatch')
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=prior.parent.base.REPO,text=True).strip()
    save(STATE / 'publication-plan.json', {'parent':parent,'manifest':manifest,'jobs':jobs,'library_commit':code,
        'driver_sha256':sha(Path(__file__)), 'new_cohort_sha256':sha(STATE/'new-cohort-summary.json'),
        'triage_sha256':sha(STATE/'existing-topic-triage.json'), 'policy_sha256':sha(prior.parent.base.REPO/'configs/taiwan-civic-policy-v1.json')})
    save(STATE / 'summary.json', {'counts':dict(counts),'unique_counts':unique,'removed_counts':dict(removed),
        'added_counts':dict(added),'new_political_topic_hold_queries':len(holds),'new_political_topic_held_rows':removed['accepted'],
        'new_civic_episodes':new['episodes'],'previous_quality_held_rows':read(prior.STATE/'summary.json')['held_rows'],
        'civic_scope':new, 'all_topic_matches_not_claimed_wrong':True,'no_runner_up_substitution':True})
    save(STATE / 'parent-tree.json',prior.parent.tree(parent))
    print(json.dumps({'prepared':True,'changed_record_shards':len(jobs),'removed':removed['accepted'],'added':new['episodes'],'active':counts['accepted']}),flush=True)


def rewrite(job):
    output = []
    for item in job['inputs']:
        src = prior.parent.parent.DEST / item['path']
        if sha(src) != item['sha256']: raise ValueError('Original frozen input changed')
        table = pq.read_table(src); keep = np.ones(len(table),dtype=bool); keep[job['exclude_indices']] = False
        expected = table.filter(pa.array(keep)); target = CACHE / item['path']; target.parent.mkdir(parents=True,exist_ok=True)
        if not len(expected): output.append({'path':item['path'],'delete':True}); continue
        pq.write_table(expected,target,compression='zstd',compression_level=3)
        if not pq.read_table(target).equals(expected): raise ValueError('Retained values/arrays changed')
        output.append({'path':item['path'],'bytes':target.stat().st_size,'sha256':sha(target),'rows':len(expected)})
    return output


def stage():
    plan = read(STATE/'publication-plan.json')
    if plan['driver_sha256'] != sha(Path(__file__)): raise ValueError('Frozen publisher changed')
    api = HfApi(); receipt_dir = STATE/'staging-batches'; receipt_dir.mkdir(exist_ok=True)
    if not (STATE/'staging-created.json').exists():
        if api.repo_info(DATASET,repo_type='dataset').sha != plan['parent']: raise ValueError('Main changed')
        api.create_branch(DATASET,repo_type='dataset',branch=BRANCH,revision=plan['parent'],exist_ok=False)
        result=api.create_commit(DATASET,repo_type='dataset',revision=BRANCH,parent_commit=plan['parent'],
            operations=[CommitOperationDelete('PUBLICATION.json'),CommitOperationAdd('STAGING_INCOMPLETE.txt',b'INCOMPLETE CIVIC REVISION. Use main until final verification.\n')],
            commit_message='Mark incomplete civic staging branch')
        save(STATE/'staging-created.json',{'head':result.oid})
    batches=[]; current=[]; size=0
    for job in plan['jobs']:
        if current and size+job['input_bytes'] > 512*1024**2: batches.append(current); current=[];size=0
        if job['input_bytes'] > 2*1024**3: raise ValueError('Single job exceeds cache bound')
        current.append(job);size+=job['input_bytes']
    if current:batches.append(current)
    head=read(STATE/'staging-created.json')['head']; all_files=[]
    for i,batch in enumerate(batches):
        receipt=receipt_dir/f'{i:04d}.json'
        if receipt.exists():
            saved=read(receipt)
            if saved['parent']!=head or saved['paths']!=[j['path'] for j in batch]: raise ValueError('Resume receipt differs')
            head=saved['head'];all_files.extend(saved['files']);continue
        if api.repo_info(DATASET,repo_type='dataset',revision=BRANCH).sha != head: raise ValueError('Staging changed')
        if shutil.disk_usage(DEST).free < 100*1024**3 + 512*1024**2: raise ValueError('Disk floor')
        with ThreadPoolExecutor(max_workers=4) as pool: files=[f for fs in pool.map(rewrite,batch) for f in fs]
        if sum(f.get('bytes',0) for f in files)>2*1024**3: raise ValueError('Cache bound')
        ops=[CommitOperationDelete(f['path']) if f.get('delete') else CommitOperationAdd(f['path'],CACHE/f['path']) for f in files]
        result=api.create_commit(DATASET,repo_type='dataset',revision=BRANCH,parent_commit=head,operations=ops,num_threads=8,
            commit_message=f'Hold unreviewed political topic queries: verified subset batch {i+1}/{len(batches)}')
        folder=STATE/'staging-readback'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True)
        prior.bounded.STATE=STATE
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda f: prior.bounded.verify({k:f[k] for k in ['path','bytes','sha256']},result.oid,folder),[f for f in files if not f.get('delete')]))
        save(receipt,{'parent':head,'head':result.oid,'paths':[j['path'] for j in batch],'files':files,'all_changed_bytes_anonymously_verified':True})
        head=result.oid;all_files.extend(files)
        for f in files:
            if not f.get('delete'):(CACHE/f['path']).unlink()
        print(json.dumps({'batch':i+1,'total':len(batches),'verified':True}),flush=True)
    save(STATE/'staging-complete.json',{'head':head,'files':all_files,'all_retained_values_and_arrays_equal':True})


def metadata(plan):
    summary=read(STATE/'summary.json'); old=read(prior.STATE/'summary.json')
    text=(prior.DEST/'README.md').read_text()
    # Update every displayed cumulative metric; historical audit paragraph counts
    # have different values and remain explicitly historical below.
    replacement={}
    for key,value in old['counts'].items():
        if key in summary['counts'] and value!=summary['counts'][key] and value>1000: replacement[f'{value:,}']=f'{summary["counts"][key]:,}'
    for key,value in old['unique_counts'].items(): replacement[f'{value:,}']=f'{summary["unique_counts"][key]:,}'
    text=__import__('re').sub('|'.join(__import__('re').escape(k) for k in sorted(replacement,key=len,reverse=True)),lambda m:replacement[m[0]],text)
    source_header = '| Source | Episodes |'
    if source_header in text:
        start = text.index(source_header); end = text.index('\n\n', start)
        text = text[:end] + '\n| OpenFormosa/taiwan-civic-v1 | 933 |' + text[end:]
    text += f'''\n## Taiwan civic policy and source-grounded supplement\n\nThe dataset owner explicitly adopts **Taiwan sovereignty and the free democratic\nself-determination of Taiwan's people**, together with democracy, liberty, human\nrights, rule of law, equality and pluralism. This is a disclosed normative policy,\nnot a claim that every government shares the same diplomatic or legal view.\nNo political party is the default. Accurate history, translations, quotations,\npeaceful dissent and criticism of Taiwan's government remain protected. Chinese\npeople and all other groups retain the same human dignity and rights.\n\nThis successor revision conservatively quarantines {summary['new_political_topic_held_rows']:,}\nadditional episodes across {summary['new_political_topic_hold_queries']:,} whole queries pending\ncontextual review. Lexical triage scanned all 10,328,696 original rows and found\n146 natural-language candidate targets. Some include actual factual errors; others\nare correct quotations or false positives. This is not a claim that every held\nanswer is false, or that all remaining political content has been semantically\ncertified. Finite source-record relation queries answer the specified record\ncontract and are not automatically endorsements of geopolitical world truth.\n\n**933 new civic episodes** comprise 920 exact article-extraction tasks from\n22 Taiwan laws, 12 explicitly authored value cases and one date-bounded, attributed\nMOFA statement. There is one target per complete input: these are unscored\nsingletons, not new v26 PPL winners. The original Constitution is limited to\nfoundational/rights clauses; all extracts are requests about the supplied version,\nnot claims of current individual legal applicability. Amendment/effective-date\nmetadata is retained. No generated answer variants, MiniCPM tokenizer, new model\nrequests, tool executions or weight updates are involved.\n\nThe skill and policy are available in [CIVIC_POLICY.json](CIVIC_POLICY.json),\n[the source and quality report]({FOLDER}/summary.json), and\n[the civic skill](skills/taiwan-civic-values/SKILL.md). Law families stay in one\nsplit, with distractors drawn only from that split. Model-visible evidence includes\nsource documents; audit reports, gold targets and construction metadata must not\nbe indexed as RAG evidence. New historical mini_* fields are null. The prior\nv26-readiness reports describe the preceding correction; this section and the\nlatest PUBLICATION.json describe the current cumulative release.\n\nThese civic episodes are a small subset of the full corpus. Their presence alone\ndoes not prove model alignment or capability gains. Measure answer accuracy,\nsource attribution, viewpoint/fact separation and retention on held-out tasks\nbefore claiming benefits; never treat an oversight keyword scan as that evaluation.\n'''
    DatasetCard(text).validate();(DEST/'README.md').write_text(text)
    notice=(prior.parent.DEST/'NOTICE.md').read_text()
    notice+='''\n\n## Taiwan civic v1\n\n920 article extracts from 22 laws: Taiwan Ministry of Justice, government open\ndata registry https://data.gov.tw/dataset/18289, XML snapshot update 2026/10/16,\nretrieved 2026-10-28. Government Data Open License Taiwan 1.0:\nhttps://data.gov.tw/license. Maintain attribution and inspect effective-date and\namendment relationships; extracts are not a warranty of current applicability.\nOne attributed MOFA 2022-04-26 statement excerpt: https://www.mofa.gov.tw/News_Content.aspx?n=95&s=97745;\nMOFA reuse declaration https://www.mofa.gov.tw/cp.aspx?n=292, OGDL-Taiwan-1.0.\nNo third-party photographs, logos or attachments are used.\nTwelve original normative cases and the authored policy/skill: Apache-2.0.\nThe political preference is owner-authored; official source attribution is not\nan assertion of source endorsement of this dataset or model.\n'''
    (DEST/'NOTICE.md').write_text(notice)
    (DEST/'TRAINING.md').write_text((prior.DEST/'TRAINING.md').read_text()+'''\n\nCivic subset: select source == "OpenFormosa/taiwan-civic-v1" from records.\nThe declared owner policy is in CIVIC_POLICY.json. Corpus row counts do not\nensure a meaningful civic token share; evaluate and choose a documented sampling\nmixture in a separately authorized training pilot. Do not duplicate episodes to\nclaim more independent data. Policy cases are authored norms; factual assertions\nstill require sources. No capability or alignment improvement has been measured.\n''')
    for source,relative in [(prior.parent.base.REPO/'configs/taiwan-civic-policy-v1.json','CIVIC_POLICY.json'),
        (prior.parent.base.REPO/'skills/taiwan-civic-values/SKILL.md','skills/taiwan-civic-values/SKILL.md'),
        (prior.parent.base.REPO/'skills/taiwan-civic-values/skill.json','skills/taiwan-civic-values/skill.json'),
        (prior.DEST/'recipe/barbet-training-v2.json','recipe/barbet-training-v2.json')]:
        target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    recipe=read(DEST/'recipe/barbet-training-v2.json');recipe.update(civic_policy='CIVIC_POLICY.json',civic_source_filter='OpenFormosa/taiwan-civic-v1',civic_efficacy_measured=False)
    save(DEST/'recipe/barbet-training-v2.json',recipe)
    for name in ['summary.json','new-cohort-summary.json','source-lock.json','existing-topic-triage.json']:
        save(DEST/FOLDER/name,read(STATE/name))


def publish():
    if (STATE/'publication-data-commit.json').exists():raise ValueError('Already published')
    plan=read(STATE/'publication-plan.json');stage=read(STATE/'staging-complete.json');api=HfApi()
    if sha(Path(__file__))!=plan['driver_sha256']:raise ValueError('Frozen publisher changed')
    if api.repo_info(DATASET,repo_type='dataset').sha!=plan['parent']:raise ValueError('Main changed')
    if api.repo_info(DATASET,repo_type='dataset',revision=BRANCH).sha!=stage['head']:raise ValueError('Staging changed')
    metadata(plan);manifest={x['path']:x for x in plan['manifest']};operations=[]
    for f in stage['files']:
        if f.get('delete'):manifest.pop(f['path']);operations.append(CommitOperationDelete(f['path']))
        else:
            manifest[f['path']]={k:f[k] for k in ['path','bytes','sha256']}
            operations.append(CommitOperationCopy(f['path'],f['path'],src_revision=stage['head']))
    uploaded=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        item={'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':sha(p)}
        manifest[item['path']]=item;uploaded.append(item);operations.append(CommitOperationAdd(item['path'],p))
    operations.append(CommitOperationDelete('PUBLICATION.json'))
    save(STATE/'publication-manifest.json',{'files':sorted(manifest.values(),key=lambda x:x['path']),'uploaded':uploaded,'changed':stage['files'],'staging_head':stage['head']})
    result=api.create_commit(DATASET,repo_type='dataset',parent_commit=plan['parent'],operations=operations,num_threads=8,
        commit_message='Add source-grounded Taiwan civic curriculum and disclosed sovereignty/self-determination policy; hold unreviewed topic queries')
    save(STATE/'publication-data-commit.json',{'commit':result.oid,'manifest_sha256':sha(STATE/'publication-manifest.json')})
    print(json.dumps({'data_commit':result.oid}),flush=True)


def verify():
    plan=read(STATE/'publication-plan.json');manifest=read(STATE/'publication-manifest.json');data=read(STATE/'publication-data-commit.json')
    if sha(STATE/'publication-manifest.json')!=data['manifest_sha256']:raise ValueError('Manifest changed')
    head=data['commit'];tree=prior.parent.tree(head);old=read(STATE/'parent-tree.json');stage=prior.parent.tree(manifest['staging_head'])
    uploaded={f['path'] for f in manifest['uploaded']};changed={f['path'] for f in manifest['changed']}
    expected={f['path'] for f in manifest['files']}|{'.gitattributes'}
    if set(tree)!=expected or any('minicpm5_tokenized' in p for p in tree):raise ValueError('Unexpected main tree')
    for p in expected-uploaded:
        if tree[p]!=(stage if p in changed else old)[p]:raise ValueError('Object identity mismatch: '+p)
    folder=STATE/'main-new-file-readback';folder.mkdir(exist_ok=True);prior.bounded.STATE=STATE
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda f:prior.bounded.verify(f,head,folder),manifest['uploaded']))
    s=read(STATE/'summary.json');previous=read(prior.STATE/'publication-final-commit.json')
    proof={**previous,'parent':plan['parent'],'data_commit':head,'active_new_episodes':s['counts']['accepted'],
        'unique_normalized_first_queries':s['unique_counts']['query_hash'],'unique_task_families':s['unique_counts']['family_id'],
        'train_episodes':s['counts']['split:train'],'validation_episodes':s['counts']['split:validation'],
        'held_queries':previous['held_queries']+s['new_political_topic_hold_queries'],
        'removed_episodes':previous['removed_episodes']+s['new_political_topic_held_rows'],
        'new_civic_episodes':s['new_civic_episodes'],'political_policy':'Taiwan sovereignty and peoples free democratic self-determination; universal human rights; no default party',
        'political_audit_exhaustive_semantic_certification':False,'staging_head':manifest['staging_head'],
        'changed_payload_files_anonymously_verified':len(manifest['changed']),
        'retained_unchanged_objects_verified':len(expected-uploaded-changed),
        'verified_files':len(manifest['files']),'verified_bytes':sum(f['bytes'] for f in manifest['files']),
        'library_commit':plan['library_commit'],'verification_method':'prior verified objects plus exact retained-row subset staging byte readback and copy identities plus all new files anonymous SHA256',
        'publication_receipt_anonymous_verified':False,'anonymous_full_file_sha256_readback':False}
    proof.pop('final_head',None)
    save(DEST/'PUBLICATION.json',proof);api=HfApi()
    if api.repo_info(DATASET,repo_type='dataset').sha!=head:raise ValueError('Head changed before proof')
    result=api.create_commit(DATASET,repo_type='dataset',parent_commit=head,
        operations=[CommitOperationAdd('PUBLICATION.json',DEST/'PUBLICATION.json')],commit_message='Record civic source/quality revision and verified cumulative delivery')
    item={'path':'PUBLICATION.json','bytes':(DEST/'PUBLICATION.json').stat().st_size,'sha256':sha(DEST/'PUBLICATION.json')}
    receipt=STATE/'proof-readback';receipt.mkdir(exist_ok=True);prior.bounded.verify(item,result.oid,receipt)
    final=prior.parent.tree(result.oid)
    if set(final)!=expected|{'PUBLICATION.json'} or any(final[p]!=tree[p] for p in tree):raise ValueError('Proof changed payloads')
    final_proof={**proof,'final_head':result.oid,'publication_receipt_anonymous_verified':True}
    save(STATE/'publication-final-commit.json',final_proof)
    save(prior.parent.base.WORK/'reports/memory-harness-v2/CURRENT_PUBLICATION.json',final_proof)
    print(json.dumps({'final_head':result.oid,'episodes':s['counts']['accepted'],'new_civic_episodes':s['new_civic_episodes']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','stage','publish','verify']);args=parser.parse_args()
    with (prior.parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[args.phase]()
