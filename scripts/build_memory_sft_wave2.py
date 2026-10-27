"""Second independent wave; the first wave and its holds are immutable."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import shutil
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import build_memory_sft_v2 as base
from scripts.memory_sft_wave2_adapters import openorca, toucan
from barbet_inference.sft import Rejected, normalize_messages, augment_system
from barbet_inference.memory import canonical

FIRST=base.STATE
STATE=base.WORK/'reports/memory-harness-v2-wave2'
OUT=base.WORK/'exports/memory-harness-v2-wave2'
RAW=base.RAW


def configure():
    base.STATE=STATE;base.OUT=OUT


def prepare():
    STATE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (STATE/'normalize-jobs.json').exists():raise ValueError('preparation already frozen')
    policy=base.REPO/'configs/memory_sft_v2_wave2.json'
    shutil.copyfile(policy,STATE/'policy-v1.json')
    lock=json.loads((RAW/'scale10m-sources/lock.json').read_text())
    sources=[s for s in lock if s['repo']=='Open-Orca/OpenOrca' or s['path'].split('/')[0] in {'Kimi-K2','OSS','Qwen3'}]
    sources.sort(key=lambda s:(0 if s['repo']=='Open-Orca/OpenOrca' else 1,s['path']))
    jobs=[];total=0
    for source in sources:
        path=Path(source['local'])
        if base.file_sha(path)!=source['sha256']:raise ValueError('source changed')
        file=pq.ParquetFile(path)
        if source['repo']=='Open-Orca/OpenOrca':
            # Upstream stores millions of rows in one row group. Repack once,
            # rather than rereading that whole group for every 2,000-row job.
            offset=0
            for batch in file.iter_batches(batch_size=2000):
                table=pa.Table.from_batches([batch]);stop=offset+len(table)
                part=RAW/'scale10m-derived-input'/path.stem/f'{offset:08d}.parquet'
                part.parent.mkdir(parents=True,exist_ok=True)
                if part.exists():
                    if not pq.read_table(part).equals(table):raise ValueError('repack conflict')
                else:pq.write_table(table,part,compression='zstd',compression_level=3)
                if not pq.read_table(part).equals(table):raise ValueError('repack readback')
                jobs.append({'source':source,'start':offset,'stop':stop,'read_local':str(part),
                    'read_sha256':base.file_sha(part),'format':'repacked_parquet','row_group':None})
                offset=stop
        else:
            offset=0
            for rg in range(file.num_row_groups):
                stop=offset+file.metadata.row_group(rg).num_rows
                jobs.append({'source':source,'start':offset,'stop':stop,'read_local':str(path),
                    'read_sha256':source['sha256'],'format':'parquet_row_group','row_group':rg})
                offset=stop
        total+=offset
    for n,job in enumerate(jobs):job['job_id']=f'{n:05d}'
    # Keep all attempted first-wave query identities, including first failures,
    # out of this expansion rather than laundering them through another source.
    ex=json.loads((FIRST/'exclusions.private.json').read_text())
    old=set(ex['old_query_hashes'])
    accepted=sqlite3.connect(f'file:{FIRST / "public-index.sqlite"}?mode=ro',uri=True)
    published={r[0] for r in accepted.execute('SELECT record_id FROM records')};accepted.close()
    aliases=sqlite3.connect(STATE/'frozen-aliases.sqlite')
    aliases.execute('CREATE TABLE aliases(node TEXT PRIMARY KEY,family_id TEXT,split TEXT)')
    aliases.execute('CREATE TABLE conflicts(node TEXT PRIMARY KEY)')
    for p in sorted((FIRST/'normalized').glob('[0-9]*.jsonl')):
        if p.name.endswith('.excluded.jsonl'):continue
        for line in p.open():old.add(json.loads(line)['query_hash'])
    for p in sorted((FIRST/'selected').glob('*.jsonl')):
        for line in p.open():
            r=json.loads(line)
            if r['record_id'] not in published:continue
            nodes=['family:'+r['source_family_id'],'query:'+r['query_hash']]
            for node in nodes:
                existing=aliases.execute('SELECT split FROM aliases WHERE node=?',(node,)).fetchone()
                if existing and existing[0]!=r['split']:aliases.execute('INSERT OR IGNORE INTO conflicts VALUES (?)',(node,))
                aliases.execute('INSERT OR IGNORE INTO aliases VALUES (?,?,?)',(node,r['family_id'],r['split']))
        aliases.commit()
    alias_count=aliases.execute('SELECT COUNT(*) FROM aliases').fetchone()[0];aliases.close()
    ex['old_query_hashes']=sorted(old)
    base.save(STATE/'exclusions.private.json',ex)
    base.save(STATE/'build-source-lock.json',{'files':sources,'policy_sha256':base.file_sha(policy)})
    base.save(STATE/'normalize-jobs.json',jobs)
    base.save(STATE/'preparation-summary.json',{'source_rows':total,'jobs':len(jobs),'sources':len(sources),
        'excluded_prior_query_hashes':len(old),'frozen_split_aliases':alias_count,
        'old_wave_unchanged':True,'source_programs_executed':False})
    print(canonical({'jobs':len(jobs),'source_rows':total,'prior_queries':len(old)}),flush=True)


def init():
    configure();base.init_normalizer()


def normalize_row(row,source,index):
    sid=source['repo'];v=openorca(row) if sid=='Open-Orca/OpenOrca' else toucan(row)
    query=v['query'];qh=base.sha(base.norm(query))
    if not query or len(query.strip())<12:raise Rejected('insufficient_task_identity')
    if qh in base.EX['old_query_hashes']:raise Rejected('prior_wave_or_held_query')
    if qh in base.EX['benchmark_query_hashes'] or any(base.sha(g) in base.EX['benchmark_13gram_hashes'] for g in base.ngrams(query)):
        raise Rejected('local_evaluation_overlap')
    tools,validators=base.tool_cache(canonical(v['tools']))
    messages=normalize_messages(v['messages'],tools,validators)
    targets=[canonical(m) for m in messages if m['role']=='assistant']
    if any(n>=4 for n in Counter(targets).values()):raise Rejected('repeated_assistant_loop')
    domain=v['domain'];messages=augment_system(messages,base.SKILLS[domain][0])
    rid=base.sha(canonical({'version':'memory-harness-sft-v2','messages':messages,'tools':tools}))
    family=base.sha(v['group'])
    provenance={'repo':sid,'revision':source['revision'],'file':source['path'],'file_sha256':source['sha256'],
        'row_index':index,'upstream_id':v['upstream_id'],'upstream_repo':sid,'license':v['license'],
        'attribution_url':'https://huggingface.co/datasets/'+sid,'raw_row_sha256':base.sha(canonical(row)),
        'new_semantic_generation':False,'selection_signal':v['selection_signal'],
        'selection_signal_is_heuristic':True,'source_task_sha256':base.sha(v['task_key']),
        'transformations':['private reasoning fields excluded','source-specific causal syntax adapter',
            'exact recognized tool boilerplate replaced by complete JSON interface','skill prefill','assistant-only loss']}
    return {'record_id':rid,'family_id':family,'source_family_id':family,
        'template_family_id':base.sha('query-family:'+base.norm(query,numbers=True)),
        'source_task_key':base.sha(v['task_key']),'query_hash':qh,'split':'unassigned','domain':domain,
        'source':sid,'upstream_source':sid,'license':v['license'],'messages':messages,'tools':tools,'memory':{},
        'source_refs':provenance,'skill_ids':base.ROUTES[domain],'skills':base.SKILLS[domain][1],
        'verification_tier':v['verification_tier'],'observation_origin':v['observation_origin'],
        'memory_placement':'chronological','actual_retrieval':False,
        'recorded_error_followup':any(re.search(r'(?i)traceback|\b(error|failed|failure|exception|timeout)\b',m['content']) for m in messages if m['role']=='tool'),
        'locally_executed_source_program':False,'assistant_turns':sum(m['role']=='assistant' and m['loss'] for m in messages),
        'tool_observations':sum(m['role']=='tool' for m in messages)}


def job(j):
    report=STATE/'normalized'/(j['job_id']+'.report.json')
    if report.exists():return json.loads(report.read_text())
    file=pq.ParquetFile(j['read_local'])
    if j['row_group'] is None:
        if base.file_sha(j['read_local'])!=j['read_sha256']:raise ValueError('repacked bytes changed')
        table=file.read()
    else:table=file.read_row_group(j['row_group'])
    table=table.slice(0,j['stop']-j['start'])
    counts=Counter();path=report.with_name(j['job_id']+'.jsonl');path.parent.mkdir(exist_ok=True)
    with path.open('w') as output,path.with_name(j['job_id']+'.excluded.jsonl').open('w') as held:
        for index,row in enumerate(table.to_pylist(),j['start']):
            counts['seen']+=1
            try:value=normalize_row(row,j['source'],index)
            except Rejected as exc:
                counts['excluded:'+str(exc)]+=1;held.write(canonical({'row_index':index,'reason':str(exc)})+'\n');continue
            output.write(canonical(value)+'\n');counts['accepted']+=1
    result={'job_id':j['job_id'],'counts':dict(counts),'path':str(path),'sha256':base.file_sha(path)}
    base.save(report,result);return result


def normalize(workers,pilot=False):
    jobs=json.loads((STATE/'normalize-jobs.json').read_text())
    if pilot:
        chosen={}
        for j in jobs:
            key=j['source']['path']
            if key not in chosen:chosen[key]={**j,'job_id':'pilot-'+j['job_id'],'stop':min(j['start']+20,j['stop'])}
        jobs=list(chosen.values())
    totals=Counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for n,result in enumerate(pool.map(job,jobs),1):
            totals.update(result['counts'])
            if n%50==0 or n==len(jobs):print(canonical({'jobs':n,'total':len(jobs),'counts':dict(totals)}),flush=True)
    base.save(STATE/('pilot-normalize-summary.json' if pilot else 'normalize-summary.json'),dict(totals))


def dedup():
    paths=[p for p in sorted((STATE/'normalized').glob('[0-9]*.jsonl')) if not p.name.endswith('.excluded.jsonl')]
    parent={}
    def find(x):
        parent.setdefault(x,x);root=x
        while parent[root]!=root:root=parent[root]
        while parent[x]!=x:old=parent[x];parent[x]=root;x=old
        return root
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for p in paths:
        for line in p.open():
            r=json.loads(line);node='family:'+r['family_id']
            union(node,'query:'+r['query_hash']);union(node,'family:'+r['template_family_id'])
    frozen=sqlite3.connect(f'file:{STATE / "frozen-aliases.sqlite"}?mode=ro',uri=True)
    ties={};conflicts=set()
    for node,family,split in frozen.execute('SELECT node,family_id,split FROM aliases'):
        if node not in parent:continue
        root=find(node)
        if root in ties and ties[root][1]!=split:conflicts.add(root)
        else:ties[root]=min(ties.get(root,(family,split)),(family,split))
    for (node,) in frozen.execute('SELECT node FROM conflicts'):
        if node in parent:conflicts.add(find(node))
    frozen.close();seen=set();tasks=set();counts=Counter();writer=None;shard=-1;size=0
    (STATE/'selected').mkdir(exist_ok=True)
    for p in paths:
        for line in p.open():
            r=json.loads(line);root=find('family:'+r['family_id'])
            if root in conflicts:counts['held_cross_wave_split_bridge']+=1;continue
            if r['source_task_key'] in tasks:counts['duplicate_source_task_or_user_sequence']+=1;continue
            identity=json.loads(canonical(r['messages']));ids={}
            for n,m in enumerate(identity):
                for k,call in enumerate(m.get('tool_calls',[])):
                    ids[call['id']]=f'call_{n}_{k}';call['id']=ids[call['id']]
                if 'tool_call_id' in m:m['tool_call_id']=ids[m['tool_call_id']]
            key=base.sha(canonical({'messages':identity,'tools':r['tools']}))
            if key in seen:counts['duplicate_episode']+=1;continue
            seen.add(key);tasks.add(r['source_task_key'])
            if root in ties:r['family_id'],r['split']=ties[root]
            else:
                r['family_id']=base.sha(root)
                r['split']='validation' if int(r['family_id'][:12],16)%100<2 else 'train'
            if writer is None or size>=2000:
                if writer:writer.close()
                shard+=1;size=0;writer=(STATE/'selected'/f'{shard:05d}.jsonl').open('w')
            writer.write(canonical(r)+'\n');size+=1;counts['selected']+=1
    if writer:writer.close()
    counts['shards']=shard+1;base.save(STATE/'dedup-summary.json',dict(counts));print(canonical(dict(counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','normalize','dedup','compile','audit']);p.add_argument('--workers',type=int,default=24);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    if not 1<=args.workers<=32:raise ValueError('CPU worker bound')
    if args.phase=='prepare':prepare()
    elif args.phase=='normalize':normalize(args.workers,args.pilot)
    elif args.phase=='dedup':dedup()
    elif args.phase=='compile':configure();base.compile_all(args.workers,args.pilot)
    else:
        from scripts import audit_memory_sft_v2 as audit
        audit.STATE=STATE;audit.main()
