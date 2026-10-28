"""Scan every currently active row; freeze topic candidates and a non-hit sample."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from barbet_inference.value_alignment import VECTOR_PATTERN, topic_reasons
from scripts import publish_taiwan_civic_v1 as civic

STATE=civic.prior.parent.base.WORK/'reports/value-alignment-v1/context-scan-v2'
ORIGINAL=civic.prior.parent.parent.DEST


def init():
    global HOLDS
    pa.set_cpu_count(1)
    HOLDS=set(civic.read(civic.prior.audit.STATE/'new-query-holds-v1.json')['query_hashes'])
    HOLDS.update(civic.read(civic.STATE/'existing-topic-triage.json')['held_query_hashes'])


def scan(spec):
    key=spec['path'].replace('/','__').replace('.parquet','')
    path=Path(spec['root'])/spec['path']
    columns=['record_id','query_hash','family_id','split','domain','source','messages_json','pangolin_tokens']
    table=pq.read_table(path,columns=columns)
    table=table.filter(pc.invert(pc.is_in(table['query_hash'],value_set=pa.array(sorted(HOLDS)))))
    hit=pc.match_substring_regex(table['messages_json'],VECTOR_PATTERN)
    candidates=[];samples=[]
    for row in table.filter(hit).to_pylist():
        messages=json.loads(row.pop('messages_json'))
        # Authored skill boilerplate is shared protocol, not source evidence.
        # Inspect all actual task/source/assistant/tool content; source-provided
        # scenario instructions must be checked from the system scenario suffix too.
        scenario_marker='SCENARIO RULES (native tool formatting takes precedence):'
        content=[]
        for m in messages:
            if m['role']!='system': content.append(m.get('content',''))
            elif scenario_marker in m.get('content',''): content.append(m['content'].split(scenario_marker,1)[1])
        body='\n'.join(content)
        calls=json.dumps([m.get('tool_calls',[]) for m in messages],ensure_ascii=False)
        reasons=topic_reasons(body+'\n'+calls)
        if reasons:
            row.update(path=spec['path'],root=spec['root'],reasons_json=json.dumps(reasons),
                       target_chars=sum(len(m.get('content','')) for m in messages if m['role']=='assistant' and m.get('loss')))
            candidates.append(row)
    # Screen-negative deterministic hash sample; select the globally smallest
    # IDs after scanning. Topic-sensitive rows are never allowed via this sample.
    candidate_ids={r['record_id'] for r in candidates}
    for row in table.filter(pc.starts_with(table['record_id'],'000')).drop(['messages_json']).to_pylist():
        if row['record_id'] not in candidate_ids:
            row.update(path=spec['path'],root=spec['root']);samples.append(row)
    folder=STATE/'scan-parts';folder.mkdir(exist_ok=True)
    result={'path':spec['path'],'rows':len(table),'candidate_count':len(candidates),'candidates':candidates,'negative_sample':samples,
        'topics':dict(Counter(t for r in candidates for t in json.loads(r['reasons_json'])))}
    civic.save(folder/(key+'.json'),result)
    return {k:v for k,v in result.items() if k not in ['candidates','negative_sample']}


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    if (STATE/'scan-summary.json').exists():raise ValueError('First full scan already frozen')
    source=civic.read(civic.prior.STATE/'plan.json')['base_manifest']
    jobs=[{'path':r['path'],'root':str(ORIGINAL)} for r in source if '/records/' in r['path']]
    new=civic.read(civic.STATE/'new-cohort-summary.json')
    jobs += [{'path':r['path'],'root':str(civic.DEST)} for r in new['files'] if '/records/' in r['path']]
    civic.save(STATE/'scan-policy.json',{'driver_sha256':civic.sha(Path(__file__)),
        'router_sha256':civic.sha(civic.prior.parent.base.REPO/'barbet_inference/value_alignment.py'),
        'parent':civic.read(civic.STATE/'publication-final-commit.json')['final_head'],
        'patterns':VECTOR_PATTERN,'source_scenario_system_suffix_included':True,'source_or_structured_type_exemptions':False,
        'matching_is_not_a_negative_verdict':True,'negative_sampling':'IDs starting 000; global smallest 256, fixed before review',
        'no_source_program_execution':True,'no_model_training':True})
    total=0;count=0;topics=Counter()
    with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        for n,result in enumerate(pool.map(scan,jobs),1):
            total+=result['rows'];count+=result['candidate_count'];topics.update(result['topics'])
            if n%100==0 or n==len(jobs):print(json.dumps({'shards':n,'total_shards':len(jobs),'active_rows':total,'candidates':count}),flush=True)
    expected=civic.read(civic.STATE/'publication-final-commit.json')['active_new_episodes']
    if total!=expected:raise ValueError('Active release coverage mismatch')
    candidates=[];negative=[]
    for path in sorted((STATE/'scan-parts').glob('*.json')):
        part=civic.read(path);candidates.extend(part['candidates']);negative.extend(part['negative_sample'])
    negative=sorted(negative,key=lambda r:r['record_id'])[:256]
    pq.write_table(pa.Table.from_pylist(candidates),STATE/'candidates.parquet',compression='zstd')
    civic.save(STATE/'negative-sample.json',negative)
    civic.save(STATE/'scan-summary.json',{'active_rows_scanned':total,'candidate_rows':count,'topics':dict(topics),
        'negative_sample_size':len(negative),'candidate_index_sha256':civic.sha(STATE/'candidates.parquet'),
        'no_semantic_alignment_claim_from_topic_scan':True,'all_sources_and_structured_queries_included':True})
    print(json.dumps({'active_rows_scanned':total,'candidate_rows':count,'topics':dict(topics)}),flush=True)


if __name__=='__main__':main()
