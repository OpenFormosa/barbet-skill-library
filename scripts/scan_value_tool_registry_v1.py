"""Complete v2 message screening with every actual model-visible tool registry."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from barbet_inference.value_alignment import VECTOR_PATTERN,topic_reasons
from scripts import scan_value_alignment_v2 as previous

STATE=previous.STATE.parent/'tool-registry-scan-v1'
civic=previous.civic


def scan(spec):
    pa.set_cpu_count(1)
    t=pq.read_table(Path(spec['root'])/spec['path'],columns=['record_id','query_hash','family_id','split','domain','source','pangolin_tokens','tools_json'])
    t=t.filter(pc.invert(pc.is_in(t['query_hash'],value_set=pa.array(sorted(previous.HOLDS)))))
    rows=[]
    for row in t.filter(pc.match_substring_regex(t['tools_json'],VECTOR_PATTERN)).to_pylist():
        reasons=topic_reasons(row.pop('tools_json'))
        if reasons:rows.append({**row,**spec,'reasons_json':json.dumps(reasons)})
    return len(t),rows


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'summary.json').exists():raise ValueError('First registry inventory already frozen')
    source=civic.read(civic.prior.STATE/'plan.json')['base_manifest']
    jobs=[{'path':r['path'],'root':str(previous.ORIGINAL)} for r in source if '/records/' in r['path']]
    jobs += [{'path':r['path'],'root':str(civic.DEST)} for r in civic.read(civic.STATE/'new-cohort-summary.json')['files'] if '/records/' in r['path']]
    civic.save(STATE/'policy.json',{'driver_sha256':civic.sha(Path(__file__)),'previous_scan_sha256':civic.sha(previous.STATE/'scan-summary.json'),'scope':'All active tool registries; union with complete message/scenario-suffix candidate inventory; no verdict from a match'})
    union={r['record_id']:r for r in pq.read_table(previous.STATE/'candidates.parquet').to_pylist()};initial=len(union);total=0;registry_count=0
    with ProcessPoolExecutor(8,initializer=previous.init) as pool:
        for n,(count,rows) in enumerate(pool.map(scan,jobs),1):
            total+=count;registry_count+=len(rows)
            for row in rows:
                rid=row['record_id']
                if rid in union:
                    reasons=sorted(set(json.loads(union[rid]['reasons_json']))|set(json.loads(row['reasons_json'])))
                    union[rid]['reasons_json']=json.dumps(reasons);union[rid]['registry_topic_match']=True
                else:union[rid]={**row,'registry_topic_match':True,'target_chars':None}
            if n%200==0:print(json.dumps({'shards':n,'rows':total,'registry_hits':registry_count,'union_rows':len(union)}),flush=True)
    if total!=civic.read(previous.STATE/'scan-summary.json')['active_rows_scanned']:raise ValueError('Registry coverage differs')
    table=pa.Table.from_pylist([{**r,'registry_topic_match':r.get('registry_topic_match',False)} for r in union.values()])
    pq.write_table(table,STATE/'candidates.parquet',compression='zstd')
    summary={'active_rows_scanned':total,'previous_message_candidates':initial,'registry_topic_matches':registry_count,'additional_registry_only_rows':len(union)-initial,'candidate_rows':len(union),'sources':dict(Counter(r['source'] for r in union.values())),'candidate_sha256':civic.sha(STATE/'candidates.parquet')}
    civic.save(STATE/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
