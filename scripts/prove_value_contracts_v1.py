"""Check every routed finite-ID record and every civic article, no model calls."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from barbet_inference.value_alignment_gate import finite_identifier_contract, law_extract_contract
from scripts import publish_taiwan_civic_v1 as civic

STATE=civic.prior.parent.base.WORK/'reports/value-alignment-v1'


def check_file(job):
    pa.set_cpu_count(1)
    t=pq.read_table(Path(job['root'])/job['path'],columns=['record_id','query_hash','messages_json','tools_json','source_refs_json','memory_json','verification_tier'])
    t=t.filter(pc.is_in(t['record_id'],value_set=pa.array(job['ids'])))
    result=[]
    for row in t.to_pylist():
        try:
            proof=finite_identifier_contract(row);status='PROVEN'
        except Exception as exc:
            proof={'reason':type(exc).__name__+': '+str(exc)};status='UNPROVEN'
        result.append({'record_id':row['record_id'],'query_hash':row['query_hash'],'status':status,**proof})
    return result


def main():
    if (STATE/'contract-summary.json').exists():raise ValueError('First contract check is already frozen')
    candidates=pq.read_table(STATE/'context-scan-v2/candidates.parquet')
    # The family identifies a candidate schema, not approval: every row below
    # must pass the full visible-question/answer/tool/fact proof.
    selected=candidates.filter(pc.equal(candidates['source'],'Wikidata5M/original-inductive-train')).to_pylist()
    groups=defaultdict(list)
    for row in selected:groups[(row['root'],row['path'])].append(row['record_id'])
    jobs=[{'root':root,'path':path,'ids':ids} for (root,path),ids in sorted(groups.items())]
    policy={'driver_sha256':civic.sha(Path(__file__)),'gate_sha256':civic.sha(civic.prior.parent.base.REPO/'barbet_inference/value_alignment_gate.py'),'candidate_index_sha256':civic.sha(STATE/'context-scan-v2/candidates.parquet'),'routed_identifier_rows':len(selected),'no_name_only_approval':True}
    civic.save(STATE/'contract-policy.json',policy)
    results=[]
    with ProcessPoolExecutor(8) as pool:
        for n,part in enumerate(pool.map(check_file,jobs),1):
            results.extend(part)
            if n%40==0:print(json.dumps({'files':n,'total_files':len(jobs),'rows':len(results),'counts':dict(Counter(r['status'] for r in results))}),flush=True)
    root=ET.parse(civic.RAW/'laws.xml').getroot()
    originals={(law.findtext('法規名稱'),a.findtext('條號')):a.findtext('條文內容','').strip() for law in root for a in law.findall('.//條文')}
    for p in sorted((civic.DEST/'data/wave-0011/records').glob('*.parquet')):
        for row in pq.read_table(p).to_pylist():
            if row['verification_tier']!='exact_source_article_extract':continue
            try:proof=law_extract_contract(row,originals);status='PROVEN'
            except Exception as exc:proof={'reason':type(exc).__name__+': '+str(exc)};status='UNPROVEN'
            results.append({'record_id':row['record_id'],'query_hash':row['query_hash'],'status':status,**proof})
    pq.write_table(pa.Table.from_pylist(results),STATE/'contract-dispositions.parquet',compression='zstd')
    summary={'counts':dict(Counter(r['status'] for r in results)),'methods':dict(Counter(r.get('method','unproven') for r in results)),'failure_reasons':dict(Counter(r.get('reason') for r in results if r['status']!='PROVEN')),'rows':len(results),'dispositions_sha256':civic.sha(STATE/'contract-dispositions.parquet'),'scope':'Exact non-normative contract proof, not semantic certification of source claims; no training or source-program execution'}
    civic.save(STATE/'contract-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
