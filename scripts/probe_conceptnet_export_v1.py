"""Preserve first diagnoses and measure exact portable lexical exports."""
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.relational_dataset import restore_relational_memory
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts import conceptnet_relation_adapter_v1 as adapter
from scripts.diagnose_conceptnet_relations_v1 import STATE as FIRST
from scripts.diagnose_wikidata_relations_v1 import independent_sql_target

STATE=adapter.INVENTORY.parent/'export-probe-v1'


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE/'summary.json').exists():raise ValueError('lexical export probe already frozen')
    audit=json.loads((FIRST/'semantic-audit-v1/summary.json').read_text())
    holds=json.loads((FIRST/'first-review-query-holds.json').read_text());held=set(holds['base_concepts'])
    pilots={r['record_id']:r for r in map(json.loads,(FIRST/'normalized/pilot-lexical.jsonl').open())}
    cases=[]
    for result in audit['results']:
        row=pilots[result['record_id']];request=json.loads((FIRST/'semantic-audit-v1'/result['id']/'request.json').read_text())
        payload=json.loads(request['messages'][1]['content'])
        if payload['messages']!=row['messages'] or payload['tools']!=row['tools']:raise ValueError('actual complete request differs')
        target=json.loads(row['messages'][-1]['content']);memory=restore_relational_memory(row)
        if canonical(independent_sql_target(row['source_refs']['query'],memory.facts))!=row['messages'][-1]['content']:
            raise ValueError('first exact target disagrees with independent SQL')
        if result['verdict']=='ACCEPT':continue
        trace=row['source_refs']['exact_computation_trace']
        if target['evidence_ids']!=sorted({eid for step in trace for eid in step['evidence_ids']}):raise ValueError('saved evidence union differs')
        cases.append({'record_id':row['record_id'],'first_verdict':result['verdict'],'first_issues':result['issues'],
            'exact_target':target,'query':row['source_refs']['query'],'actual_review_request_matches_complete_saved_messages':True,
            'target_valid_json':True,'target_evidence_matches_all_steps':True,'independent_SQL_equal':True,
            'still_held':row['source_task_key'] in held})
    if not all(c['still_held'] for c in cases) or len(cases)!=len(held):raise ValueError('first hold coverage differs')
    base.save(STATE/'first-review-independent-analysis.json',{'cases':cases,'actual_requests_compared':len(pilots),
        'first_audit_sha256':base.file_sha(FIRST/'semantic-audit-v1/summary.json'),
        'driver_sha256':base.file_sha(Path(__file__)),'model_requests':0,'answers_rewritten':0,'first_dispositions_unchanged':True,
        'interpretation':'Several critiques contradict valid JSON, present evidence IDs, or assert correctness while choosing REVISE. All first holds persist. The diagnosis is not a full-corpus error rate; finite-record exactness is separate from world truth.'})
    graph=adapter.SourceGraph();skills=SkillLibrary(base.REPO).render(adapter.SKILLS);base.init_tokenizer()
    anchors=list(graph.db.execute('SELECT base_uri,subject FROM anchors ORDER BY base_uri'))
    samples=random.Random(2026102708).sample(anchors,1024);configs={n:[] for n in ['records','tokenized','minicpm5_tokenized']}
    counts=Counter();raw_bytes=0;archive_path=STATE/'candidates.bca'
    if archive_path.exists():raise ValueError('partial probe archive exists; preserve')
    with archive.ArchiveWriter(archive_path) as writer:
        for anchor,subject in samples:
            try:
                if anchor in held:raise Rejected('first_nonaccept_base_concept')
                row=graph.make_row(anchor,subject,skills);memory=restore_relational_memory(row)
                if canonical(independent_sql_target(row['source_refs']['query'],memory.facts))!=row['messages'][-1]['content']:
                    raise ValueError('probe exact target disagrees')
                for call in row['source_refs']['actual_tool_calls']:
                    if base.sha(canonical(memory.lookup(**call['arguments'])))!=call['observation_sha256']:raise ValueError('restored observation differs')
                    counts['restored_observations']+=1
                record,pang,mini=base.compile_row(row)
            except Rejected as exc:counts['held:'+str(exc)]+=1;continue
            for name,value in [('records',record),('tokenized',pang),('minicpm5_tokenized',mini)]:configs[name].append(value)
            raw=(canonical(row)+'\n').encode();writer.append(raw);raw_bytes+=len(raw)
            counts['compiled']+=1;counts['pangolin_tokens']+=record['pangolin_tokens'];counts['minicpm_tokens']+=record['minicpm_tokens']
    files=[]
    for name,values in configs.items():
        table=pa.Table.from_pylist(values,schema=None if name=='records' else base.TOKEN_SCHEMA);dest=STATE/(name+'.parquet')
        if dest.exists():raise ValueError('partial probe array exists')
        pq.write_table(table,dest,compression='zstd',compression_level=6,row_group_size=128)
        if not pq.read_table(dest).equals(table):raise ValueError('complete probe arrays differ')
        files.append({'config':name,'path':str(dest),'bytes':dest.stat().st_size,'sha256':base.file_sha(dest)})
    restored=sum(len(raw) for _,_,raw in archive.iter_records(archive_path))
    if restored!=raw_bytes:raise ValueError('probe archive full byte coverage differs')
    measured=archive_path.stat().st_size+sum(f['bytes'] for f in files);mean=measured/counts['compiled']
    result={'driver_sha256':base.file_sha(Path(__file__)),'adapter_sha256':base.file_sha(Path(adapter.__file__)),
        'original_compiler_sha256':base.file_sha(Path(base.__file__)),'first_audit_sha256':base.file_sha(FIRST/'semantic-audit-v1/summary.json'),
        'first_review_analysis_sha256':base.file_sha(STATE/'first-review-independent-analysis.json'),
        'counts':dict(counts),'files':files,'archive_bytes':archive_path.stat().st_size,'archive_sha256':base.file_sha(archive_path),
        'raw_payload_bytes':raw_bytes,'measured_combined_bytes_per_row':mean,'projected_full_anchor_pool_bytes':round(mean*len(anchors)),
        'projection_is_not_a_storage_upper_bound':True,'all_three_configs_and_archive_exact_readback':True,
        'accepted_training_rows':0,'model_requests':0,'source_programs_executed':False,'first_holds_preserved':True}
    base.save(STATE/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
