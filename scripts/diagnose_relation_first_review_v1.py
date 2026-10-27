"""Compare first graph-review allegations to immutable complete inputs."""
import json
from pathlib import Path

from scripts import build_memory_sft_v2 as base
from scripts.diagnose_wikidata_relations_v1 import STATE


def main():
    rows={r['record_id']:r for r in (json.loads(line) for line in (STATE/'normalized/pilot-relations.jsonl').open())}
    audit=json.loads((STATE/'semantic-audit-v1/summary.json').read_text())
    proofs={r['record_id']:r for r in json.loads((STATE/'independent-sql-proofs.json').read_text())}
    cases=[];holds=[]
    for result in audit['results']:
        row=rows[result['record_id']];folder=STATE/'semantic-audit-v1'/result['id']
        request=json.loads((folder/'request.json').read_text())
        payload=json.loads(request['messages'][1]['content'])
        if payload['messages']!=row['messages'] or payload['tools']!=row['tools']:
            raise ValueError('actual first reviewer request differs from saved full sample')
        target=json.loads(row['messages'][-1]['content'])
        if not proofs[result['record_id']]['independent_sql_target_equal']:
            raise ValueError('pre-review exact SQL proof missing')
        if result['verdict']=='ACCEPT':continue
        trace=row['source_refs']['exact_computation_trace']
        traversed=sorted({eid for step in trace for eid in step['evidence_ids']})
        if target['evidence_ids']!=traversed:raise ValueError('saved evidence differs from complete derivation')
        item={'record_id':row['record_id'],'query_hash':row['query_hash'],'source_anchor_id':row['source_task_key'],
            'first_verdict_preserved':result['verdict'],'first_issues':result['issues'],
            'actual_review_request_contains_complete_original_messages':True,'target_is_valid_complete_json':True,
            'target_evidence_equals_all_saved_traversal_ids':True,'pre_review_independent_sql_target_equal':True,
            'exact_target':target,'query':row['source_refs']['query'],
            'status':'held; independent finite-record calculation agrees with target, but first nonaccept is not replayed or rewritten'}
        cases.append(item);holds.append({'query_hash':row['query_hash'],'anchor_id':row['source_task_key'],
            'record_id':row['record_id'],'reason':'first_relation_semantic_nonaccept'})
    base.save(STATE/'first-review-error-analysis.json',{'driver_sha256':base.file_sha(Path(__file__)),
        'first_audit_sha256':base.file_sha(STATE/'semantic-audit-v1/summary.json'),
        'pre_review_independent_sql_proofs_sha256':base.file_sha(STATE/'independent-sql-proofs.json'),
        'all_actual_request_samples_compared':len(rows),'cases':cases,
        'interpretation':'A non-accept count is not an error-rate estimate. Several critiques contradict the literal saved targets or assert correctness while choosing REVISE. Exact finite-record calculation is separately verified; no claim that source-world facts or general semantic quality were proved.',
        'model_requests':0,'first_dispositions_unchanged':True,'answers_rewritten':0})
    base.save(STATE/'first-review-query-holds.json',{'holds':holds,'query_hashes':sorted(r['query_hash'] for r in holds),
        'anchor_ids':sorted(r['anchor_id'] for r in holds),'no_second_choice_fallback':True,
        'first_audit_sha256':base.file_sha(STATE/'semantic-audit-v1/summary.json')})
    print(json.dumps({'actual_requests_equal':len(rows),'first_nonaccept_queries_held':len(holds),'model_requests':0}),flush=True)


if __name__=='__main__':main()
