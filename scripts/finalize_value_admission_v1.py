"""Freeze whole-query holds only after all predeclared first reviews finish."""
from collections import Counter
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from barbet_inference.value_admission import disposition
from scripts import audit_value_alignment_v1 as audit
from scripts import publish_taiwan_civic_v1 as civic

STATE=audit.STATE


def main():
    if (STATE/'admission.json').exists():raise ValueError('First admission decisions already frozen')
    plan=civic.read(STATE/'semantic-plan.json');summary=civic.read(STATE/'semantic-summary.json')
    if not civic.read(STATE/'controls.json')['passed']:raise ValueError('Controls failed')
    results={r['id']:r for r in summary['results']}
    if set(results)!={j['id'] for j in plan['jobs']}:raise ValueError('First review coverage is incomplete')
    positives=set();negatives={};negative_categories=Counter()
    for job in plan['jobs']:
        folder=STATE/'semantic-audit-v1'/job['id'];result=results[job['id']]
        if civic.read(folder/'disposition.json')!=result:raise ValueError('Disposition changed')
        if result['verdict']=='ACCEPT':
            request=civic.read(folder/'request.json');receipt=civic.read(folder/'receipt.json')
            payload=json.loads(request['messages'][1]['content'])
            if payload['messages']!=job['messages'] or payload['tools']!=job['tools'] or receipt['request_sha256']!=audit.audit.digest(request) or receipt['model']!=audit.audit.MODEL or receipt['finish_reason']!='stop':
                raise ValueError('Positive not bound to complete actual record')
            positives.add(job['record_id'])
        else:
            negatives[job['record_id']]=job['query_hash'];negative_categories[result['verdict']]+=1
    contracts=pq.read_table(STATE/'contract-dispositions.parquet').to_pylist()
    bindings={r['record_id']:r for r in pq.read_table(STATE/'system-bindings.parquet').to_pylist()}
    proofs={r['record_id'] for r in contracts if r['status']=='PROVEN' and bindings[r['record_id']]['status']=='BOUND'}
    for r in bindings.values():
        if r['status']!='BOUND':negatives[r['record_id']]=r['query_hash']
    candidates=pq.read_table(STATE/'tool-registry-scan-v1/candidates.parquet').to_pylist()
    decisions=[];holds=set(negatives.values())
    for r in candidates:
        rid=r['record_id']
        verdict='HOLD' if rid in negatives else ('ACCEPT' if rid in positives else None)
        status=disposition(query_held=False,sensitive=True,contextual_verdict=verdict,bound_contract=rid in proofs)
        status=status.replace('HOLD_','HELD_',1)
        if status.startswith('HELD'):holds.add(r['query_hash'])
        decisions.append({'record_id':rid,'query_hash':r['query_hash'],'source':r['source'],'status':status})
    # Non-hit sample failures and any failed mandatory civic case also hold the
    # whole query. A positive row never overrides another row's query hold.
    for r in decisions:r['query_retained']=r['query_hash'] not in holds
    old=set(civic.read(civic.prior.audit.STATE/'new-query-holds-v1.json')['query_hashes'])|set(civic.read(civic.STATE/'existing-topic-triage.json')['held_query_hashes'])
    if holds&old:raise ValueError('Previous holds appeared in active parent')
    pq.write_table(pa.Table.from_pylist(decisions),STATE/'candidate-dispositions.parquet',compression='zstd')
    negative_sample={r['record_id'] for r in civic.read(STATE/'context-scan-v2/negative-sample.json')}
    final_candidates={r['record_id'] for r in candidates}
    eligible_negative_sample=negative_sample-final_candidates
    sample_results=[r for r in results.values() if r['record_id'] in eligible_negative_sample]
    result={'query_hashes':sorted(holds),'new_held_queries':len(holds),'prior_held_queries':len(old),
        'candidate_rows':len(candidates),'candidate_dispositions':dict(Counter(r['status'] for r in decisions)),
        'candidate_queries_retained_rows':sum(r['query_retained'] for r in decisions),
        'semantic_first_reviews':len(results),'semantic_counts':summary['counts'],'semantic_negative_categories':dict(negative_categories),
        'screen_negative_sample':{'original_message_screen_sample':len(negative_sample),'remaining_screen_negative_after_registry_scan':len(eligible_negative_sample),'results':dict(Counter(r['verdict'] for r in sample_results))},
        'new_query_hold_is_not_a_claim_all_rows_are_wrong':True,'query_wide_no_runner_up_substitution':True,
        'no_exhaustive_semantic_alignment_guarantee':True,'driver_sha256':civic.sha(Path(__file__)),
        'decision_index_sha256':civic.sha(STATE/'candidate-dispositions.parquet'),'semantic_summary_sha256':civic.sha(STATE/'semantic-summary.json'),
        'contract_summary_sha256':civic.sha(STATE/'contract-summary.json'),'system_binding_summary_sha256':civic.sha(STATE/'system-binding-summary.json'),'policy_sha256':civic.sha(civic.prior.parent.base.REPO/'configs/value-alignment-policy-v1.json')}
    civic.save(STATE/'admission.json',result);print(json.dumps({k:v for k,v in result.items() if k!='query_hashes'}),flush=True)


if __name__=='__main__':main()
