"""Supplement existing historical heuristic flags with source-recorded evidence."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import audit_barbet_continuation_v1 as audit
from scripts import publish_barbet_only_v1 as parent
from scripts import publish_barbet_readiness_v1 as correction
from barbet_inference.refinement_evidence import recorded_refinement_evidence

REMOVED = None


def init():
    global REMOVED
    REMOVED = {rid for job in parent.read(correction.STATE/'plan.json')['jobs'] for rid in job['removed_ids']}
    pa.set_cpu_count(1)


def job(path):
    part=parent.read(path);c=part['counts']
    if not any(c.get('source:'+name) for name in ['voidful/agent-sft','nvidia/Nemotron-SFT-Agentic-v2','Agent-Ark/Toucan-1.5M']):
        return {'rows':[],'counts':{}}
    rows=[];counts=Counter()
    for batch in pq.ParquetFile(parent.parent.DEST/part['path']).iter_batches(batch_size=128,
        columns=['record_id','source','messages_json','tool_observations','recorded_error_followup']):
        for row in batch.to_pylist():
            if row['record_id'] in REMOVED or not row['tool_observations']:continue
            counts['eligible_tool_episodes_scanned']+=1
            counts['historical_error_word_flags']+=int(row['recorded_error_followup'])
            evidence=recorded_refinement_evidence(json.loads(row['messages_json']))
            if evidence['explicit_error_observations']:counts['episodes_with_explicit_recorded_error']+=1
            if evidence['same_tool_retries']:
                counts['episodes_with_recorded_same_tool_retry']+=1
                if evidence['retry_observations_without_explicit_error']:counts['episodes_with_retry_nonerror_observation']+=1
                if evidence['retry_observations_with_explicit_success']:counts['episodes_with_retry_explicit_success']+=1
                rows.append({'record_id':row['record_id'],'source':row['source'],**evidence})
    return {'rows':rows,'counts':dict(counts)}


if __name__=='__main__':
    if (audit.STATE/'refinement-evidence-summary.json').exists():raise ValueError('First evidence scan already complete')
    parent.base.save(audit.STATE/'refinement-evidence-policy.json',{'driver_sha256':parent.base.file_sha(Path(__file__)),
        'evidence_extractor_sha256':parent.base.file_sha(parent.base.REPO/'barbet_inference/refinement_evidence.py'),
        'original_rows_and_heuristic_flags_not_modified':True,'requires_explicit_error_then_later_same_tool_action':True,
        'upstream_recording_not_local_execution_or_end_to_end_correctness':True,'workers':8})
    rows=[];counts=Counter();parts=sorted((audit.STATE/'files').glob('*.json'))
    with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        for n,part in enumerate(pool.map(job,parts),1):
            rows.extend(part['rows']);counts.update(part['counts'])
            if n%150==0 or n==len(parts):print(json.dumps({'files':n,'total':len(parts),'retry_episodes':len(rows)}),flush=True)
    if len({r['record_id'] for r in rows})!=len(rows):raise ValueError('Duplicate evidence identity')
    out=audit.STATE/'refinement-recorded-evidence-v1.parquet'
    pq.write_table(pa.Table.from_pylist(sorted(rows,key=lambda x:x['record_id'])),out,compression='zstd')
    summary={'counts':dict(counts),'sidecar_sha256':parent.base.file_sha(out),'sidecar_bytes':out.stat().st_size,
        'historical_recorded_error_followup_is_word_heuristic':True,
        'strict_tags_are_recorded_error_retry_sequences_not_reward_or_verified_repair':True,
        'retained_rows_only':True,'original_payloads_and_targets_unchanged':True}
    parent.base.save(audit.STATE/'refinement-evidence-summary.json',summary);print(json.dumps(summary),flush=True)
