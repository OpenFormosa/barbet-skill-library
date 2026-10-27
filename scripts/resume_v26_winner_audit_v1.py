"""Finish only unstarted winner reviews; interrupted first requests stay held."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.error

from scripts import audit_memory_sft_v2 as reviewer
from scripts.build_v26_complete_release import STATE
from scripts import build_memory_sft_v2 as base


def hold(job, reason):
    result = {'id': job['id'], 'record_id': job['record_id'], 'domain': job['domain'],
        'verdict': 'HOLD', 'checks': [], 'issues': [reason], 'usage': None,
        'orchestration_hold': True, 'not_a_model_semantic_verdict': True,
        'first_request_preserved_without_replay': True}
    reviewer.dump(STATE / 'semantic-audit-v1' / job['id'] / 'disposition.json', result)
    return result


def resume_one(job):
    folder = STATE / 'semantic-audit-v1' / job['id']
    if (folder / 'disposition.json').exists():
        return json.loads((folder / 'disposition.json').read_text())
    if (folder / 'request.json').exists() and not (folder / 'receipt.json').exists():
        return hold(job, 'Interrupted first request has no completion receipt. The original run recorded a local two-slot wait timeout; preserve this unresolved first attempt and do not replay it.')
    if (folder / 'receipt.json').exists() and not (folder / 'request.json').exists():
        raise ValueError('response receipt without its original request')
    try:
        return reviewer.review(job)
    except (TimeoutError, urllib.error.URLError) as exc:
        return hold(job, 'First request transport/slot failure, no replay: ' + type(exc).__name__)


def main():
    reviewer.STATE = STATE
    root = STATE / 'semantic-audit-v1'; plan = json.loads((root / 'plan.json').read_text())
    if (root / 'summary.json').exists(): raise ValueError('winner diagnosis already complete')
    if (plan['prompt_sha256'] != reviewer.digest(reviewer.PROMPT)
            or plan['schema_sha256'] != reviewer.digest(reviewer.SCHEMA)
            or plan['revision'] != reviewer.REV):
        raise ValueError('first-review protocol changed')
    controls = json.loads((root / 'controls.json').read_text())
    if not controls['passed']: raise ValueError('original controls did not pass')
    if len(plan['jobs']) != 32: raise ValueError('unexpected frozen winner plan')
    reviewer.dump(root / 'resume-policy-v1.json', {
        'original_plan_sha256': base.file_sha(root / 'plan.json'),
        'original_exception_log_sha256': base.file_sha(base.WORK / 'reports/v26-complete-first-winner-audit-v1.log'),
        'completed_verdicts_not_replayed': True, 'missing_receipt_first_attempts_held': True,
        'only_genuinely_unstarted_jobs_may_send_first_requests': True,
        'semantic_audit_cohorts_now_run_sequentially_to_avoid_slot_starvation': True})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resume_one, plan['jobs']))
    counts = Counter(r['verdict'] for r in results)
    reviewer.dump(root / 'summary.json', {'controls': controls['results'], 'results': results,
        'counts': dict(counts), 'excluded_record_ids': [r['record_id'] for r in results if r['verdict'] != 'ACCEPT'],
        'scope': plan['scope'] + ' Planned samples include explicitly separated infrastructure/incomplete-output holds; these are not semantic approvals.',
        'planned_samples': len(plan['jobs']),
        'completed_model_response_receipts': sum((root / j['id'] / 'receipt.json').exists() for j in plan['jobs']),
        'valid_semantic_verdicts': sum(counts[v] for v in ['ACCEPT', 'REVISE', 'REJECT']),
        'orchestration_holds': sum(bool(r.get('orchestration_hold')) for r in results),
        'all_first_requests_and_original_negative_outcomes_preserved': True})
    print(json.dumps({'counts': dict(counts), 'scope': '32 planned samples; incomplete/orchestration holds explicitly retained'}), flush=True)


if __name__ == '__main__': main()
