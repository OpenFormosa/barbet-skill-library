"""Prospective full staging after the first NI diagnosis, preserving all holds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

from scripts import build_natural_instructions_memory_v1 as stage
from scripts import build_memory_sft_v2 as base
from barbet_inference.sft import Rejected

STATE = stage.STATE
ORIGINAL_NORMALIZE = stage.normalize_instance
ORIGINAL_INIT = stage.init
UNSUPPORTED_FACT_TASKS = {
    'task167_strategyqa_question_generation.json': 'Requires generating new multi-fact questions beyond the supplied term description; the additional facts are not supplied.',
    'task471_haspart_answer_generation.json': 'Requires entity-specific part/whole facts from a bare entity name without a supplied fact collection.',
    'task472_haspart_classification.json': 'Requires entity-specific part/whole facts from two entity names without a supplied fact collection.',
    'task897_freebase_qa_topic_question_generation.json': 'Requires factual trivia generation from a topic name, without supplied supporting facts.',
    'task898_freebase_qa_answer_generation.json': 'Bare factual questions explicitly refer to external resources, but their contents are absent.',
    'task901_freebase_qa_category_question_generation.json': 'A category such as film is not evidence for a factual trivia question; external resources are mentioned but not provided.'}


def topic_is_literal(target, answer):
    text = base.norm(target); phrase = base.norm(answer).strip(' .!?\"\'')
    return bool(phrase and re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text))


def prepare():
    path = STATE / 'full-normalization-policy-v2.json'
    if path.exists(): return json.loads(path.read_text())
    audit_path = STATE / 'semantic-audit-v1/summary.json'
    audit = json.loads(audit_path.read_text()); bad = set(audit['excluded_record_ids']); held_queries = set()
    for p in (STATE / 'normalized').glob('pilot-*.jsonl'):
        if p.name.endswith('.excluded.jsonl'): continue
        for line in p.open():
            r = json.loads(line)
            if r['record_id'] in bad: held_queries.add(r['query_hash'])
    jobs = []; excluded = []; counts = Counter()
    for j in json.loads((STATE / 'normalize-jobs.json').read_text()):
        source = j['source']; name = Path(source['path']).name
        if name in UNSUPPORTED_FACT_TASKS:
            excluded.append({'file': source['path'], 'sha256': source['sha256'],
                             'instances': source['instances'], 'reason': UNSUPPORTED_FACT_TASKS[name]})
            counts['held_missing_facts_raw_instances'] += source['instances']
        else:
            jobs.append({**j, 'pilot': False}); counts['raw_instances_to_stage'] += source['instances']
    policy = {'version': 'ni-contextual-memory-full-v2', 'first_diagnosis_sha256': base.file_sha(audit_path),
        'first_diagnosis_counts': audit['counts'], 'held_query_hashes': sorted(held_queries),
        'excluded_task_files': excluded, 'counts': dict(counts),
        'topic_extraction_gate': 'For task899 only, every reference topic must occur as a normalized complete phrase in the supplied question. Hold the source instance otherwise; do not rewrite references.',
        'false_positive_review_note': 'Some first non-accept reviews contain contradictory praise or ambiguous rubric complaints. They remain held; their counts are not a corpus error-rate estimate.',
        'script_sha256': base.file_sha(Path(__file__)),
        'candidate_only': True, 'accepted_training_rows': 0, 'no_negative_review_replay': True}
    base.save(path, policy); base.save(STATE / 'full-normalize-jobs-v2.json', jobs)
    print(json.dumps({'tasks': len(jobs), 'held_queries': len(held_queries), 'counts': dict(counts)}), flush=True)
    return policy


def init():
    global HELD, POLICY_SHA
    ORIGINAL_INIT()
    policy = json.loads((STATE / 'full-normalization-policy-v2.json').read_text())
    HELD = set(policy['held_query_hashes']); POLICY_SHA = base.file_sha(STATE / 'full-normalization-policy-v2.json')
    stage.normalize_instance = normalize_instance


def normalize_instance(task, instance, source, index):
    query_hash = base.sha(base.norm(task['Definition'][0] + '\n\n' + instance['input']))
    if query_hash in HELD: raise Rejected('first_source_diagnostic_nonaccept_query')
    if Path(source['path']).name == 'task899_freebase_qa_topic_generation.json':
        if any(not topic_is_literal(instance['input'], a) for a in instance['output']):
            raise Rejected('topic_reference_requires_unprovided_alias_fact')
    rows = ORIGINAL_NORMALIZE(task, instance, source, index)
    for r in rows: r['source_refs']['full_normalization_policy_sha256'] = POLICY_SHA
    return rows


def main(workers):
    policy = prepare()
    if policy['script_sha256'] != base.file_sha(Path(__file__)): raise ValueError('frozen full-stage policy code changed')
    jobs = json.loads((STATE / 'full-normalize-jobs-v2.json').read_text()); counts = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(stage.job, jobs), 1):
            counts.update(result['counts'])
            if n % 25 == 0 or n == len(jobs): print(json.dumps({'tasks': n, 'total': len(jobs), 'counts': dict(counts)}), flush=True)
    base.save(STATE / 'normalize-summary.json', {'counts': dict(counts), 'accepted_training_episodes': 0,
        'needs_v26_ranking_and_release_gates': True, 'full_normalization_policy_sha256': base.file_sha(STATE / 'full-normalization-policy-v2.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=8); args = p.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('CPU worker bound')
    main(args.workers)
