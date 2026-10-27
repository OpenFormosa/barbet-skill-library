"""First source diagnosis for contextual OpenMath candidates, without training.

The augmented expected answer is a majority vote, not an independent oracle.
All public solutions remain inert text. No source code or solution is executed.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pyarrow.parquet as pq
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED, augment_system, normalize_messages
from scripts import build_memory_sft_v2 as base
from scripts import audit_memory_sft_v2 as reviewer

STATE = base.WORK / 'reports/memory-harness-v2-openmath-v1'
RAW = base.RAW / 'scale10m-openmath'
SOURCES = {'augmented_gsm8k', 'augmented_math', 'gsm8k', 'math'}
KNOWN_BAD = {
    'Find the sum of all three-digit positive integers whose cubes end with the digits 125.':
        {'correct_answer': '11990', 'upstream_answer': '2000',
         'check': 'Exhaustive integer enumeration over 100..999 yields 22 integers, 125+40*k for k=0..21, with sum 11990.'},
    'What is the largest number, all of whose digits are 1 or 4, and whose digits add up to 12?':
        {'correct_answer': '111111111111', 'upstream_answer': '441111',
         'check': 'At most 12 digits because each is at least 1. Twelve ones satisfy the sum; every shorter positive decimal number is smaller.'}}


def adapt(raw, source, index, skills):
    problem = raw['problem']; answer = raw['generated_solution']; category = raw['problem_source']
    if category not in SOURCES or not all(isinstance(v, str) and v.strip() for v in [problem, answer, raw['expected_answer']]):
        raise Rejected('invalid_openmath_fields')
    if RESERVED.search(problem + answer): raise Rejected('literal_reserved_delimiter')
    if base.norm(problem) in {base.norm(p) for p in KNOWN_BAD}:
        raise Rejected('independently_confirmed_incorrect_majority_label')
    qh = base.sha(base.norm(problem)); placement = 'front' if int(qh[:8], 16) % 2 else 'back'
    url = f"https://huggingface.co/datasets/{source['repo']}/blob/{source['revision']}/{source['path']}"
    item = MemoryItem('problem', problem, url, observation_origin='user_problem_premises')
    prompt = compose_memory_prompt('Solve the complete math problem in memory item problem. Show a solution and a final answer.', [item], placement=placement)
    messages = augment_system(normalize_messages([{'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': answer}], [], {}), skills[0])
    rid = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': []}))
    family = base.sha('query-family:' + base.norm(problem, numbers=True))
    return {'record_id': rid, 'query_hash': qh, 'family_id': family, 'source_family_id': family,
        'template_family_id': family, 'source_task_key': qh, 'split': 'unassigned', 'domain': 'math',
        'source': source['repo'], 'upstream_source': source['repo'], 'license': 'cc-by-4.0',
        'messages': messages, 'tools': [], 'memory': {'documents': [item.__dict__]},
        'skill_ids': base.ROUTES['math'], 'skills': skills[1],
        'verification_tier': ('upstream_majority_vote_not_independent_math_verification' if category.startswith('augmented_')
                              else 'upstream_training_reference_not_locally_math_verified'),
        'observation_origin': 'user_problem_premises', 'memory_placement': placement,
        'actual_retrieval': False, 'recorded_error_followup': False, 'locally_executed_source_program': False,
        'assistant_turns': 1, 'tool_observations': 0,
        'source_refs': {'repo': source['repo'], 'revision': source['revision'], 'file': source['path'],
            'file_sha256': source['sha256'], 'row_index': index, 'license': 'cc-by-4.0',
            'attribution_url': url, 'raw_row_sha256': base.sha(canonical(raw)), 'problem_source': category,
            'expected_answer_sha256': base.sha(raw['expected_answer']),
            'expected_answer_origin': 'majority_vote' if category.startswith('augmented_') else 'upstream_training_gold',
            'expected_answer_in_prompt': False, 'new_semantic_generation': False,
            'transformations': ['complete problem as causal memory', 'one deterministic placement per input',
                                'unchanged public solution', 'skill prefill', 'assistant-only loss']}}


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('OpenMath first source pilot already frozen')
    skills = SkillLibrary(base.REPO).render(base.ROUTES['math'])
    files = json.loads((RAW / 'lock.json').read_text())
    if len(files) != 32 or any('/train-' not in f['path'] for f in files):
        raise ValueError('only complete main training split admitted, no overlapping subsplits')
    exclusions = json.loads((base.WORK / 'reports/memory-harness-v2/exclusions.private.json').read_text())
    exact = set(exclusions['benchmark_query_hashes']) | set(exclusions['old_query_hashes'])
    grams = set(exclusions['benchmark_13gram_hashes'])
    pools = defaultdict(dict); counts = Counter(); total_raw = 0
    for source in files:
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('source bytes changed')
        parquet = pq.ParquetFile(source['local']); total_raw += parquet.metadata.num_rows
        # This is explicitly a bounded first-source diagnostic, not a random
        # corpus-accuracy sample. Hash ordering selects among first 512 per file.
        rows = next(parquet.iter_batches(batch_size=512)).to_pylist()
        for index, raw in enumerate(rows):
            counts['raw_pilot_rows'] += 1; qh = base.sha(base.norm(raw['problem']))
            if qh in exact or any(base.sha(g) in grams for g in base.ngrams(raw['problem'])):
                counts['held:prior_or_evaluation_overlap'] += 1; continue
            try: row = adapt(raw, source, index, skills)
            except Rejected as exc:
                counts['held:' + str(exc)] += 1; continue
            pool = pools[raw['problem_source']]
            key = base.sha('openmath-source-first-diagnosis:' + qh)
            if qh not in pool or row['record_id'] < pool[qh][1]['record_id']: pool[qh] = (key, row)
            if len(pool) > 48:
                keep = sorted(pool, key=lambda q: pool[q][0])[:12]; pool = {q: pool[q] for q in keep}
                pools[raw['problem_source']] = pool
    selected = [r for pool in pools.values() for _, r in sorted(pool.values(), key=lambda x: x[0])[:12]]
    if set(pools) != SOURCES: raise ValueError('source diagnostic did not cover all four categories')
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-openmath.jsonl').open('w') as handle:
        for row in sorted(selected, key=lambda r: r['record_id']): handle.write(canonical(row) + '\n')
    base.save(STATE / 'known-incorrect-reference-holds-v1.json', {
        'holds': [{'problem': p, 'query_hash': base.sha(base.norm(p)), **v} for p, v in KNOWN_BAD.items()],
        'source_programs_executed': False, 'answers_not_rewritten': True,
        'scope': 'Two directly inspected counterexamples; not a corpus error-rate estimate.'})
    base.save(STATE / 'build-source-lock.json', {'files': files})
    base.save(STATE / 'pilot-summary.json', {'main_split_raw_rows': total_raw, 'diagnostic_rows': len(selected),
        'diagnostic_categories': dict(Counter(r['source_refs']['problem_source'] for r in selected)),
        'pilot_screening_counts': dict(counts), 'all_source_files_rehashed': True,
        'driver_sha256': base.file_sha(Path(__file__)), 'accepted_training_rows': 0,
        'all_alternative_answers_require_v26_ranking': True,
        'majority_vote_is_not_independent_ground_truth': True, 'weights_trained': False})
    print(json.dumps({'source_raw_rows': total_raw, 'diagnostic_rows': len(selected), 'counts': dict(counts)}), flush=True)


def audit():
    summary = json.loads((STATE / 'pilot-summary.json').read_text())
    if summary['driver_sha256'] != base.file_sha(Path(__file__)): raise ValueError('frozen diagnostic code changed')
    reviewer.STATE = STATE
    plan_path = STATE / 'semantic-audit-v1/plan.json'
    if plan_path.exists(): plan = json.loads(plan_path.read_text())
    else:
        # Reuse the established controls and exact reviewer protocol, while
        # declaring a separate first source cohort and its actual sample scope.
        original = json.loads((base.WORK / 'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
        rows = [json.loads(line) for line in (STATE / 'normalized/pilot-openmath.jsonl').open()]
        jobs = [{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'math',
                 'messages': r['messages'], 'tools': [], 'observation_origin': r['observation_origin']} for r in rows]
        plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
        plan.update(jobs=jobs, first_non_accept_is_final_for_that_record=True,
            scope='48 deterministic first-source cases, 12 distinct problems per source category, selected from first 512 rows of each of 32 files. Not a representative corpus-accuracy estimate or review of PPL winners.')
        reviewer.dump(plan_path, plan)
    reviewer.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'audit']); args = parser.parse_args()
    prepare() if args.phase == 'prepare' else audit()
