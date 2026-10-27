"""Stage licensed official-train NI tasks as causal memory/skill candidates.

Staged rows are not accepted episodes. Multiple references still require v26
ranking, first-review holds, global identity/split checks and export validation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3

from scripts import build_memory_sft_v2 as base
from scripts.extend_v26_overlap_selection import STATE as PRIOR
from scripts.natural_instructions_memory_adapter import adapt, task_contract, exact_openorca_views
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected, augment_system, normalize_messages

STATE = base.WORK / 'reports/memory-harness-v2-natural-instructions-v1'
RAW = base.RAW / 'scale10m-natural-instructions'
FIRST = base.WORK / 'reports/memory-harness-v2'


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'normalize-jobs.json').exists():
        raise ValueError('NI source preparation already frozen')
    refs = base.RAW / 'scale10m-reference'
    train_path = refs / 'natural-instructions-splits--default--train_tasks.txt'
    test_path = refs / 'natural-instructions-splits--default--test_tasks.txt'
    train, test = set(train_path.read_text().split()), set(test_path.read_text().split())
    if train & test: raise ValueError('official task split overlaps')
    counts = Counter(); jobs = []
    for source in json.loads((RAW / 'lock.json').read_text()):
        if Path(source['path']).stem not in train or Path(source['path']).stem in test:
            raise ValueError('nontraining task in source lock')
        if base.file_sha(source['local']) != source['sha256']:
            raise ValueError('pinned task bytes changed')
        task = json.loads(Path(source['local']).read_text())
        counts['raw_instances_seen'] += len(task['Instances'])
        try: task_contract(task)
        except Rejected as exc:
            counts['held_instances:' + str(exc)] += len(task['Instances']); continue
        jobs.append({'job_id': f'{len(jobs):05d}', 'source': source})
        counts['licensed_in_scope_instances'] += len(task['Instances'])
    db = sqlite3.connect(STATE / 'prior-query-aliases.sqlite')
    db.execute('CREATE TABLE prior_queries(query_hash TEXT PRIMARY KEY)')
    db.execute('ATTACH DATABASE ? AS previous', (str(PRIOR / 'candidates.sqlite'),))
    db.execute('INSERT OR IGNORE INTO prior_queries SELECT query_hash FROM previous.candidates')
    original = json.loads((FIRST / 'exclusions.private.json').read_text())
    prior_holds = json.loads((PRIOR / 'candidate-holds.private.json').read_text())
    extras = set(original['old_query_hashes']) | set(prior_holds['query_hashes']) | set(prior_holds['supplemental_prior_queries'])
    db.executemany('INSERT OR IGNORE INTO prior_queries VALUES (?)', [(q,) for q in extras]); db.commit()
    counts['prior_query_aliases'] = db.execute('SELECT COUNT(*) FROM prior_queries').fetchone()[0]
    db.close()
    shutil.copyfile(FIRST / 'exclusions.private.json', STATE / 'exclusions.private.json')
    base.save(STATE / 'normalize-jobs.json', jobs)
    base.save(STATE / 'build-source-lock.json', {'files': [j['source'] for j in jobs],
        'train_task_split_sha256': base.file_sha(train_path), 'test_task_split_sha256': base.file_sha(test_path)})
    base.save(STATE / 'preparation-summary.json', {'counts': dict(counts), 'tasks': len(jobs),
        'official_test_tasks_excluded': len(test), 'upstream_reference_outputs_are_not_prompt_memory': True,
        'answer_alternatives_require_minimum_v26_ppl': True, 'counting_status': 'CANDIDATES_ONLY_NOT_ACCEPTED',
        'adapter_sha256': base.file_sha(base.REPO / 'scripts/natural_instructions_memory_adapter.py'),
        'builder_sha256': base.file_sha(Path(__file__)), 'model_weights_trained': False})
    print(canonical({'tasks': len(jobs), 'counts': dict(counts)}), flush=True)


def init():
    global EX, PREVIOUS, SKILLS
    raw = json.loads((STATE / 'exclusions.private.json').read_text())
    EX = {k: set(raw[k]) for k in ['benchmark_query_hashes', 'benchmark_13gram_hashes']}
    PREVIOUS = sqlite3.connect(f'file:{STATE / "prior-query-aliases.sqlite"}?mode=ro', uri=True)
    library = SkillLibrary(base.REPO)
    SKILLS = {d: library.render(base.ROUTES[d]) for d in ['reading', 'coding']}


def benchmark_overlap(text):
    return (base.sha(base.norm(text)) in EX['benchmark_query_hashes']
            or any(base.sha(g) in EX['benchmark_13gram_hashes'] for g in base.ngrams(text)))


def normalize_instance(task, instance, source, index):
    url = f"https://github.com/{source['repo']}/blob/{source['revision']}/{source['path']}"
    v = adapt(task, instance, source_url=url)
    aliases = {base.sha(base.norm(q)) for q in exact_openorca_views(v['definition'], v['target_input'])}
    aliases.update([base.sha(base.norm(v['query'])), base.sha(base.norm(v['target_input']))])
    placeholders = ','.join('?' for _ in aliases)
    if PREVIOUS.execute(f'SELECT 1 FROM prior_queries WHERE query_hash IN ({placeholders}) LIMIT 1', tuple(aliases)).fetchone():
        raise Rejected('prior_question_or_known_openorca_wrapper')
    if benchmark_overlap(v['query']) or benchmark_overlap(v['target_input']):
        raise Rejected('local_evaluation_overlap')
    for n in v['worked_example_indices']:
        if benchmark_overlap(task['Positive Examples'][n]['input']):
            raise Rejected('worked_example_evaluation_overlap')
    domain = v['domain']; result = []
    for answer in v['answers']:
        raw = [{'role': 'user', 'content': v['prompt']}, {'role': 'assistant', 'content': answer}]
        messages = augment_system(normalize_messages(raw, [], {}), SKILLS[domain][0])
        rid = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': []}))
        ref = {'repo': source['repo'], 'revision': source['revision'], 'file': source['path'],
            'file_sha256': source['sha256'], 'row_index': index, 'upstream_id': instance['id'],
            'upstream_repo': source['repo'], 'upstream_source_names': source['upstream_source'],
            'upstream_urls': source['upstream_urls'], 'license': v['license'], 'attribution_url': url,
            'raw_row_sha256': base.sha(canonical(instance)), 'new_semantic_generation': False,
            'reference_index': instance['output'].index(answer), 'reference_alternatives': len(v['answers']),
            'source_task_sha256': v['source_task_key'], 'categories': v['categories'],
            'worked_example_indices': v['worked_example_indices'],
            'target_or_duplicate_example_indices_excluded': v['target_or_duplicate_example_indices_excluded'],
            'target_answers_in_memory': False,
            'transformations': ['source task definition retained as explicit task instruction',
                'different-input public worked examples and complete target input in causal memory',
                'one deterministic memory placement per task input, not duplicated views',
                'all distinct reference outputs staged for later v26 ranking', 'skill prefill', 'assistant-only loss']}
        result.append({'record_id': rid, 'family_id': v['family_key'], 'source_family_id': v['family_key'],
            'template_family_id': base.sha('query-family:' + base.norm(v['query'], numbers=True)),
            'source_task_key': v['source_task_key'], 'query_hash': base.sha(base.norm(v['query'])),
            'split': 'unassigned', 'domain': domain, 'source': source['repo'],
            'upstream_source': '|'.join(source['upstream_source']), 'license': v['license'],
            'messages': messages, 'tools': [], 'memory': v['memory'], 'source_refs': ref,
            'skill_ids': base.ROUTES[domain], 'skills': SKILLS[domain][1],
            'verification_tier': 'upstream_instruction_reference_not_locally_semantic_verified',
            'observation_origin': 'provided_document_and_public_worked_examples',
            'memory_placement': v['memory_placement'], 'actual_retrieval': False,
            'recorded_error_followup': False, 'locally_executed_source_program': False,
            'assistant_turns': 1, 'tool_observations': 0})
    return result


def job(spec):
    name = ('pilot-' if spec.get('pilot') else '') + spec['job_id']
    path = STATE / 'normalized' / (name + '.jsonl'); receipt = path.with_suffix('.report.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    source = spec['source']
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('source changed')
    task = json.loads(Path(source['local']).read_text()); instances = task['Instances']
    if spec.get('pilot'): instances = instances[:12]
    counts = Counter(); path.parent.mkdir(exist_ok=True)
    with path.open('w') as output, path.with_name(name + '.excluded.jsonl').open('w') as held:
        for n, instance in enumerate(instances):
            counts['source_instances_seen'] += 1
            try: rows = normalize_instance(task, instance, source, n)
            except Rejected as exc:
                counts['held:' + str(exc)] += 1
                held.write(canonical({'row_index': n, 'upstream_id': instance['id'], 'reason': str(exc)}) + '\n'); continue
            counts['candidate_tasks'] += 1
            for row in rows:
                output.write(canonical(row) + '\n'); counts['candidate_answers'] += 1
    result = {'job_id': name, 'counts': dict(counts), 'path': str(path), 'sha256': base.file_sha(path)}
    base.save(receipt, result); return result


def normalize(workers, pilot):
    binding = json.loads((STATE / 'preparation-summary.json').read_text())
    if binding['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/natural_instructions_memory_adapter.py'):
        raise ValueError('frozen NI adapter changed')
    jobs = json.loads((STATE / 'normalize-jobs.json').read_text())
    jobs = [{**j, 'pilot': pilot} for j in jobs]; counts = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(job, jobs), 1):
            counts.update(result['counts'])
            if n % 25 == 0 or n == len(jobs):
                print(canonical({'tasks': n, 'total': len(jobs), 'counts': dict(counts)}), flush=True)
    base.save(STATE / ('pilot-normalize-summary.json' if pilot else 'normalize-summary.json'),
              {'counts': dict(counts), 'accepted_training_episodes': 0, 'needs_v26_ranking_and_release_gates': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'normalize'])
    p.add_argument('--workers', type=int, default=8); p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 8: raise ValueError('NI staging CPU worker bound')
    if args.phase == 'prepare': prepare()
    else: normalize(args.workers, args.pilot)
