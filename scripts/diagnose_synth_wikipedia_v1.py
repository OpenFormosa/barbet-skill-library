"""First source diagnosis for SYNTH with real, pinned article retrieval.

Select by question hash before looking at answer coverage or review results.
No source answer, generated reasoning or script is used to retrieve evidence.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from scripts import audit_memory_sft_v2 as reviewer
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts import package_memory_sft_v2 as package
from scripts.inventory_synth_grounded_memory_v1 import content_words
from scripts.synth_wikipedia_memory_v1 import STATE, RAW, wikipedia_title, fetch_article, adapt

SOURCE_RAW = base.RAW / 'scale10m-synth-pilot'
COLUMNS = ['synth_id', 'language', 'exercise', 'query', 'query_seed_url', 'query_seed_text',
           'additional_seed_url', 'seed_license', 'synthetic_answer']
ADAPTER = base.REPO / 'scripts/synth_wikipedia_memory_v1.py'


def select():
    STATE.mkdir(parents=True, exist_ok=True)
    if (STATE / 'selection.json').exists(): raise ValueError('source sample already frozen')
    ni.init(); pools = defaultdict(dict); counts = Counter()
    prior = sqlite3.connect('file:' + str(base.WORK / 'reports/memory-harness-v2-opencode-v1/prior-query-aliases-v1.sqlite') + '?mode=ro', uri=True)
    opencode = sqlite3.connect('file:' + str(base.WORK / 'reports/memory-harness-v2-opencode-v1/v26-selection-v1/candidates.sqlite') + '?mode=ro', uri=True)
    sources = json.loads((SOURCE_RAW / 'lock.json').read_text())
    for source in sources:
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('source file changed')
        offset = 0
        for batch in pq.ParquetFile(source['local']).iter_batches(columns=COLUMNS, batch_size=1024):
            for raw in batch.to_pylist():
                row_index = offset; offset += 1; counts['raw_rows'] += 1
                if raw['language'] != 'en' or raw['exercise'].lower() not in {'memorization', 'rag'}: continue
                counts['english_memory_or_rag'] += 1
                if raw['seed_license'] != 'CC-By-SA (4.0)':
                    counts['held:seed_license_outside_pilot'] += 1; continue
                try:
                    titles = {wikipedia_title(raw['query_seed_url'])}
                    if raw['additional_seed_url']: titles.add(wikipedia_title(raw['additional_seed_url']))
                except Rejected as exc: counts['held:' + str(exc)] += 1; continue
                if not all(isinstance(raw[k], str) and raw[k].strip() for k in ['query', 'synthetic_answer', 'query_seed_text']):
                    counts['held:missing_source_text'] += 1; continue
                qh = base.sha(base.norm(raw['query'])); order = base.sha('synth-pinned-wikipedia-first-diagnosis:' + qh)
                cohort = raw['exercise'].lower() + ('/two_articles' if len(titles) == 2 else '/one_article')
                pool = pools[cohort]
                if order in pool or (len(pool) >= 8 and order >= max(pool)): continue
                if prior.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (qh,)).fetchone() or opencode.execute('SELECT 1 FROM candidates WHERE query_hash=? LIMIT 1', (qh,)).fetchone():
                    counts['held:prior_query_among_sample_contenders'] += 1; continue
                if ni.benchmark_overlap(raw['query']) or ni.benchmark_overlap(raw['query_seed_text']):
                    counts['held:local_evaluation_overlap_among_contenders'] += 1; continue
                pool[order] = {'source': source, 'row_index': row_index, 'raw': raw, 'query_hash': qh,
                               'cohort': cohort, 'titles': sorted(titles)}
                if len(pool) > 8: del pool[max(pool)]
        print(json.dumps({'source_file': source['path'], 'rows_seen': counts['raw_rows'],
                          'sample_candidates': sum(map(len, pools.values()))}), flush=True)
    selected = [v for k in sorted(pools) for _, v in sorted(pools[k].items())]
    # Selection keys use question hashes; deduplicate a repeated source question
    # across source-link cohorts without drawing a replacement based on answers.
    seen = set(); unique = []
    for row in selected:
        if row['query_hash'] not in seen: unique.append(row); seen.add(row['query_hash'])
    prior.close(); opencode.close()
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'adapter_sha256': base.file_sha(ADAPTER),
        'source_lock_sha256': base.file_sha(SOURCE_RAW / 'lock.json'), 'counts': dict(counts),
        'sample_count': len(unique), 'cohorts': dict(Counter(r['cohort'] for r in unique)),
        'selection': 'Lowest deterministic query-hash ranks, at most eight per exercise/article-count cohort across four actual source files; no answer overlap, perplexity or reviewer feedback used.',
        'answer_version_selection': 'Diagnostic source sample only; not accepted standard answers or PPL winners.',
        'source_linked_candidate_pool_disclosed': True, 'accepted_training_episodes': 0,
        'weights_trained': False, 'source_programs_executed': False,
        'source_card_reported_samples': 79648272,
        'source_card_count_is_not_locally_verified': True}
    base.save(STATE / 'selection.json', {'policy': policy, 'samples': unique})
    base.save(STATE / 'build-source-lock.json', {'files': sources})
    print(json.dumps(policy), flush=True)


def verify_binding():
    selected = json.loads((STATE / 'selection.json').read_text()); policy = selected['policy']
    if policy['driver_sha256'] != base.file_sha(Path(__file__)) or policy['adapter_sha256'] != base.file_sha(ADAPTER):
        raise ValueError('frozen first-diagnosis code changed')
    if policy['source_lock_sha256'] != base.file_sha(SOURCE_RAW / 'lock.json'): raise ValueError('source lock changed')
    return selected


def fetch():
    selected = verify_binding(); RAW.mkdir(parents=True, exist_ok=True)
    titles = sorted({title for row in selected['samples'] for title in row['titles']})
    def one(title):
        try: return {'title': title, 'article': fetch_article(title)}
        except Exception as exc:
            value = {'title': title, 'held': type(exc).__name__, 'message': str(exc)[:500]}
            path = RAW / base.sha(title) / 'first-fetch-failure.json'
            if not path.exists(): base.save(path, value)
            return value
    result = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(one, titles):
            result.append(item)
            print(json.dumps({'title': item['title'], 'status': 'fetched' if 'article' in item else 'held'}), flush=True)
    path = STATE / 'article-fetch.json'
    if path.exists(): raise ValueError('first source fetch manifest already frozen')
    base.save(path, {'articles': result, 'selection_sha256': base.file_sha(STATE / 'selection.json'),
                     'concurrent_requests_at_most': 4})


def prepare():
    selected = verify_binding()
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('retrieval pilot already prepared')
    fetched = json.loads((STATE / 'article-fetch.json').read_text())
    if fetched['selection_sha256'] != base.file_sha(STATE / 'selection.json'): raise ValueError('sample changed')
    articles = {v['title']: v['article'] for v in fetched['articles'] if 'article' in v}
    skills = SkillLibrary(base.REPO).render(base.ROUTES['reading'])
    ni.init(); base.init_tokenizer(); package.init()
    rows = []; held = []; budgets = []; coverage = []
    for sample in selected['samples']:
        try:
            if any(t not in articles for t in sample['titles']): raise Rejected('source_article_unavailable')
            row = adapt(sample['raw'], sample['source'], sample['row_index'],
                        [articles[t] for t in sample['titles']], skills)
            if any(ni.benchmark_overlap(m['text']) for m in row['memory']['documents']):
                raise Rejected('retrieved_evidence_local_evaluation_overlap')
            record, _, _ = base.compile_row(row)
            budget = package.request_budget(record)
            payload = {'messages': row['messages'], 'tools': [], 'observation_origin': row['observation_origin']}
            audit_messages = [{'role': 'system', 'content': reviewer.PROMPT},
                              {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
            text = package.CHAT.render(messages=audit_messages, tools=[], add_generation_prompt=True,
                                       enable_thinking=False, bos_token='', eos_token='<|im_end|>')
            n = len(package.QWEN.encode(text, add_special_tokens=False).ids)
            if n + 900 > 1000000: raise Rejected('audit_request_exceeds_served_context')
            budget.update(audit_input_tokens=n, audit_output_reserve=900,
                          audit_render_sha256=base.sha(text),
                          pangolin_tokens=record['pangolin_tokens'], minicpm_tokens=record['minicpm_tokens'])
            budgets.append(budget); rows.append(row)
            raw = sample['raw']; answer_words = content_words(raw['synthetic_answer'])
            before = content_words(raw['query_seed_text'] + '\n' + raw['query'])
            after = content_words(raw['query'] + '\n' + '\n'.join(v['text'] for v in row['memory']['documents']))
            coverage.append({'record_id': row['record_id'], 'query_hash': row['query_hash'], 'cohort': sample['cohort'],
                'seed_only_content_word_coverage': len(answer_words & before) / max(1, len(answer_words)),
                'retrieved_content_word_coverage': len(answer_words & after) / max(1, len(answer_words)),
                'diagnostic_only_not_an_entailment_test': True})
        except Rejected as exc: held.append({'query_hash': sample['query_hash'], 'reason': str(exc)})
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-synth-wikipedia.jsonl').open('w') as handle:
        for row in rows: handle.write(canonical(row) + '\n')
    base.save(STATE / 'pilot-summary.json', {'samples_prepared': len(rows), 'holds': held,
        'accepted_training_episodes': 0, 'source_article_count': len(articles),
        'all_questions_and_targets_unchanged': True, 'answer_used_for_retrieval': False,
        'source_linked_article_pool_only': True, 'coverage_diagnostics': coverage,
        'all_training_and_review_requests_budgeted_without_truncation': True})
    base.save(STATE / 'pilot-budget-proof.json', budgets)
    original = json.loads((base.WORK / 'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
    plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
    plan.update(jobs=[{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'reading',
                      'messages': r['messages'], 'tools': [], 'observation_origin': r['observation_origin']} for r in rows],
        first_non_accept_is_final_for_that_record=True,
        scope='At most 32 source cases selected prospectively by question hash across four files, with actual question-only retrieval from one or two pinned source-linked Wikipedia articles. Feasibility diagnosis only; not whole-corpus accuracy, all-row verification, global Wikipedia retrieval or PPL winner review.')
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', plan)
    base.save(STATE / 'audit-binding.json', {'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'auditor_sha256': base.file_sha(Path(reviewer.__file__)), 'normalized_sha256': base.file_sha(folder / 'pilot-synth-wikipedia.jsonl')})
    print(json.dumps({'prepared': len(rows), 'held': len(held), 'articles': len(articles)}), flush=True)


def audit():
    verify_binding(); binding = json.loads((STATE / 'audit-binding.json').read_text())
    for key, path in [('plan_sha256', STATE / 'semantic-audit-v1/plan.json'),
                      ('auditor_sha256', Path(reviewer.__file__)),
                      ('normalized_sha256', STATE / 'normalized/pilot-synth-wikipedia.jsonl')]:
        if base.file_sha(path) != binding[key]: raise ValueError('first audit binding changed')
    reviewer.STATE = STATE; reviewer.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['select', 'fetch', 'prepare', 'audit'])
    args = parser.parse_args(); globals()[args.phase]()
