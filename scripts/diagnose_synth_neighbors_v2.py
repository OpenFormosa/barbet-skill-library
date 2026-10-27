"""New, previously unreviewed SYNTH cases for bounded neighboring evidence.

The 24-article pool was fixed before the v1 reviews. All 16 v1 questions are
excluded, including the six accepts; negative decisions are never retried.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from barbet_inference.evidence_retrieval import retrieve_with_neighbors
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import Rejected, RESERVED, augment_system, normalize_messages
from scripts import audit_memory_sft_v2 as reviewer
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts import package_memory_sft_v2 as package
from scripts import synth_wikipedia_memory_v1 as previous
from scripts.diagnose_synth_wikipedia_v1 import SOURCE_RAW, COLUMNS
from scripts.inventory_synth_grounded_memory_v1 import content_words, NUMBERS

STATE = base.WORK / 'reports/memory-harness-v2-synth-neighbors-v2'
RETRIEVER = base.REPO / 'barbet_inference/evidence_retrieval.py'


def prepare():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('new source cohort already frozen')
    original_rows = [json.loads(line) for line in (previous.STATE / 'normalized/pilot-synth-wikipedia.jsonl').open()]
    reviewed_queries = {r['query_hash'] for r in original_rows}
    article_list = json.loads((previous.STATE / 'article-fetch.json').read_text())['articles']
    articles = {a['title']: a['article'] for a in article_list if 'article' in a}
    documents = {t: previous.article_documents(a) for t, a in articles.items()}
    ni.init(); base.init_tokenizer(); package.init()
    tokenizer = Tokenizer.from_file(str(base.WORK / 'data/blobs/tokenizer/tokenizer.json'))
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    skills = SkillLibrary(base.REPO).render(base.ROUTES['reading'])
    prior_root = base.WORK / 'reports/memory-harness-v2-opencode-v1'
    prior = sqlite3.connect(f'file:{prior_root / "prior-query-aliases-v1.sqlite"}?mode=ro', uri=True)
    coding = sqlite3.connect(f'file:{prior_root / "v26-selection-v1/candidates.sqlite"}?mode=ro', uri=True)
    pools = defaultdict(dict); counts = Counter(); source_files = json.loads((SOURCE_RAW / 'lock.json').read_text())
    candidate_diagnostics = []
    origin = 'query_only_retrieval_or_full_evidence_from_pinned_source_linked_articles'
    for source in source_files:
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('raw source changed')
        offset = 0
        for batch in pq.ParquetFile(source['local']).iter_batches(columns=COLUMNS, batch_size=2048):
            for raw in batch.to_pylist():
                index = offset; offset += 1; counts['raw_rows'] += 1
                if raw['language'] != 'en' or raw['exercise'].lower() != 'memorization' or raw['seed_license'] != 'CC-By-SA (4.0)': continue
                try:
                    titles = {previous.wikipedia_title(raw['query_seed_url'])}
                    if raw['additional_seed_url']: titles.add(previous.wikipedia_title(raw['additional_seed_url']))
                except Rejected: continue
                if not titles <= set(articles): continue
                counts['source_rows_with_all_articles_already_pinned'] += 1
                qh = base.sha(base.norm(raw['query']))
                if qh in reviewed_queries: counts['held:already_reviewed_query'] += 1; continue
                if not raw['query'].strip() or not raw['synthetic_answer'].strip() or RESERVED.search(raw['query'] + raw['synthetic_answer']):
                    counts['held:invalid_source_text'] += 1; continue
                if prior.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (qh,)).fetchone() or coding.execute('SELECT 1 FROM candidates WHERE query_hash=? LIMIT 1', (qh,)).fetchone():
                    counts['held:prior_query'] += 1; continue
                if ni.benchmark_overlap(raw['query']): counts['held:local_evaluation_query'] += 1; continue
                ordered_titles = sorted(titles)
                docs = [d for t in ordered_titles for d in documents[t]]
                selected, trace = retrieve_with_neighbors(raw['query'], docs, token_count,
                    max_tokens=12000, top_k=8, neighbor_radius=1, target_chunk_chars=2400,
                    observation_origin=origin)
                if not selected: counts['held:no_evidence_fits'] += 1; continue
                text = raw['query'] + '\n' + '\n'.join(d.text for d in selected)
                words = content_words(raw['synthetic_answer'])
                coverage = len(words & content_words(text)) / max(1, len(words))
                missing_numbers = sorted(set(NUMBERS.findall(raw['synthetic_answer'])) - set(NUMBERS.findall(text)))
                if coverage < 0.70 or missing_numbers:
                    counts['held:offline_evidence_coverage_gate'] += 1; continue
                if any(ni.benchmark_overlap(d.text) for d in selected): counts['held:local_evaluation_evidence'] += 1; continue
                counts['candidate_answers_after_lexical_gate_not_semantic_verification'] += 1
                cohort = 'one_article' if len(titles) == 1 else 'two_articles'
                order = base.sha('synth-neighbors-first-new-case-diagnosis:' + qh); pool = pools[cohort]
                if qh in pool or (len(pool) >= 8 and order >= max(x[0] for x in pool.values())): continue
                row = previous.adapt(raw, source, index, [articles[t] for t in ordered_titles], skills)
                items = [MemoryItem(d.id, d.title + '\n' + d.text, d.source, observation_origin=origin) for d in selected]
                prompt = compose_memory_prompt(raw['query'], items, placement=row['memory_placement'])
                row['messages'] = augment_system(normalize_messages([{'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': raw['synthetic_answer']}], [], {}), skills[0])
                row['record_id'] = base.sha(canonical({'version': 'memory-harness-sft-v2', 'messages': row['messages'], 'tools': []}))
                row['memory'] = {'documents': [item.__dict__ for item in items]}; row['observation_origin'] = origin
                row['actual_retrieval'] = trace['method'] == 'question_lexical_with_neighbors'
                row['source_refs'] = copy.deepcopy(row['source_refs'])
                row['source_refs']['retrieval'] = {**trace, 'query': raw['query'],
                    'pool': 'one or two source-linked articles from the 24 articles fixed before v1 reviews',
                    'retriever_sha256': base.file_sha(RETRIEVER)}
                row['source_refs']['offline_quality_gate'] = {'content_word_coverage': coverage,
                    'minimum_content_word_coverage': 0.70, 'unsupported_numeric_literals': missing_numbers,
                    'entailment_verified': False, 'answer_not_used_for_retrieval': True}
                pool[qh] = (order, row)
                if len(pool) > 8: del pool[max(pool, key=lambda q: pool[q][0])]
        print(json.dumps({'source_file': source['path'], 'counts': dict(counts)}), flush=True)
    rows = [r for cohort in sorted(pools) for _, r in sorted(pools[cohort].values(), key=lambda x: x[0])]
    selected_rows = []; budgets = []; held = []
    for row in rows:
        try:
            record, _, _ = base.compile_row(row); budget = package.request_budget(record)
            payload = {'messages': row['messages'], 'tools': [], 'observation_origin': row['observation_origin']}
            messages = [{'role': 'system', 'content': reviewer.PROMPT}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
            text = package.CHAT.render(messages=messages, tools=[], add_generation_prompt=True, enable_thinking=False, bos_token='', eos_token='<|im_end|>')
            tokens = len(package.QWEN.encode(text, add_special_tokens=False).ids)
            if tokens + 900 > 1000000: raise Rejected('served_audit_budget')
            budget.update(audit_input_tokens=tokens, audit_output_reserve=900,
                pangolin_tokens=record['pangolin_tokens'], minicpm_tokens=record['minicpm_tokens'])
        except Rejected as exc: held.append({'query_hash': row['query_hash'], 'reason': str(exc)}); continue
        selected_rows.append(row); budgets.append(budget)
    prior.close(); coding.close()
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-synth-neighbors.jsonl').open('w') as f:
        for row in selected_rows: f.write(canonical(row) + '\n')
    base.save(STATE / 'pilot-budget-proof.json', budgets)
    original = json.loads((previous.STATE / 'semantic-audit-v1/plan.json').read_text())
    plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
    plan.update(jobs=[{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'reading',
        'messages': r['messages'], 'tools': [], 'observation_origin': r['observation_origin']} for r in selected_rows],
        scope='Previously unreviewed English memorization questions whose one or two source-linked articles are already in the prospectively frozen 24-article pool; neighboring whole-block retrieval and offline lexical/numeric gates. Up to eight per article-count cohort. Different cases and gates from v1, not a controlled accuracy comparison. No all-row approval.',
        first_non_accept_is_final_for_that_record=True)
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', plan)
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'retriever_sha256': base.file_sha(RETRIEVER),
        'v1_adapter_sha256': base.file_sha(Path(previous.__file__)), 'v1_first_audit_sha256': base.file_sha(previous.STATE / 'semantic-audit-v1/summary.json'),
        'article_fetch_sha256': base.file_sha(previous.STATE / 'article-fetch.json'),
        'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'auditor_sha256': base.file_sha(Path(reviewer.__file__)), 'counts': dict(counts),
        'first_reviewed_queries_excluded': sorted(reviewed_queries), 'budget_holds': held,
        'samples_prepared': len(selected_rows), 'accepted_training_episodes': 0,
        'source_programs_executed': False, 'model_weights_trained': False}
    base.save(STATE / 'pilot-summary.json', policy)
    print(json.dumps({'prepared': len(selected_rows), 'counts': dict(counts), 'budget_holds': len(held)}), flush=True)


def audit():
    policy = json.loads((STATE / 'pilot-summary.json').read_text())
    for key, path in [('driver_sha256', Path(__file__)), ('retriever_sha256', RETRIEVER),
        ('v1_adapter_sha256', Path(previous.__file__)), ('plan_sha256', STATE / 'semantic-audit-v1/plan.json'),
        ('auditor_sha256', Path(reviewer.__file__))]:
        if base.file_sha(path) != policy[key]: raise ValueError('new first-diagnosis binding changed')
    if not policy['samples_prepared']: raise ValueError('no eligible new diagnostic cases')
    reviewer.STATE = STATE; reviewer.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'audit'])
    args = parser.parse_args(); globals()[args.phase]()
