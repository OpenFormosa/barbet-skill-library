"""Measure a grounded-memory candidate slice; no admission or model claims.

Reference-answer overlap is an offline quality heuristic, never a retrieval
query. Only the upstream supplied seed text may become memory. Synthetic private
reasoning and executable scripts are neither inspected nor projected here.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from scripts import build_memory_sft_v2 as base

RAW = base.RAW / 'scale10m-synth-pilot'
STATE = base.WORK / 'reports/memory-harness-v2-synth-grounding-v1'
STOP = set('a an the is are was were be been being am to of in on at by for and or if so as with from that this these those it its they their them he she his her we our you your i me my not no yes can could would should will may might have has had do does did than then there here which who what when where how why each all any some also about more most such very only one two into over under through between after before during without within both other same different'.split())
WORDS = re.compile(r'[a-z]+(?:\'[a-z]+)?')
NUMBERS = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?![\w.])')


def content_words(text): return {w for w in WORDS.findall(text.lower()) if w not in STOP and len(w) > 2}


def main():
    STATE.mkdir(exist_ok=True)
    if (STATE / 'inventory-v1.json').exists(): raise ValueError('SYNTH pilot inventory already frozen')
    sources = json.loads((RAW / 'lock.json').read_text()); counts = Counter(); exercises = Counter()
    pools = defaultdict(list); licenses = Counter(); seen_questions = set()
    columns = ['synth_id', 'language', 'exercise', 'query', 'query_seed_url', 'query_seed_text',
               'additional_seed_url', 'seed_license', 'synthetic_answer']
    for source in sources:
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('pinned SYNTH pilot changed')
        offset = 0
        for batch in pq.ParquetFile(source['local']).iter_batches(columns=columns, batch_size=1024):
            for row in batch.to_pylist():
                index = offset; offset += 1; counts['raw_rows'] += 1
                if row['language'] != 'en': continue
                exercises[row['exercise']] += 1
                if row['exercise'].lower() not in ['memorization', 'rag']: continue
                counts['english_memory_or_rag_rows'] += 1
                seed, query, answer = row['query_seed_text'], row['query'], row['synthetic_answer']
                if not all(isinstance(v, str) and v.strip() for v in [seed, query, answer]): continue
                if row['additional_seed_url'] and row['additional_seed_url'] != row['query_seed_url']:
                    counts['held_unavailable_additional_source'] += 1; continue
                if len(seed) < len(answer): counts['held_seed_shorter_than_answer'] += 1; continue
                if set(NUMBERS.findall(answer)) - set(NUMBERS.findall(seed + '\n' + query)):
                    counts['held_unsupported_numeric_literal'] += 1; continue
                words = content_words(answer); evidence = content_words(seed + '\n' + query)
                coverage = len(words & evidence) / len(words) if words else 0
                for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    if coverage >= threshold: counts['answer_content_word_coverage_ge_' + str(threshold)] += 1
                if coverage < 0.8: continue
                qh = base.sha(base.norm(query))
                if qh in seen_questions: counts['repeated_normalized_query_in_candidate_slice'] += 1; continue
                seen_questions.add(qh); licenses[row['seed_license']] += 1
                cohort = row['exercise'].lower() + ('/coverage_ge_0.9' if coverage >= 0.9 else '/coverage_0.8_to_0.9')
                candidate = {'source': source, 'row_index': index, 'synth_id': row['synth_id'],
                    'query_hash': qh, 'content_word_coverage': coverage, 'cohort': cohort,
                    'seed_sha256': base.sha(seed), 'answer_sha256': base.sha(answer), 'seed_license': row['seed_license']}
                pool = pools[cohort]; pool.append((base.sha('synth-grounding-first-pilot:' + qh), candidate))
                if len(pool) > 16: pool.sort(key=lambda x: x[0]); del pool[8:]
        print(json.dumps({'source_file': source['path'], 'rows_seen': counts['raw_rows'],
                          'distinct_lexical_candidates': len(seen_questions)}), flush=True)
    selected = [item for key in sorted(pools) for _, item in sorted(pools[key], key=lambda x: x[0])[:8]]
    result = {'counts': dict(counts), 'english_exercises': dict(exercises), 'candidate_seed_licenses': dict(licenses),
        'distinct_lexical_candidate_queries': len(seen_questions), 'diagnostic_candidates': selected,
        'driver_sha256': base.file_sha(Path(__file__)), 'source_lock_sha256': base.file_sha(RAW / 'lock.json'),
        'accepted_training_episodes': 0, 'grounding_not_semantically_verified': True,
        'scope': 'Four actually downloaded hash-verified source files, not the full 68M source. Lexical overlap and seed length are feasibility filters only, not entailment or answer correctness.',
        'upstream_seed_url_content_not_independently_verified': True,
        'no_target_answer_used_as_retrieval_query': True, 'private_reasoning_not_imported': True,
        'source_programs_executed': False, 'weights_trained': False}
    base.save(STATE / 'inventory-v1.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'diagnostic_candidates'}), flush=True)


if __name__ == '__main__': main()
