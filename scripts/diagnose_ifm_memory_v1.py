"""First IFM coding/math/tool source review, before any answer admission."""
import argparse
from collections import Counter, defaultdict
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
from scripts.ifm_memory_adapter_v1 import adapt
from scripts.inventory_ifm_context_v1 import project, RAW

STATE = base.WORK / 'reports/memory-harness-v2-ifm-v1'
ADAPTER = base.REPO / 'scripts/ifm_memory_adapter_v1.py'


def prior_holds():
    result = set()
    for name in ['memory-harness-v2-openmath-v1', 'memory-harness-v2-synth-wikipedia-v1']:
        state = base.WORK / 'reports' / name
        audit = json.loads((state / 'semantic-audit-v1/summary.json').read_text())
        bad = set(audit['excluded_record_ids'])
        for path in (state / 'normalized').glob('pilot-*.jsonl'):
            for line in path.open():
                row = json.loads(line)
                if row['record_id'] in bad: result.add(row['query_hash'])
    known = json.loads((base.WORK / 'reports/memory-harness-v2-openmath-v1/known-incorrect-reference-holds-v1.json').read_text())
    result.update(r['query_hash'] for r in known['holds'])
    return result


def prepare():
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('IFM source diagnosis already frozen')
    ni.init(); base.init_tokenizer(); package.init()
    library = SkillLibrary(base.REPO); skills = {d: library.render(base.ROUTES[d]) for d in ['math', 'coding', 'tool_use']}
    root = base.WORK / 'reports/memory-harness-v2-opencode-v1'
    previous = sqlite3.connect(f'file:{root / "prior-query-aliases-v1.sqlite"}?mode=ro', uri=True)
    coding = sqlite3.connect(f'file:{root / "v26-selection-v1/candidates.sqlite"}?mode=ro', uri=True)
    # The immutable candidate table indexes complete inputs, not query_hash.
    # One exact read-only projection avoids a million-row scan per contender.
    coding_queries = {q for q, in coding.execute('SELECT query_hash FROM candidates')}
    holds = prior_holds(); pools = defaultdict(dict); counts = Counter()
    all_sources = json.loads((RAW / 'lock.json').read_text())
    sources = [s for s in all_sources if s['path'].split('/')[0] in {'agent', 'code', 'math'}]
    for n, source in enumerate(sources, 1):
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('IFM source changed')
        cohort = source['path'].split('/')[0] + '/' + Path(source['path']).name.split('-')[0]
        pool = pools[cohort]
        for index, raw in enumerate(next(pq.ParquetFile(source['local']).iter_batches(columns=['messages'], batch_size=512)).to_pylist()):
            counts['bounded_source_rows_seen'] += 1
            try: projected, question = project(raw)
            except (ValueError, KeyError, TypeError): counts['held:source_projection'] += 1; continue
            qh = base.sha(base.norm(question)); order = base.sha('ifm-first-context-source-diagnosis:' + qh)
            if qh in pool or (len(pool) >= 6 and order >= max(v[0] for v in pool.values())): continue
            if qh in holds or previous.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (qh,)).fetchone() or qh in coding_queries:
                counts['held:prior_query_or_first_hold_among_contenders'] += 1; continue
            if any(ni.benchmark_overlap(m.get('content', '')) for m in projected if m.get('role') == 'user' and isinstance(m.get('content'), str)):
                counts['held:local_evaluation_overlap_among_contenders'] += 1; continue
            try: row = adapt(raw, source, index, skills)
            except Rejected as exc: counts['held:' + str(exc)] += 1; continue
            pool[qh] = (order, row)
            if len(pool) > 6:
                worst = max(pool, key=lambda q: pool[q][0]); del pool[worst]
        if n % 10 == 0 or n == len(sources): print(json.dumps({'source_files': n, 'total': len(sources), 'samples': sum(map(len, pools.values()))}), flush=True)
    selected = []; seen = set(); budgets = []; preflight_holds = []
    for cohort in sorted(pools):
        for _, row in sorted(pools[cohort].values(), key=lambda x: x[0]):
            if row['query_hash'] in seen: counts['cross_effort_duplicate_diagnostic_query'] += 1; continue
            seen.add(row['query_hash'])
            try:
                record, _, _ = base.compile_row(row); budget = package.request_budget(record)
                payload = {'messages': row['messages'], 'tools': row['tools'], 'observation_origin': row['observation_origin']}
                audit_messages = [{'role': 'system', 'content': reviewer.PROMPT},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
                text = package.CHAT.render(messages=audit_messages, tools=[], add_generation_prompt=True,
                    enable_thinking=row['domain'] == 'math', bos_token='', eos_token='<|im_end|>')
                tokens = len(package.QWEN.encode(text, add_special_tokens=False).ids)
                reserve = 2048 if row['domain'] == 'math' else 900
                if tokens + reserve > 1000000: raise Rejected('served_audit_budget')
                budget.update(audit_input_tokens=tokens, audit_output_reserve=reserve, audit_render_sha256=base.sha(text),
                              pangolin_tokens=record['pangolin_tokens'], minicpm_tokens=record['minicpm_tokens'])
            except Rejected as exc: preflight_holds.append({'query_hash': row['query_hash'], 'reason': str(exc)}); continue
            selected.append(row); budgets.append(budget)
    previous.close(); coding.close()
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-ifm.jsonl').open('w') as handle:
        for row in selected: handle.write(canonical(row) + '\n')
    base.save(STATE / 'build-source-lock.json', {'files': sources})
    base.save(STATE / 'pilot-budget-proof.json', budgets)
    original = json.loads((base.WORK / 'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
    plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
    plan.update(jobs=[{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': r['domain'],
                      'messages': r['messages'], 'tools': r['tools'], 'observation_origin': r['observation_origin']} for r in selected],
        first_non_accept_is_final_for_that_record=True,
        scope='Up to six distinct source queries per category/effort from the first 512 rows of each actually downloaded agent/code/math source file, after explicit registry, credential, source text and static syntax checks. Repeated cross-effort diagnostic queries excluded. Not representative accuracy, PPL winner approval or whole-corpus semantic verification.')
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', plan)
    policy = {'counts': dict(counts), 'diagnostic_samples': len(selected),
        'cohorts': dict(Counter(r['source_refs']['source_category'] + '/' + r['source_refs']['source_effort'] for r in selected)),
        'preflight_holds': preflight_holds, 'prior_nonaccept_query_holds': sorted(holds),
        'driver_sha256': base.file_sha(Path(__file__)), 'adapter_sha256': base.file_sha(ADAPTER),
        'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'auditor_sha256': base.file_sha(Path(reviewer.__file__)),
        'source_inventory_sha256': base.file_sha(STATE / 'inventory-v1/summary.json'),
        'opencode_prior_candidates_sha256': base.file_sha(root / 'v26-selection-v1/candidates.sqlite'),
        'opencode_prior_query_membership_size': len(coding_queries),
        'accepted_training_rows': 0, 'same_question_alternatives_require_v26_ppl': True,
        'all_selected_native_and_review_budgets_measured_without_truncation': True,
        'source_programs_executed': False, 'weights_trained': False}
    base.save(STATE / 'pilot-summary.json', policy)
    print(json.dumps(policy), flush=True)


def audit():
    policy = json.loads((STATE / 'pilot-summary.json').read_text())
    for key, path in [('driver_sha256', Path(__file__)), ('adapter_sha256', ADAPTER),
                      ('plan_sha256', STATE / 'semantic-audit-v1/plan.json'), ('auditor_sha256', Path(reviewer.__file__))]:
        if base.file_sha(path) != policy[key]: raise ValueError('frozen source audit binding changed')
    reviewer.STATE = STATE; reviewer.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'audit'])
    args = parser.parse_args(); globals()[args.phase]()
