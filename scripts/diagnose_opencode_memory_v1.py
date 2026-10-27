"""First immutable OpenCode source diagnosis after recorded-test filtering."""
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
from scripts import build_natural_instructions_memory_v1 as earlier
from scripts.inventory_opencode_v1 import STATE, RAW
from scripts.opencode_memory_adapter_v1 import adapt


def prepare():
    if (STATE / 'pilot-summary.json').exists(): raise ValueError('source pilot already frozen')
    inventory = json.loads((STATE / 'inventory-v1/summary.json').read_text())
    policy = json.loads((STATE / 'inventory-v1/policy.json').read_text())
    if policy['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/opencode_memory_adapter_v1.py'):
        raise ValueError('source adapter changed after inventory')
    db = sqlite3.connect(f'file:{STATE / "inventory-v1/questions.sqlite"}?mode=ro', uri=True)
    aliases = {q for q, in db.execute('SELECT query_hash FROM eligible_answers GROUP BY query_hash HAVING COUNT(DISTINCT literal_question_hash)>1')}; db.close()
    earlier.init(); skills = SkillLibrary(base.REPO).render(base.ROUTES['coding'])
    pools = defaultdict(dict); counts = Counter(); files = json.loads((RAW / 'lock.json').read_text())
    for source in files:
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('raw source changed')
        for index, raw in enumerate(next(pq.ParquetFile(source['local']).iter_batches(batch_size=512)).to_pylist()):
            counts['pilot_source_rows_seen'] += 1; qh = base.sha(base.norm(raw['input']))
            if qh in aliases: counts['held:unequal_condition_alias'] += 1; continue
            if earlier.PREVIOUS.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (qh,)).fetchone():
                counts['held:prior_query'] += 1; continue
            if earlier.benchmark_overlap(raw['input']): counts['held:local_evaluation_overlap'] += 1; continue
            try: row = adapt(raw, source, index, skills)
            except Rejected as exc: counts['held:' + str(exc)] += 1; continue
            cohort = raw['domain'] + '/' + raw['generation_algorithm']; pool = pools[cohort]
            key = base.sha('opencode-first-source-diagnosis:' + qh)
            if qh not in pool or row['record_id'] < pool[qh][1]['record_id']: pool[qh] = (key, row)
            if len(pool) > 48:
                keep = sorted(pool, key=lambda q: pool[q][0])[:12]; pools[cohort] = {q: pool[q] for q in keep}
    selected = [r for pool in pools.values() for _, r in sorted(pool.values(), key=lambda x: x[0])[:12]]
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-opencode.jsonl').open('w') as handle:
        for row in sorted(selected, key=lambda r: r['record_id']): handle.write(canonical(row) + '\n')
    base.save(STATE / 'candidate-input-alias-holds-v1.json', {'query_hashes': sorted(aliases),
        'rule': 'Different literal inputs that share a normalized single-turn query remain held until a separately validated common-input policy exists.'})
    base.save(STATE / 'build-source-lock.json', {'files': files})
    base.save(STATE / 'pilot-summary.json', {'diagnostic_samples': len(selected), 'counts': dict(counts),
        'cohorts': dict(Counter(r['source_refs']['upstream_domain'] + '/' + r['source_refs']['generation_algorithm'] for r in selected)),
        'source_gate_distinct_questions_before_further_holds': inventory['questions_with_a_source_gate_eligible_answer'],
        'unequal_literal_condition_query_holds': len(aliases), 'adapter_sha256': policy['adapter_sha256'],
        'driver_sha256': base.file_sha(Path(__file__)), 'accepted_training_rows': 0, 'weights_trained': False})
    print(json.dumps({'samples': len(selected), 'pilot_counts': dict(counts), 'alias_queries_held': len(aliases)}), flush=True)


def audit():
    first = json.loads((STATE / 'pilot-summary.json').read_text())
    if first['driver_sha256'] != base.file_sha(Path(__file__)): raise ValueError('source diagnosis driver changed')
    reviewer.STATE = STATE; path = STATE / 'semantic-audit-v1/plan.json'
    if not path.exists():
        original = json.loads((base.WORK / 'reports/memory-harness-v2/semantic-audit-v1/plan.json').read_text())
        rows = [json.loads(line) for line in (STATE / 'normalized/pilot-opencode.jsonl').open()]
        plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
        plan.update(jobs=[{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'coding',
                          'messages': r['messages'], 'tools': [], 'observation_origin': r['observation_origin']} for r in rows],
            first_non_accept_is_final_for_that_record=True,
            scope='Up to 12 distinct source questions per domain/generation cohort, selected deterministically from the first 512 rows of every source file after recorded-test/judge gates. Not representative corpus accuracy or a review of later PPL winners.')
        reviewer.dump(path, plan)
    reviewer.main()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'audit']); args = p.parse_args()
    prepare() if args.phase == 'prepare' else audit()
