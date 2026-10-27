"""First NI winner diagnosis and exact public packaging; no source-code execution."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq
from barbet_inference.memory import canonical
from scripts import audit_memory_sft_v2 as reviewer
from scripts import build_memory_sft_v2 as base
from scripts import package_memory_sft_v2 as package
from scripts.finalize_natural_instructions_v26_v1 import STATE, OUT, SOURCE
from scripts.prune_v26_query_aliases_v1 import STATE as PRIOR

PUBLIC = base.WORK / 'exports/memory-harness-v2-natural-instructions-public-v1'
WAVE = 'wave-0004'


def choose_samples(rows, reviewed, quota=8):
    pools = defaultdict(list)
    for row in rows:
        if row['record_id'] in reviewed:
            continue
        method = row['source_refs']['answer_selection']['method']
        pool = pools[(row['domain'], method)]
        pool.append((base.sha('ni-first-winner-diagnosis:' + row['record_id']), row))
        if len(pool) > 2 * quota:
            pool.sort(key=lambda item: item[0]); del pool[quota:]
    return [row for key in sorted(pools) for _, row in sorted(pools[key], key=lambda item: item[0])[:quota]]


def selected_rows():
    for path in sorted((STATE / 'selected').glob('*.jsonl')):
        for line in path.open():
            yield json.loads(line)


def prepare_audit():
    if not (STATE / 'release-scope.json').exists():
        raise ValueError('complete NI winner selection required')
    policy_path = STATE / 'winner-diagnosis-policy-v1.json'
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if policy['driver_sha256'] != base.file_sha(Path(__file__)):
            raise ValueError('frozen NI release driver changed')
        return
    original = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    reviewed = {r['record_id'] for r in original['results']}
    samples = choose_samples(selected_rows(), reviewed)
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-selected-winners.jsonl').open('w') as handle:
        for row in samples:
            handle.write(canonical(row) + '\n')
    controls = json.loads((SOURCE / 'semantic-audit-v1/plan.json').read_text())['controls']
    jobs = [{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': r['domain'],
             'messages': r['messages'], 'tools': r['tools'], 'observation_origin': r['observation_origin'],
             'selection_method': r['source_refs']['answer_selection']['method'],
             'categories': r['source_refs']['categories']} for r in samples]
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', {
        'schema_sha256': reviewer.digest(reviewer.SCHEMA), 'prompt_sha256': reviewer.digest(reviewer.PROMPT),
        'model': reviewer.MODEL, 'revision': reviewer.REV, 'controls': controls, 'jobs': jobs,
        'scope': 'Up to eight distinct winners per domain and selection method, deterministically sampled. Excludes previously reviewed source records. Diagnostic only, not corpus accuracy or all-row semantic approval.',
        'first_non_accept_is_final_for_that_record': True})
    base.save(policy_path, {'driver_sha256': base.file_sha(Path(__file__)),
        'first_source_summary_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'selected_summary_sha256': base.file_sha(STATE / 'dedup-summary.json'),
        'samples': len(samples), 'source_reviewed_records_not_replayed': len(reviewed),
        'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json')})
    print(json.dumps({'winner_diagnostic_samples': len(samples), 'source_reviews_not_replayed': len(reviewed)}), flush=True)


def package_init():
    package.STATE = STATE; package.OUT = OUT; package.PUBLIC = PUBLIC; package.WAVE = WAVE
    package.EXTRA_ROW_GATE = None; package.init()
    # Also retain source-level holds, even though full normalization already
    # excluded them. A later stage can never revive a source negative.
    source = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    policy = json.loads((SOURCE / 'full-normalization-policy-v2.json').read_text())
    package.HOLD_IDS.update(source['excluded_record_ids'])
    package.HOLD_QUERIES.update(policy['held_query_hashes'])


def check_cumulative_index(db):
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
              for key in ['record_id', 'query_hash', 'family_id']}
    if unique['record_id'] != unique['query_hash']:
        raise ValueError('NI contains multiple selected views of one normalized question')
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('NI cross-split identity')
    if db.execute('SELECT 1 FROM records r JOIN previous.records p USING(query_hash) LIMIT 1').fetchone():
        raise ValueError('new NI wave overlaps a published question')
    if db.execute('SELECT 1 FROM records r JOIN previous.records p USING(family_id) WHERE r.split!=p.split LIMIT 1').fetchone():
        raise ValueError('published family fold changed')
    return unique


def package_all(workers):
    prepare_audit()
    if (STATE / 'public-package-summary.json').exists():
        raise ValueError('NI public package already complete')
    audit = json.loads((STATE / 'semantic-audit-v1/summary.json').read_text())
    compiled = json.loads((STATE / 'compile-summary.json').read_text())
    paths = sorted((STATE / 'compiled').glob('[0-9]*.json'))
    if len(paths) != json.loads((STATE / 'dedup-summary.json').read_text())['shards']:
        raise ValueError('missing NI compiled shard')
    counts = Counter(); files = []
    with ProcessPoolExecutor(max_workers=workers, initializer=package_init) as pool:
        for n, part in enumerate(pool.map(package.job, paths), 1):
            counts.update(part['counts']); files.extend(part['files'])
            print(json.dumps({'packaged_shards': n, 'total': len(paths), 'accepted': counts['accepted']}), flush=True)
    if counts['seen'] != compiled['accepted']:
        raise ValueError('compiled rows missing from public gate')
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('CREATE TABLE records(record_id TEXT PRIMARY KEY,family_id TEXT,query_hash TEXT,split TEXT,domain TEXT,source TEXT,upstream_source TEXT)')
    for path in sorted((STATE / 'public-package').glob('*.index.parquet')):
        rows = pq.read_table(path).to_pylist()
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?)', [tuple(r[k] for k in
            ['record_id', 'family_id', 'query_hash', 'split', 'domain', 'source', 'upstream_source']) for r in rows])
    db.commit(); db.execute('ATTACH DATABASE ? AS previous', (str(PRIOR / 'public-index.sqlite'),))
    unique = check_cumulative_index(db)
    ids = {r[0] for r in db.execute('SELECT record_id FROM records')}; db.close()
    proofs = set()
    for item in files:
        if base.file_sha(PUBLIC / item['path']) != item['sha256']:
            raise ValueError('public shard hash changed')
        if item['config'] == 'budget-proof':
            values = pq.read_table(PUBLIC / item['path'], columns=['record_id']).column('record_id').to_pylist()
            if len(set(values)) != len(values) or proofs.intersection(values):
                raise ValueError('duplicate budget proof')
            proofs.update(values)
        if item['config'] == 'records':
            for row in pq.read_table(PUBLIC / item['path'], columns=['source_refs_json']).to_pylist():
                method = json.loads(row['source_refs_json'])['answer_selection']['method']
                counts['selection:' + method] += 1
    if ids != proofs or len(ids) != counts['accepted']:
        raise ValueError('accepted identity or proof coverage differs')
    base.save(STATE / 'public-package-summary.json', {
        'wave': WAVE, 'status': 'LOCAL_VERIFIED_NOT_YET_PUBLISHED', 'counts': dict(counts),
        'unique_counts': unique, 'files': files, 'ppl_binding_sha256': base.file_sha(STATE / 'ppl-score-binding.json'),
        'source_audit': json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())['counts'],
        'winner_audit': audit['counts'], 'winner_audit_samples': len(audit['results']),
        'all_records_semantically_verified': False, 'all_three_configs_exact_readback': True,
        'all_answer_free_qwen_chat_and_completion_budgets_measured': True,
        'all_native_budgets_include_4096_output_reserve': True, 'all_previous_folds_preserved': True,
        'no_published_question_overlap': True, 'first_nonaccept_query_holds_preserved': True,
        'all_budget_proof_identities_equal': True, 'legacy_active_rows': 0,
        'accepted_episodes_target': 10000000, 'full_goal_complete': False,
        'model_weights_trained': False, 'model_efficacy_measured': False})
    print(json.dumps({'accepted': counts['accepted'], 'unique_counts': unique, 'files': len(files)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare-audit', 'audit', 'package'])
    p.add_argument('--workers', type=int, default=16); args = p.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('NI public-package worker bound')
    if args.phase == 'prepare-audit': prepare_audit()
    elif args.phase == 'audit':
        prepare_audit(); reviewer.STATE = STATE; reviewer.main()
    else: package_all(args.workers)
