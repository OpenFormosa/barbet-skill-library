"""Stage complete eligible OpenCode answers with preserved first holds.

AST parsing is syntax inspection only. No downloaded code, tests or assertions
are executed, and upstream test metadata is never supplied as prompt evidence.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import shutil
import sqlite3

import pyarrow.parquet as pq
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts.inventory_opencode_v1 import STATE, RAW
from scripts.opencode_memory_adapter_v1 import adapt

PYTHON_FENCE = re.compile(r'^```(?:python|py)\s*\n(.*?)^```\s*$', re.MULTILINE | re.DOTALL)


def syntax_gate(answer):
    if not isinstance(answer, str) or len(answer.encode()) > 2 * 1024 * 1024:
        raise Rejected('response_outside_bounded_static_python_scope')
    blocks = PYTHON_FENCE.findall(answer)
    if not blocks:
        raise Rejected('no_complete_explicit_python_block_for_static_check')
    for code in blocks:
        try:
            ast.parse(code, mode='exec')
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            raise Rejected('python_block_static_syntax_failure') from exc
    return len(blocks)


def prepare():
    path = STATE / 'full-normalization-policy-v1.json'
    if path.exists():
        policy = json.loads(path.read_text())
        if policy['driver_sha256'] != base.file_sha(Path(__file__)):
            raise ValueError('frozen OpenCode normalizer changed')
        return policy
    audit = json.loads((STATE / 'semantic-audit-v1/summary.json').read_text())
    pilot = json.loads((STATE / 'pilot-summary.json').read_text())
    if pilot['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/opencode_memory_adapter_v1.py'):
        raise ValueError('frozen source adapter changed')
    bad = set(audit['excluded_record_ids']); held = set()
    for line in (STATE / 'normalized/pilot-opencode.jsonl').open():
        row = json.loads(line)
        if row['record_id'] in bad: held.add(row['query_hash'])
    aliases = json.loads((STATE / 'candidate-input-alias-holds-v1.json').read_text())['query_hashes']
    prior = STATE / 'prior-query-aliases-v1.sqlite'
    if prior.exists(): raise ValueError('partial prior-query construction; inspect before resume')
    shutil.copyfile(ni.STATE / 'prior-query-aliases.sqlite', prior)
    db = sqlite3.connect(prior)
    db.execute('ATTACH DATABASE ? AS ni', (str(ni.STATE / 'v26-selection-v1/candidates.sqlite'),))
    db.execute('INSERT OR IGNORE INTO prior_queries SELECT query_hash FROM ni.candidates'); db.commit()
    # NI definition wrappers are not independent questions. Freeze the bare
    # target inputs as conservative aliases, including previous source holds.
    for job in json.loads((ni.STATE / 'full-normalize-jobs-v2.json').read_text()):
        source = job['source']
        if base.file_sha(source['local']) != source['sha256']: raise ValueError('NI source changed')
        task = json.loads(Path(source['local']).read_text())
        db.executemany('INSERT OR IGNORE INTO prior_queries VALUES (?)',
                       [(base.sha(base.norm(r['input'])),) for r in task['Instances']]); db.commit()
    prior_count = db.execute('SELECT COUNT(*) FROM prior_queries').fetchone()[0]; db.close()
    inventory = json.loads((STATE / 'inventory-v1/summary.json').read_text())
    jobs = []
    for i, part in enumerate(inventory['parts']):
        jobs.append({'job_id': f'{i:03d}', 'source': part['source'],
                     'inventory_part': part['part'], 'inventory_sha256': part['sha256'],
                     'raw_rows': part['counts']['raw_rows']})
    policy = {'version': 'opencode-contextual-memory-full-v1',
        'driver_sha256': base.file_sha(Path(__file__)), 'adapter_sha256': pilot['adapter_sha256'],
        'first_source_audit_sha256': base.file_sha(STATE / 'semantic-audit-v1/summary.json'),
        'first_source_audit_counts': audit['counts'], 'held_query_hashes': sorted(held),
        'unequal_condition_alias_query_hashes': aliases,
        'prior_query_database_sha256': base.file_sha(prior), 'prior_query_identities': prior_count,
        'static_gate': 'Every explicit python/py fenced block must parse as Python syntax; require at least one. Parsing never executes source programs or establishes semantic correctness.',
        'source_metadata_gate': 'Recorded source tests all pass with consistent test count and score 1.0; all three recorded judge scores at least 4. Not locally rerun or independent correctness evidence.',
        'same_query_all_answer_versions_retained_until_v26_ranking': True,
        'all_first_negative_queries_held_without_replay': True, 'source_code_executed': False,
        'candidate_only': True, 'accepted_training_rows': 0}
    base.save(STATE / 'full-normalize-jobs-v1.json', jobs); base.save(path, policy)
    print(json.dumps({'files': len(jobs), 'first_query_holds': len(held), 'alias_queries_held': len(aliases), 'prior_queries': prior_count}), flush=True)
    return policy


def init():
    global POLICY, HOLDS, ALIASES, PREVIOUS, SKILLS, POLICY_SHA
    POLICY = json.loads((STATE / 'full-normalization-policy-v1.json').read_text())
    if POLICY['driver_sha256'] != base.file_sha(Path(__file__)):
        raise ValueError('normalization driver changed')
    ni.init()
    HOLDS = set(POLICY['held_query_hashes']); ALIASES = set(POLICY['unequal_condition_alias_query_hashes'])
    PREVIOUS = sqlite3.connect(f'file:{STATE / "prior-query-aliases-v1.sqlite"}?mode=ro', uri=True)
    SKILLS = SkillLibrary(base.REPO).render(base.ROUTES['coding'])
    POLICY_SHA = base.file_sha(STATE / 'full-normalization-policy-v1.json')


def job(spec):
    name = 'full-' + spec['job_id']; path = STATE / 'normalized' / (name + '.jsonl')
    receipt = path.with_suffix('.report.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    if path.exists(): raise ValueError('partial candidate shard; inspect without overwriting')
    if shutil.disk_usage(base.WORK).free < 100 * 1024 ** 3:
        raise ValueError('free disk below the authorized reserve')
    source = spec['source']
    if base.file_sha(source['local']) != source['sha256'] or base.file_sha(spec['inventory_part']) != spec['inventory_sha256']:
        raise ValueError('source or inventory bytes changed')
    inventory = pq.read_table(spec['inventory_part']).to_pylist(); counts = Counter(); offset = 0
    with path.open('w') as output, path.with_name(name + '.excluded.jsonl').open('w') as excluded:
        for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                idx = offset; offset += 1; counts['raw_rows_seen'] += 1; entry = inventory[idx]
                if entry['row_index'] != idx or base.sha(base.norm(raw['input'])) != entry['query_hash']:
                    raise ValueError('source row and inventory differ')
                query = entry['query_hash']
                try:
                    if not entry['eligible']: raise Rejected(entry['hold_reason'])
                    if query in HOLDS: raise Rejected('first_source_diagnostic_nonaccept_query')
                    if query in ALIASES: raise Rejected('unequal_literal_input_alias')
                    if PREVIOUS.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (query,)).fetchone():
                        raise Rejected('prior_source_or_wrapper_query')
                    if ni.benchmark_overlap(raw['input']): raise Rejected('local_evaluation_overlap')
                    blocks = syntax_gate(raw['output']); row = adapt(raw, source, idx, SKILLS)
                    row['source_refs']['full_normalization_policy_sha256'] = POLICY_SHA
                    row['source_refs']['python_static_syntax'] = {'blocks': blocks, 'parsed': True, 'executed': False}
                except Rejected as exc:
                    reason = str(exc); counts['held:' + reason] += 1
                    excluded.write(canonical({'row_index': idx, 'query_hash': query, 'reason': reason}) + '\n')
                    continue
                output.write(canonical(row) + '\n'); counts['candidate_answers'] += 1
    if offset != spec['raw_rows']: raise ValueError('source row coverage mismatch')
    result = {'path': str(path), 'sha256': base.file_sha(path), 'counts': dict(counts),
              'policy_sha256': POLICY_SHA, 'source_sha256': source['sha256']}
    base.save(receipt, result); return result


def main(workers):
    policy = prepare()
    if policy['prior_query_database_sha256'] != base.file_sha(STATE / 'prior-query-aliases-v1.sqlite'):
        raise ValueError('prior query exclusion changed')
    if (STATE / 'normalize-summary.json').exists(): raise ValueError('normalization already complete')
    jobs = json.loads((STATE / 'full-normalize-jobs-v1.json').read_text()); counts = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(job, jobs), 1):
            counts.update(result['counts'])
            print(json.dumps({'files': n, 'total': len(jobs), 'counts': dict(counts)}), flush=True)
    base.save(STATE / 'normalize-summary.json', {'counts': dict(counts), 'accepted_training_rows': 0,
        'full_normalization_policy_sha256': base.file_sha(STATE / 'full-normalization-policy-v1.json'),
        'needs_v26_ranking_and_release_gates': True, 'source_programs_executed': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=4); args = p.parse_args()
    if not 1 <= args.workers <= 4: raise ValueError('OpenCode normalization CPU worker bound')
    main(args.workers)
