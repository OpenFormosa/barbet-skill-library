"""Full IFM math candidates, with every eligible effort version kept for v26.

This stage admits no training records and never executes source code. Source
contracts and first dispositions are immutable; normalized-query aliases are
held when the literal initial question differs. No answer is ranked here.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from barbet_inference.source_contract_holds import held_tool_schemas
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts.ifm_memory_adapter_v1 import adapt
from scripts.inventory_ifm_context_v1 import project

SOURCE = base.WORK / 'reports/memory-harness-v2-ifm-v1'
STATE = SOURCE / 'math-full-v1'
PREVIOUS = base.WORK / 'reports/memory-harness-v2-opencode-v1'
CONTRACT = base.REPO / 'configs/source_tool_contract_holds_v1.json'
SCALE = base.REPO / 'configs/memory_sft_v2_scale_10m.json'


def nonaccept_queries(audit, rows):
    bad = set(audit['excluded_record_ids'])
    found = {r['record_id']: r['query_hash'] for r in rows if r['record_id'] in bad}
    if set(found) != bad: raise ValueError('first nonaccept coverage is incomplete')
    return set(found.values())


def ambiguous_literal_queries(rows):
    first = {}; ambiguous = set()
    for row in rows:
        query, literal = row['query_hash'], row['literal_first_query_hash']
        if query in first and first[query] != literal: ambiguous.add(query)
        else: first[query] = literal
    return ambiguous


def adapt_math(raw, source, index, skills, contract):
    row = adapt(raw, source, index, skills)
    if held_tool_schemas(row['tools'], contract): raise Rejected('unresolved_source_tool_contract')
    messages = row['messages']
    if (row['domain'] != 'math' or row['tools'] or messages[-1]['role'] != 'assistant'
            or not messages[-1].get('loss', True)
            or any(m['role'] not in {'system', 'user'} for m in messages[:-1])
            or sum(m['role'] == 'user' for m in messages) != 1):
        raise Rejected('outside_single_question_math_scope')
    return row


def processed_bytes():
    paths = [base.WORK / 'exports', *sorted((base.WORK / 'reports').glob('memory-harness-v2*'))]
    # One invocation counts shared hardlinks once, including older evidence.
    out = subprocess.check_output(['du', '-s', '-c', '-B1', *map(str, paths)], text=True)
    return int(out.splitlines()[-1].split()[0])


def prepare():
    STATE.mkdir(exist_ok=True); (STATE / 'normalized').mkdir(exist_ok=True)
    path = STATE / 'normalization-policy.json'
    if path.exists():
        policy = json.loads(path.read_text())
        if policy['driver_sha256'] != base.file_sha(Path(__file__)): raise ValueError('normalizer changed')
        return policy
    resources = json.loads(SCALE.read_text())['resources']
    used = processed_bytes()
    # Reserve a bounded stage in advance; later compilation has a separate gate.
    stage_bound = 10 * 1024 ** 3
    if used + stage_bound > resources['new_processed_cap_bytes']:
        raise ValueError('insufficient processed budget for full math candidate stage')
    if shutil.disk_usage(base.WORK).free < resources['minimum_free_disk_bytes'] + stage_bound:
        raise ValueError('insufficient free disk above reserved floor')
    pilot = json.loads((SOURCE / 'pilot-summary.json').read_text())
    if pilot['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/ifm_memory_adapter_v1.py'):
        raise ValueError('first-audited adapter changed')
    holds = set(); evidence = []
    for name in ['memory-harness-v2-ifm-v1', 'memory-harness-v2-openmath-v1',
                 'memory-harness-v2-synth-wikipedia-v1', 'memory-harness-v2-synth-neighbors-v2']:
        root = base.WORK / 'reports' / name; audit_path = root / 'semantic-audit-v1/summary.json'
        audit = json.loads(audit_path.read_text())
        rows = [json.loads(line) for p in (root / 'normalized').glob('pilot-*.jsonl') for line in p.open()]
        holds.update(nonaccept_queries(audit, rows))
        evidence.append({'path': str(audit_path), 'sha256': base.file_sha(audit_path), 'counts': audit['counts']})
    source_audit = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    if len(source_audit['results']) != pilot['diagnostic_samples']: raise ValueError('source first audit incomplete')
    expected_controls = {'control-wrong-number': 'REJECT', 'control-unsupported-success': 'REJECT',
                         'control-positive-number': 'ACCEPT', 'control-positive-evidence': 'ACCEPT'}
    if {r['id']: r['verdict'] for r in source_audit['controls']} != expected_controls:
        raise ValueError('source diagnosis controls failed')
    known = base.WORK / 'reports/memory-harness-v2-openmath-v1/known-incorrect-reference-holds-v1.json'
    holds.update(r['query_hash'] for r in json.loads(known.read_text())['holds'])
    public_holds = base.WORK / 'reports/memory-harness-v2-public-source-contracts-v1/query-wide-holds-v1.json'
    holds.update(json.loads(public_holds.read_text())['query_hashes'])
    prior = STATE / 'prior-query-aliases.sqlite'
    if prior.exists(): raise ValueError('partial prior-query freeze; inspect instead of overwriting')
    shutil.copyfile(PREVIOUS / 'prior-query-aliases-v1.sqlite', prior)
    db = sqlite3.connect(prior)
    db.execute('ATTACH DATABASE ? AS coding', (str(PREVIOUS / 'v26-selection-v1/candidates.sqlite'),))
    db.execute('INSERT OR IGNORE INTO prior_queries SELECT query_hash FROM coding.candidates'); db.commit()
    prior_count = db.execute('SELECT COUNT(*) FROM prior_queries').fetchone()[0]; db.close()
    inventory = json.loads((SOURCE / 'inventory-v1/summary.json').read_text())
    parts = [part for part in inventory['parts'] if part['category'] == 'math']
    metadata = []
    for part in parts:
        if base.file_sha(part['part']) != part['sha256']: raise ValueError('source inventory changed')
        metadata.extend(pq.read_table(part['part'], columns=['query_hash', 'literal_first_query_hash']).to_pylist())
    aliases = ambiguous_literal_queries(metadata)
    jobs = [{'job_id': f'{i:03d}', 'source': part['source'], 'raw_rows': part['counts']['raw_rows']}
            for i, part in enumerate(parts)]
    library = SkillLibrary(base.REPO)
    skills = {d: library.render(base.ROUTES[d]) for d in ['math', 'coding', 'tool_use']}
    policy = {'version': 'ifm-math-context-full-v1', 'driver_sha256': base.file_sha(Path(__file__)),
        'adapter_sha256': pilot['adapter_sha256'], 'source_inventory_sha256': base.file_sha(SOURCE / 'inventory-v1/summary.json'),
        'first_audits': evidence, 'first_nonaccept_query_hashes': sorted(holds),
        'known_incorrect_math_holds_sha256': base.file_sha(known), 'public_source_contract_holds_sha256': base.file_sha(public_holds),
        'source_contract_policy_sha256': base.file_sha(CONTRACT), 'benchmark_exclusions_sha256': base.file_sha(ni.STATE / 'exclusions.private.json'),
        'unequal_literal_query_aliases': sorted(aliases), 'prior_query_database_sha256': base.file_sha(prior),
        'prior_query_identities': prior_count, 'rendered_skills_sha256': base.sha(canonical(skills)),
        'all_eligible_effort_versions_retained_until_native_v26': True,
        'source_scope': 'All seven downloaded math files and all effort variants; one user and one supervised final assistant, without tools or missing figures.',
        'quality_scope': 'First diagnostics and structural/source checks, not whole-corpus semantic verification.',
        'memory_is_original_question_not_teacher_reasoning': True, 'raw_source_rows': sum(j['raw_rows'] for j in jobs),
        'new_candidate_stage_byte_bound': stage_bound, 'processed_bytes_before_stage': used,
        'resource_config_sha256': base.file_sha(SCALE), 'accepted_training_rows': 0,
        'source_programs_executed': False, 'weights_trained': False}
    base.save(STATE / 'normalization-jobs.json', jobs); base.save(path, policy)
    print(json.dumps({'files': len(jobs), 'raw_rows': policy['raw_source_rows'], 'first_query_holds': len(holds),
                      'literal_alias_holds': len(aliases), 'prior_queries': prior_count}), flush=True)
    return policy


def init():
    global POLICY, HOLDS, ALIASES, PRIOR, SKILLS, TOOL_POLICY, POLICY_SHA
    POLICY = json.loads((STATE / 'normalization-policy.json').read_text())
    if POLICY['driver_sha256'] != base.file_sha(Path(__file__)) or POLICY['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/ifm_memory_adapter_v1.py'):
        raise ValueError('frozen source normalization changed')
    if POLICY['source_contract_policy_sha256'] != base.file_sha(CONTRACT): raise ValueError('source contract hold policy changed')
    if POLICY['benchmark_exclusions_sha256'] != base.file_sha(ni.STATE / 'exclusions.private.json'): raise ValueError('benchmark exclusions changed')
    ni.init(); HOLDS = set(POLICY['first_nonaccept_query_hashes']); ALIASES = set(POLICY['unequal_literal_query_aliases'])
    PRIOR = sqlite3.connect(f'file:{STATE / "prior-query-aliases.sqlite"}?mode=ro', uri=True)
    library = SkillLibrary(base.REPO); SKILLS = {d: library.render(base.ROUTES[d]) for d in ['math', 'coding', 'tool_use']}
    if base.sha(canonical(SKILLS)) != POLICY['rendered_skills_sha256']: raise ValueError('rendered skills changed')
    TOOL_POLICY = json.loads(CONTRACT.read_text()); POLICY_SHA = base.file_sha(STATE / 'normalization-policy.json')


def job(spec):
    path = STATE / 'normalized' / ('full-' + spec['job_id'] + '.jsonl'); receipt = path.with_suffix('.report.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    if path.exists(): raise ValueError('partial candidate shard; inspect without overwriting')
    source = spec['source']
    if base.file_sha(source['local']) != source['sha256']: raise ValueError('source bytes changed')
    counts = Counter(); excluded = []; offset = 0; written = 0
    # Per-shard allocations sum to the same bound and cannot race past it.
    jobs = json.loads((STATE / 'normalization-jobs.json').read_text())
    shard_bound = POLICY['new_candidate_stage_byte_bound'] // len(jobs)
    with path.open('wb') as handle:
        for batch in pq.ParquetFile(source['local']).iter_batches(columns=['messages'], batch_size=1024):
            for raw in batch.to_pylist():
                index = offset; offset += 1; query = None; counts['raw_rows_seen'] += 1
                try:
                    try: projected, question = project(raw)
                    except (KeyError, TypeError, ValueError) as exc: raise Rejected('invalid_visible_source_projection') from exc
                    query = base.sha(base.norm(question))
                    if query in HOLDS: raise Rejected('first_nonaccept_or_known_bad_query')
                    if query in ALIASES: raise Rejected('same_normalized_query_different_literal_input')
                    if PRIOR.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (query,)).fetchone():
                        raise Rejected('prior_source_or_wrapper_query')
                    if any(ni.benchmark_overlap(m.get('content', '')) for m in projected if m.get('role') == 'user' and isinstance(m.get('content'), str)):
                        raise Rejected('local_evaluation_overlap')
                    row = adapt_math(raw, source, index, SKILLS, TOOL_POLICY)
                    row['source_refs']['full_normalization_policy_sha256'] = POLICY_SHA
                except Rejected as exc:
                    reason = str(exc); counts['held:' + reason] += 1
                    excluded.append({'row_index': index, 'query_hash': query, 'reason': reason}); continue
                data = (canonical(row) + '\n').encode()
                if written + len(data) > shard_bound: raise ValueError('candidate shard byte bound reached; no truncation or publication')
                handle.write(data); written += len(data); counts['candidate_answers'] += 1
    if offset != spec['raw_rows']: raise ValueError('source row coverage mismatch')
    excluded_path = path.with_suffix('.excluded.parquet')
    schema = pa.schema([('row_index', pa.int64()), ('query_hash', pa.string()), ('reason', pa.string())])
    pq.write_table(pa.Table.from_pylist(excluded, schema=schema), excluded_path, compression='zstd')
    result = {'path': str(path), 'sha256': base.file_sha(path), 'counts': dict(counts),
        'source_sha256': source['sha256'], 'policy_sha256': POLICY_SHA, 'bytes': written,
        'excluded_path': str(excluded_path), 'excluded_sha256': base.file_sha(excluded_path)}
    base.save(receipt, result); return result


def main(workers):
    policy = prepare()
    if base.file_sha(STATE / 'prior-query-aliases.sqlite') != policy['prior_query_database_sha256']:
        raise ValueError('prior-query membership changed')
    if (STATE / 'normalize-summary.json').exists(): raise ValueError('normalization already complete')
    jobs = json.loads((STATE / 'normalization-jobs.json').read_text()); counts = Counter(); total_bytes = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(job, jobs), 1):
            counts.update(result['counts']); total_bytes += result['bytes']
            print(json.dumps({'files': n, 'total': len(jobs), 'counts': dict(counts), 'candidate_bytes': total_bytes}), flush=True)
    if counts['raw_rows_seen'] != policy['raw_source_rows']: raise ValueError('full math source coverage mismatch')
    base.save(STATE / 'normalize-summary.json', {'counts': dict(counts), 'candidate_bytes': total_bytes,
        'full_normalization_policy_sha256': base.file_sha(STATE / 'normalization-policy.json'),
        'accepted_training_rows': 0, 'needs_v26_ranking_and_release_gates': True,
        'source_programs_executed': False, 'weights_trained': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4: raise ValueError('normalization CPU worker bound')
    main(args.workers)
