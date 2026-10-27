"""All prospective IFM Python candidates, with bounded lossless storage.

The first-reviewed adapter stays unchanged. Every eligible answer version is
retained for later native v26 comparison. No source program or test is executed.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from barbet_inference import candidate_archive as archive
from barbet_inference.library import SkillLibrary
from barbet_inference.memory import canonical
from barbet_inference.sft import Rejected
from barbet_inference.source_contract_holds import held_tool_schemas
from scripts import build_memory_sft_v2 as base
from scripts import build_natural_instructions_memory_v1 as ni
from scripts.ifm_memory_adapter_v1 import adapt
from scripts.inventory_ifm_context_v1 import project
from scripts.normalize_ifm_math_memory_v1 import ambiguous_literal_queries, processed_bytes, nonaccept_queries

SOURCE = base.WORK / 'reports/memory-harness-v2-ifm-v1'
MATH = SOURCE / 'math-full-v1'
STATE = SOURCE / 'code-full-v1'
CONTRACT = base.REPO / 'configs/source_tool_contract_holds_v1.json'
RESOURCES = base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json'


def adapt_code(raw, source, index, skills, contract):
    row = adapt(raw, source, index, skills)
    if row['domain'] != 'coding':
        raise Rejected('outside_source_coding_scope')
    if held_tool_schemas(row['tools'], contract):
        raise Rejected('unresolved_source_tool_contract')
    return row


def prepare():
    STATE.mkdir(exist_ok=True); (STATE / 'normalized').mkdir(exist_ok=True)
    path = STATE / 'normalization-policy.json'
    if path.exists():
        policy = json.loads(path.read_text())
        if policy['driver_sha256'] != base.file_sha(Path(__file__)):
            raise ValueError('frozen code normalizer differs')
        return policy
    resources = json.loads(RESOURCES.read_text())['resources']
    used = processed_bytes(); free = shutil.disk_usage(base.WORK).free
    stage_bound = 20 * 1024 ** 3
    if used + stage_bound > resources['new_processed_cap_bytes'] or free < resources['minimum_free_disk_bytes'] + stage_bound:
        raise ValueError('insufficient prospective processed storage or free floor')
    probe = base.WORK / 'reports/memory-harness-v2-candidate-archive-v1'
    evidence = json.loads((probe / 'summary.json').read_text())
    if not evidence['all_comparisons_passed'] or evidence['samples'] != 86:
        raise ValueError('exact source and native-render storage equivalence required')
    if json.loads((probe / 'plan.json').read_text())['archive_reader_sha256'] != base.file_sha(Path(archive.__file__)):
        raise ValueError('checked archive implementation differs')
    pilot = json.loads((SOURCE / 'pilot-summary.json').read_text())
    if pilot['adapter_sha256'] != base.file_sha(base.REPO / 'scripts/ifm_memory_adapter_v1.py'):
        raise ValueError('first-reviewed source adapter changed')
    previous = json.loads((MATH / 'normalization-policy.json').read_text())
    holds = set(previous['first_nonaccept_query_hashes']); proof_files = []
    winner = MATH / 'selected-v1/semantic-audit-v1/summary.json'
    rows = [json.loads(line) for p in (MATH / 'selected-v1/normalized').glob('*.jsonl') for line in p.open()]
    holds.update(nonaccept_queries(json.loads(winner.read_text()), rows)); proof_files.append(winner)
    scope = MATH / 'selected-v1/source-scope-v2/query-holds.json'
    holds.update(json.loads(scope.read_text())['query_hashes']); proof_files.append(scope)
    independent = SOURCE / 'independent-source-counterexamples-v1.json'
    holds.update(r['query_hash'] for r in json.loads(independent.read_text())['cases']); proof_files.append(independent)
    inventory = json.loads((SOURCE / 'inventory-v1/summary.json').read_text())
    parts = [p for p in inventory['parts'] if p['category'] == 'code']; metadata = []
    for part in parts:
        if base.file_sha(part['part']) != part['sha256']:
            raise ValueError('source identity inventory changed')
        metadata.extend(pq.read_table(part['part'], columns=['query_hash', 'literal_first_query_hash']).to_pylist())
    aliases = ambiguous_literal_queries(metadata)
    # Fixed per-file allocations sum to the stage bound and cannot race past it.
    minimum_shard = 256 * 1024 ** 2
    # Keep one GiB inside the reservation for exclusion metadata and receipts.
    remainder = stage_bound - 1024 ** 3 - minimum_shard * len(parts)
    source_bytes = sum(p['source']['bytes'] for p in parts)
    jobs = [{'job_id': f'{i:03d}', 'source': p['source'], 'raw_rows': p['counts']['raw_rows'],
             'candidate_byte_bound': minimum_shard + remainder * p['source']['bytes'] // source_bytes}
            for i, p in enumerate(parts)]
    if sum(j['candidate_byte_bound'] for j in jobs) > stage_bound or remainder < 0:
        raise ValueError('invalid shard allocations')
    skills = {d: SkillLibrary(base.REPO).render(base.ROUTES[d]) for d in ['math', 'coding', 'tool_use']}
    prior = MATH / 'prior-query-aliases.sqlite'; math_candidates = MATH / 'v26-selection-v1/candidates.sqlite'
    if base.file_sha(prior) != previous['prior_query_database_sha256']:
        raise ValueError('previous query exclusion bytes differ')
    policy = {'version': 'ifm-code-full-framed-v1', 'driver_sha256': base.file_sha(Path(__file__)),
        'adapter_sha256': pilot['adapter_sha256'], 'archive_implementation_sha256': base.file_sha(Path(archive.__file__)),
        'archive_format': archive.FORMAT, 'archive_equivalence_proof_sha256': base.file_sha(probe / 'summary.json'),
        'source_inventory_sha256': base.file_sha(SOURCE / 'inventory-v1/summary.json'),
        'first_source_audit_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'previous_normalization_policy_sha256': base.file_sha(MATH / 'normalization-policy.json'),
        'additional_first_hold_proofs': [{'path': str(p), 'sha256': base.file_sha(p)} for p in proof_files],
        'first_nonaccept_query_hashes': sorted(holds), 'unequal_literal_query_aliases': sorted(aliases),
        'prior_query_database': str(prior), 'prior_query_database_sha256': base.file_sha(prior),
        'all_prior_math_candidate_database': str(math_candidates), 'all_prior_math_candidate_database_sha256': base.file_sha(math_candidates),
        'source_contract_policy_sha256': base.file_sha(CONTRACT),
        'benchmark_exclusions_sha256': base.file_sha(ni.STATE / 'exclusions.private.json'),
        'rendered_skills_sha256': base.sha(canonical(skills)),
        'raw_source_rows': sum(j['raw_rows'] for j in jobs), 'new_candidate_stage_byte_bound': stage_bound,
        'processed_bytes_before_stage': used, 'free_disk_bytes_before_stage': free,
        'resource_config_sha256': base.file_sha(RESOURCES), 'all_eligible_answer_versions_retained': True,
        'source_scope': 'All 38 downloaded code files and all effort variants. Require complete explicit Python fences that statically parse; preserve declared tools and recorded observations when present.',
        'quality_scope': 'Unchanged first-reviewed structural adapter and first query holds. No claim of whole-corpus correctness or local source-code execution.',
        'accepted_training_rows': 0, 'source_programs_executed': False, 'weights_trained': False}
    base.save(STATE / 'normalization-jobs.json', jobs); base.save(path, policy)
    print(json.dumps({'files': len(jobs), 'raw_rows': policy['raw_source_rows'], 'literal_alias_holds': len(aliases),
                      'first_query_holds': len(holds), 'reserved_stage_bytes': stage_bound}), flush=True)
    return policy


def init():
    global POLICY, HOLDS, ALIASES, PRIOR, PRIOR_MATH, SKILLS, TOOL_POLICY, POLICY_SHA
    POLICY = json.loads((STATE / 'normalization-policy.json').read_text())
    for key, path in [('driver_sha256', Path(__file__)), ('adapter_sha256', base.REPO / 'scripts/ifm_memory_adapter_v1.py'),
                      ('archive_implementation_sha256', Path(archive.__file__)), ('source_contract_policy_sha256', CONTRACT),
                      ('benchmark_exclusions_sha256', ni.STATE / 'exclusions.private.json'), ('resource_config_sha256', RESOURCES)]:
        if POLICY[key] != base.file_sha(path):
            raise ValueError('frozen code normalization binding differs: ' + key)
    ni.init(); HOLDS = set(POLICY['first_nonaccept_query_hashes']); ALIASES = set(POLICY['unequal_literal_query_aliases'])
    PRIOR = sqlite3.connect(f'file:{POLICY["prior_query_database"]}?mode=ro', uri=True)
    db = sqlite3.connect(f'file:{POLICY["all_prior_math_candidate_database"]}?mode=ro', uri=True)
    PRIOR_MATH = {r[0] for r in db.execute('SELECT query_hash FROM candidates')}; db.close()
    skills = SkillLibrary(base.REPO); SKILLS = {d: skills.render(base.ROUTES[d]) for d in ['math', 'coding', 'tool_use']}
    if base.sha(canonical(SKILLS)) != POLICY['rendered_skills_sha256']:
        raise ValueError('frozen skill content differs')
    TOOL_POLICY = json.loads(CONTRACT.read_text()); POLICY_SHA = base.file_sha(STATE / 'normalization-policy.json')


def job(spec):
    path = STATE / 'normalized' / ('full-' + spec['job_id'] + '.bca'); receipt = path.with_suffix('.report.json')
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if base.file_sha(path) != result['sha256'] or result['policy_sha256'] != POLICY_SHA:
            raise ValueError('completed candidate shard differs')
        return result
    if path.exists():
        raise ValueError('partial source archive exists; preserve and inspect')
    source = spec['source']
    if base.file_sha(source['local']) != source['sha256']:
        raise ValueError('pinned source bytes changed')
    counts = Counter(); excluded = []; offset = 0; raw_bytes = 0; raw_digest = hashlib.sha256()
    with archive.ArchiveWriter(path) as writer:
        for batch in pq.ParquetFile(source['local']).iter_batches(columns=['messages'], batch_size=512):
            for raw in batch.to_pylist():
                index = offset; offset += 1; query = None; counts['raw_rows_seen'] += 1
                try:
                    try:
                        projected, question = project(raw)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise Rejected('invalid_visible_source_projection') from exc
                    query = base.sha(base.norm(question))
                    if query in HOLDS:
                        raise Rejected('first_nonaccept_or_known_bad_query')
                    if query in ALIASES:
                        raise Rejected('same_normalized_query_different_literal_input')
                    if query in PRIOR_MATH or PRIOR.execute('SELECT 1 FROM prior_queries WHERE query_hash=?', (query,)).fetchone():
                        raise Rejected('prior_source_or_wrapper_query')
                    if any(ni.benchmark_overlap(m.get('content', '')) for m in projected if m['role'] == 'user' and isinstance(m.get('content'), str)):
                        raise Rejected('local_evaluation_overlap')
                    row = adapt_code(raw, source, index, SKILLS, TOOL_POLICY)
                    row['source_refs']['full_normalization_policy_sha256'] = POLICY_SHA
                    data = (canonical(row) + '\n').encode()
                    if len(data) > archive.MAX_RAW_BYTES:
                        raise Rejected('outside_bounded_complete_candidate_frame_scope')
                except Rejected as exc:
                    reason = str(exc); counts['held:' + reason] += 1
                    excluded.append({'row_index': index, 'query_hash': query, 'reason': reason}); continue
                if writer.handle.tell() + len(data) + 65536 + archive.HEADER.size > spec['candidate_byte_bound']:
                    raise ValueError('candidate shard storage reservation exhausted; never truncate')
                writer.append(data); raw_bytes += len(data); raw_digest.update(data); counts['candidate_answers'] += 1
    if offset != spec['raw_rows']:
        raise ValueError('complete coding source coverage mismatch')
    recovered = hashlib.sha256(); reread = 0
    for unused_offset, unused_length, data in archive.iter_records(path):
        recovered.update(data); reread += 1
    if recovered.hexdigest() != raw_digest.hexdigest() or reread != counts.get('candidate_answers', 0):
        raise ValueError('lossless archive full readback differs')
    excluded_path = path.with_suffix('.excluded.parquet')
    schema = pa.schema([('row_index', pa.int64()), ('query_hash', pa.string()), ('reason', pa.string())])
    pq.write_table(pa.Table.from_pylist(excluded, schema=schema), excluded_path, compression='zstd')
    result = {'path': str(path), 'format': archive.FORMAT, 'sha256': base.file_sha(path), 'counts': dict(counts),
        'source_sha256': source['sha256'], 'policy_sha256': POLICY_SHA, 'archive_bytes': path.stat().st_size,
        'raw_payload_bytes': raw_bytes, 'decoded_payload_sha256': raw_digest.hexdigest(), 'all_frames_exactly_reread': True,
        'excluded_path': str(excluded_path), 'excluded_sha256': base.file_sha(excluded_path)}
    base.save(receipt, result); return result


def main(workers):
    policy = prepare()
    for key, path_key in [('prior_query_database_sha256', 'prior_query_database'),
                          ('all_prior_math_candidate_database_sha256', 'all_prior_math_candidate_database')]:
        if base.file_sha(policy[path_key]) != policy[key]:
            raise ValueError('prior question exclusions changed')
    if (STATE / 'normalize-summary.json').exists():
        raise ValueError('coding normalization already complete')
    jobs = json.loads((STATE / 'normalization-jobs.json').read_text()); counts = Counter(); archived = raw = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for n, result in enumerate(pool.map(job, jobs), 1):
            counts.update(result['counts']); archived += result['archive_bytes']; raw += result['raw_payload_bytes']
            print(json.dumps({'files': n, 'total': len(jobs), 'counts': dict(counts), 'candidate_archive_bytes': archived}), flush=True)
    if counts['raw_rows_seen'] != policy['raw_source_rows']:
        raise ValueError('full coding source coverage differs')
    base.save(STATE / 'normalize-summary.json', {'counts': dict(counts), 'candidate_archive_bytes': archived,
        'raw_candidate_payload_bytes': raw, 'archive_format': archive.FORMAT, 'all_original_candidate_bytes_preserved': True,
        'full_normalization_policy_sha256': base.file_sha(STATE / 'normalization-policy.json'),
        'accepted_training_rows': 0, 'needs_v26_ranking_and_release_gates': True,
        'source_programs_executed': False, 'weights_trained': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('coding normalization worker bound')
    main(args.workers)
