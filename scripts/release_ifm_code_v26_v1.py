"""Export full IFM coding winners after complete v26 ranking and first diagnosis."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pyarrow.parquet as pq

from barbet_inference.memory import canonical
from scripts import build_memory_sft_v2 as base
from scripts import finalize_v26_answer_selection as selector
from scripts import audit_memory_sft_v2 as reviewer
from scripts import audit_memory_sft_math_v2 as audit_transport
from scripts import package_memory_sft_v2 as package
from scripts import release_natural_instructions_v26_v1 as shared
from scripts import normalize_ifm_code_memory_v1 as normalizer
from scripts.rank_ifm_code_v26_v1 import STATE as SCORE, FROZEN
from scripts.publish_ifm_math_v26_v1 import STATE as PRIOR
from barbet_inference import candidate_archive as archive
from scripts import rank_ifm_code_v26_v1 as ranking
import shutil

SOURCE = normalizer.SOURCE
STATE = normalizer.STATE / 'selected-v1'
OUT = base.WORK / 'exports/memory-harness-v2-ifm-code-selected-v1'
PUBLIC = base.WORK / 'exports/memory-harness-v2-ifm-code-public-v1'
WAVE = 'wave-0008'


def stage_resources():
    resources = json.loads(normalizer.RESOURCES.read_text())['resources']; reserve = 12 * 1024 ** 3
    if (normalizer.processed_bytes() + reserve > resources['new_processed_cap_bytes']
            or shutil.disk_usage(base.WORK).free < resources['minimum_free_disk_bytes'] + reserve):
        raise ValueError('insufficient coding finalization/compilation/export storage')


def finalize():
    stage_resources(); ranking.configure()
    frozen = json.loads((FROZEN / 'summary.json').read_text())
    if frozen['frozen_database_sha256'] != base.file_sha(FROZEN / 'frozen-aliases.sqlite'):
        raise ValueError('prior family aliases changed')
    if not frozen['all_candidate_aliases_included'] or not frozen['all_ancestral_folds_preserved']:
        raise ValueError('prior fold coverage incomplete')
    selector.STATE = STATE; selector.OUT = OUT; selector.SCORE = SCORE; selector.SECOND = FROZEN
    selector.load_candidate = archive.load_candidate
    selector.ROWS_PER_SHARD = 10000; selector.main()
    base.save(STATE / 'release-scope.json', {'scope': 'IFM coding only; all eligible effort answer versions compared under identical full conditions.',
        'frozen_prior_sha256': base.file_sha(FROZEN / 'summary.json'),
        'storage_loader_sha256': base.file_sha(Path(archive.__file__)),
        'candidate_storage_format': archive.FORMAT,
        'unchanged_selection_algorithm_sha256': base.file_sha(Path(selector.__file__)),
        'native_runner_sha256': base.file_sha(ranking.RUNNER),
        'all_minima_independently_recomputed': True, 'all_first_negative_source_queries_held': True,
        'unequal_literal_input_aliases_held_before_scoring': True, 'source_programs_executed': False,
        'all_records_semantically_verified': False, 'not_yet_published': True, 'weights_trained': False})


def prepare_audit():
    binding_path = STATE / 'winner-diagnosis-policy-v1.json'
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        for name, path in [('driver_sha256', Path(__file__)), ('transport_sha256', Path(audit_transport.__file__)),
                           ('plan_sha256', STATE / 'semantic-audit-v1/plan.json')]:
            if binding[name] != base.file_sha(path): raise ValueError('frozen IFM coding winner audit changed')
        return
    if not (STATE / 'release-scope.json').exists(): raise ValueError('complete winner selection required')
    compiled = json.loads((STATE / 'compile-summary.json').read_text())
    compiled_ids = set()
    for path in sorted((STATE / 'compiled').glob('[0-9]*.json')):
        receipt = json.loads(path.read_text())
        for item in receipt['files']:
            if item['config'] == 'records':
                if base.file_sha(OUT / item['path']) != item['sha256']: raise ValueError('compiled values changed')
                compiled_ids.update(pq.read_table(OUT / item['path'], columns=['record_id']).column('record_id').to_pylist())
    if len(compiled_ids) != compiled['accepted']: raise ValueError('compiled diagnostic candidate coverage differs')
    reviewed_queries = {json.loads(line)['query_hash'] for line in (SOURCE / 'normalized/pilot-ifm.jsonl').open()}
    pools = defaultdict(dict)
    for path in sorted((STATE / 'selected').glob('*.jsonl')):
        for line in path.open():
            row = json.loads(line)
            if row['query_hash'] in reviewed_queries or row['record_id'] not in compiled_ids: continue
            key = row['source_refs']['answer_selection']['method']; pool = pools[key]
            order = base.sha('ifm-code-first-winner-v1:' + row['query_hash'])
            pool[row['query_hash']] = (order, row)
            if len(pool) > 24:
                top = sorted(pool, key=lambda q: pool[q][0])[:12]
                pools[key] = {q: pool[q] for q in top}
    selected = [r for key in sorted(pools) for _, r in sorted(pools[key].values(), key=lambda x: x[0])[:12]]
    package.init(); base.init_tokenizer(); budgets = []; native = []
    jobs = [{'id': 'sample-' + r['record_id'], 'record_id': r['record_id'], 'domain': 'coding',
             'messages': r['messages'], 'tools': r['tools'], 'observation_origin': r['observation_origin'],
             'selection_method': r['source_refs']['answer_selection']['method']} for r in selected]
    original = json.loads((SOURCE / 'semantic-audit-v1/plan.json').read_text())
    for job in original['controls'] + jobs:
        body = audit_transport.body_for(job)
        rendered = package.CHAT.render(messages=body['messages'], tools=[], add_generation_prompt=True,
            enable_thinking=body['chat_template_kwargs']['enable_thinking'], bos_token='', eos_token='<|im_end|>')
        tokens = len(package.QWEN.encode(rendered, add_special_tokens=False).ids)
        if tokens + body['max_tokens'] > 1000000: raise ValueError('complete audit request exceeds served budget')
        budgets.append({'id': job['id'], 'input_tokens': tokens, 'output_reserve': body['max_tokens'],
                        'render_sha256': base.sha(rendered), 'request_sha256': reviewer.digest(body)})
    for row in selected:
        record, _, _ = base.compile_row(row); native.append(package.request_budget(record))
    folder = STATE / 'normalized'; folder.mkdir(exist_ok=True)
    with (folder / 'pilot-selected-winners.jsonl').open('w') as handle:
        for row in selected: handle.write(canonical(row) + '\n')
    plan = {k: original[k] for k in ['schema_sha256', 'prompt_sha256', 'model', 'revision', 'controls']}
    plan.update(jobs=jobs, first_non_accept_is_final_for_that_record=True,
        scope='Up to 12 new normalized questions per PPL/singleton selection method, sampled by fixed question hash. All 54 previously reviewed IFM questions excluded. Not corpus accuracy or all-row semantic verification.',
        coding_completion_reserve=900, math_control_completion_reserve=audit_transport.MATH_RESERVE, reasoning_saved=False, previous_first_audits_replayed=False)
    reviewer.dump(STATE / 'semantic-audit-v1/plan.json', plan)
    base.save(STATE / 'winner-audit-budgets.json', {'review': budgets, 'native_inference': native, 'no_truncation': True})
    base.save(binding_path, {'driver_sha256': base.file_sha(Path(__file__)),
        'transport_sha256': base.file_sha(Path(audit_transport.__file__)),
        'plan_sha256': base.file_sha(STATE / 'semantic-audit-v1/plan.json'),
        'budget_proofs_sha256': base.file_sha(STATE / 'winner-audit-budgets.json'),
        'source_audit_sha256': base.file_sha(SOURCE / 'semantic-audit-v1/summary.json'),
        'compile_summary_sha256': base.file_sha(STATE / 'compile-summary.json'),
        'shared_package_driver_sha256': base.file_sha(Path(shared.__file__)),
        'samples': len(selected), 'first_source_reviewed_queries_excluded': len(reviewed_queries),
        'all_first_incomplete_source_reviews_remain_held': True})
    print(json.dumps({'winner_samples': len(selected), 'source_queries_not_replayed': len(reviewed_queries),
                      'coding_completion_reserve': 900}), flush=True)


def package_init():
    package.STATE = STATE; package.OUT = OUT; package.PUBLIC = PUBLIC; package.WAVE = WAVE
    package.EXTRA_ROW_GATE = None; package.init()
    first = json.loads((SOURCE / 'semantic-audit-v1/summary.json').read_text())
    policy = json.loads((normalizer.STATE / 'normalization-policy.json').read_text())
    package.HOLD_IDS.update(first['excluded_record_ids'])
    package.HOLD_QUERIES.update(policy['first_nonaccept_query_hashes'])
    package.HOLD_QUERIES.update(policy['unequal_literal_query_aliases'])


def package_all(workers):
    prepare_audit()
    binding = json.loads((STATE / 'winner-diagnosis-policy-v1.json').read_text())
    if binding['shared_package_driver_sha256'] != base.file_sha(Path(shared.__file__)):
        raise ValueError('shared complete export gate changed')
    proof = json.loads((PRIOR / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']: raise ValueError('previous IFM math release not verified')
    shared.STATE = STATE; shared.OUT = OUT; shared.PUBLIC = PUBLIC; shared.SOURCE = SOURCE
    shared.WAVE = WAVE; shared.PRIOR = PRIOR; shared.prepare_audit = prepare_audit; shared.package_init = package_init
    stage_resources(); shared.package_all(workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['finalize', 'compile', 'prepare-audit', 'audit', 'package'])
    parser.add_argument('--workers', type=int, default=8); args = parser.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('CPU worker bound')
    if args.phase == 'finalize': finalize()
    elif args.phase == 'compile':
        if not (STATE / 'release-scope.json').exists(): raise ValueError('selection incomplete')
        stage_resources(); base.STATE = STATE; base.OUT = OUT; base.compile_all(args.workers, False)
    elif args.phase == 'prepare-audit': prepare_audit()
    elif args.phase == 'audit':
        prepare_audit(); reviewer.STATE = STATE; reviewer.review = audit_transport.review; reviewer.main()
    else: package_all(args.workers)
