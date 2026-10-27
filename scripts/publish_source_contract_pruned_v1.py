"""Remove newly held source-contract queries, retaining every other byte value.

This creates a new release; archived source files, first reviews and scores are
unchanged. No replacement answer is chosen after a new query-wide quality hold.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import DatasetCard, HfApi, CommitOperationAdd

from scripts import build_memory_sft_v2 as base
from scripts import compact_memory_release_v2 as exact
from scripts import publish_memory_cumulative_v2 as transport
from scripts import publish_opencode_v26_v1 as parent
from scripts.publish_natural_instructions_v26_v1 import link_checked, make_card, STATE as NI_PARENT
from scripts.audit_public_source_contracts_v1 import STATE as DIAGNOSIS, POLICY

STATE = DIAGNOSIS / 'publication-pruned-v1'
DEST = base.WORK / 'exports/memory-harness-v2-cumulative-0006'
FIRST = base.WORK / 'reports/memory-harness-v2'
CONFIGS = ['records', 'tokenized', 'minicpm5_tokenized']


def filter_aligned_views(tables, held_queries):
    records = tables['records']; ids = records.column('record_id').to_pylist()
    if len(ids) != len(set(ids)): raise ValueError('duplicate source record identity')
    if any(t.column('record_id').to_pylist() != ids for t in tables.values()):
        raise ValueError('source configs are not aligned')
    keep = [i for i, q in enumerate(records.column('query_hash').to_pylist()) if q not in held_queries]
    kept = set(keep)
    take = pa.array(keep, type=pa.int64())
    return {name: table.take(take) for name, table in tables.items()}, [ids[i] for i in range(len(ids)) if i not in kept]


def contribution(row):
    counts = Counter({'accepted': 1, 'split:' + row['split']: 1})
    for key in ['pangolin_tokens', 'pangolin_supervised_tokens', 'minicpm_tokens', 'minicpm_supervised_tokens', 'assistant_turns', 'tool_observations']:
        counts[key] += row[key]
    for key in ['domain', 'source', 'upstream_source', 'license', 'verification_tier', 'memory_placement', 'observation_origin']:
        counts[key + ':' + str(row[key])] += 1
    for key in ['actual_retrieval', 'recorded_error_followup']: counts[key] += int(row[key])
    length = max(row['pangolin_tokens'], row['minicpm_tokens'])
    counts['length_bucket:' + str(next(n for n in [4096, 8192, 16384, 32768, 65536, 131072] if length <= n))] += 1
    counts['selection:' + json.loads(row['source_refs_json'])['answer_selection']['method']] += 1
    return counts


def filter_job(task):
    kind, item = task; name = base.sha(item['path']); receipt = STATE / 'filtered' / (name + '.json')
    if receipt.exists(): return json.loads(receipt.read_text())
    holds = json.loads((DIAGNOSIS / 'query-wide-holds-v1.json').read_text())
    held_ids = set(holds['record_ids']); held_queries = set(holds['query_hashes'])
    manifest = {x['path']: x for x in json.loads((parent.STATE / 'summary.json').read_text())['files']}
    exact.DEST = DEST; paths = [item['path']] if kind == 'proof' else [item['path'].replace('/records/', '/' + n + '/') for n in CONFIGS]
    for path in paths:
        if base.file_sha(parent.DEST / path) != manifest[path]['sha256']: raise ValueError('verified parent bytes changed')
    files = []; removed = []; delta = Counter()
    if kind == 'proof':
        table = pq.read_table(parent.DEST / item['path']); ids = table.column('record_id').to_pylist()
        keep = [i for i, rid in enumerate(ids) if rid not in held_ids]
        removed = [rid for rid in ids if rid in held_ids]
        if not removed:
            link_checked(parent.DEST / item['path'], DEST / item['path'], item['sha256']); files.append(item)
        else:
            chosen = table.take(pa.array(keep, type=pa.int64()))
            files.append(exact.write_checked(chosen, DEST / item['path'], wave=Path(item['path']).parts[1], config='budget-proof', split=None))
        rows_seen = len(ids)
    else:
        meta = pq.read_table(parent.DEST / item['path'], columns=['record_id', 'query_hash'])
        rows_seen = meta.num_rows
        matched = [r['record_id'] for r in meta.to_pylist() if r['query_hash'] in held_queries]
        if not matched:
            for path in paths:
                link_checked(parent.DEST / path, DEST / path, manifest[path]['sha256']); files.append(manifest[path])
        else:
            tables = {name: pq.read_table(parent.DEST / path) for name, path in zip(CONFIGS, paths)}
            chosen, removed = filter_aligned_views(tables, held_queries)
            if set(removed) != set(matched) or not set(removed) <= held_ids:
                raise ValueError('query-wide hold membership changed')
            for row in tables['records'].to_pylist():
                if row['record_id'] in held_ids: delta.update(contribution(row))
            for name, path in zip(CONFIGS, paths):
                parts = Path(path).parts
                files.append(exact.write_checked(chosen[name], DEST / path, wave=parts[1], config=name, split=Path(path).name.split('-')[0]))
    result = {'kind': kind, 'path': item['path'], 'rows_seen': rows_seen, 'removed_ids': removed,
        'removed_counts': dict(delta), 'files': files, 'retained_values_equal_to_parent': True}
    base.save(receipt, result); return result


def prepare():
    STATE.mkdir(exist_ok=True); (STATE / 'filtered').mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists(): raise ValueError('pruned publication plan already frozen')
    proof = json.loads((parent.STATE / 'publication-final-commit.json').read_text())
    diagnosis = json.loads((DIAGNOSIS / 'summary.json').read_text())
    holds = json.loads((DIAGNOSIS / 'query-wide-holds-v1.json').read_text())
    prior = json.loads((parent.STATE / 'summary.json').read_text()); prior_plan = json.loads((parent.STATE / 'publication-plan.json').read_text())
    if not proof['anonymous_full_file_sha256_readback'] or diagnosis['verified_public_head'] != proof['final_head']:
        raise ValueError('diagnosis does not bind the verified parent')
    if diagnosis['policy_sha256'] != base.file_sha(POLICY): raise ValueError('source contract policy changed')
    policy = {'driver_sha256': base.file_sha(Path(__file__)), 'parent_head': proof['final_head'],
        'parent_plan_sha256': base.file_sha(parent.STATE / 'publication-plan.json'),
        'source_contract_policy_sha256': base.file_sha(POLICY), 'diagnosis_sha256': base.file_sha(DIAGNOSIS / 'summary.json'),
        'holds_sha256': base.file_sha(DIAGNOSIS / 'query-wide-holds-v1.json'),
        'only_query_wide_deletion': True, 'retained_answers_scores_tokens_and_labels_unchanged': True,
        'no_second_choice_fallback': True, 'model_requests': 0}
    policy_path = STATE / 'policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy: raise ValueError('pruning policy changed')
    if not policy_path.exists(): base.save(policy_path, policy)
    tasks = [('record', f) for f in prior['files'] if f['config'] == 'records']
    tasks += [('proof', f) for f in prior['files'] if f['config'] == 'budget-proof']
    files = []; deltas = Counter(); seen = Counter(); removed = {'record': set(), 'proof': set()}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n, result in enumerate(pool.map(filter_job, tasks), 1):
            files += result['files']; deltas.update(result['removed_counts']); seen[result['kind']] += result['rows_seen']
            if removed[result['kind']] & set(result['removed_ids']): raise ValueError('duplicate held identity in parent')
            removed[result['kind']].update(result['removed_ids'])
            if n % 50 == 0 or n == len(tasks): print(json.dumps({'files_checked': n, 'total': len(tasks)}), flush=True)
    if any(removed[k] != set(holds['record_ids']) for k in removed): raise ValueError('record/proof removal coverage mismatch')
    if any(seen[k] != proof['active_new_episodes'] for k in seen): raise ValueError('parent record/proof coverage differs')
    counts = Counter(prior['counts']); counts.subtract(deltas)
    counts['excluded:new_source_tool_contract_query_guard'] += len(holds['record_ids'])
    if any(v < 0 for v in counts.values()): raise ValueError('negative release count')
    index_path = STATE / 'public-index.sqlite'
    if not index_path.exists(): shutil.copyfile(parent.STATE / 'public-index.sqlite', index_path)
    db = sqlite3.connect(index_path)
    db.executemany('DELETE FROM records WHERE record_id=?', [(rid,) for rid in holds['record_ids']]); db.commit()
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0] for key in ['record_id', 'query_hash', 'family_id']}
    if unique['record_id'] != counts['accepted']: raise ValueError('pruned record count mismatch')
    if db.execute('SELECT 1 FROM records WHERE query_hash IN (' + ','.join('?' for _ in holds['query_hashes']) + ') LIMIT 1', holds['query_hashes']).fetchone():
        raise ValueError('held query survives in current index')
    db.close()
    for item in prior_plan['manifest']:
        if item['path'].startswith(('data/', 'proofs/')) or item['path'] in ['README.md', 'NOTICE.md', 'PUBLICATION.json']: continue
        link_checked(parent.DEST / item['path'], DEST / item['path'], item['sha256'])
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    summary = {'wave': 'cumulative-0006', 'counts': dict(counts), 'unique_counts': unique, 'files': files,
        'new_rows': 0, 'removed_rows': len(holds['record_ids']), 'exact_schema_matching_rows': diagnosis['matched_records'],
        'held_queries': len(holds['query_hashes']), 'removed_counts': dict(deltas),
        'prior_release': proof['final_head'], 'library_commit': code,
        'all_retained_records_and_token_arrays_equal': True, 'all_budget_proofs_preserved_for_retained_ids': True,
        'all_prior_folds_preserved_by_deletion': True, 'first_dispositions_and_scores_unchanged': True,
        'source_contract_policy_sha256': base.file_sha(POLICY), 'accepted_episode_target': 10000000,
        'goal_complete': False, 'model_weights_trained': False, 'model_efficacy_measured': False}
    base.save(STATE / 'summary.json', summary)
    folder = DEST / 'reports/cumulative-0006'; folder.mkdir(parents=True, exist_ok=True)
    base.save(folder / 'summary.json', {k: v for k, v in summary.items() if k != 'files'})
    base.save(folder / 'data-files.json', files)
    base.save(folder / 'source-contract-scan.json', {k: v for k, v in diagnosis.items() if k != 'parts'})
    for source, name in [(POLICY, 'source-tool-contract-holds-v1.json'), (DIAGNOSIS / 'query-wide-holds-v1.json', 'query-wide-holds.json'),
                         (parent.DEST / 'PUBLICATION.json', 'prior-publication.json'), (FIRST / 'source-contract-pruning-tests-v1.log', 'source-contract-pruning-tests-v1.log')]:
        shutil.copyfile(source, folder / name)
    notice = (parent.DEST / 'NOTICE.md').read_text() + '\n\n## Source-contract exclusion update\n\nNo new upstream material was added. Records using an unresolved SMS/account-versus-token/address source function contract, and every active record sharing those normalized first queries, are held. Retained source text, answers, likelihood scores and token labels are unchanged. Prior revisions remain in Hub history. See reports/cumulative-0006/source-tool-contract-holds-v1.json.\n'
    (DEST / 'NOTICE.md').write_text(notice)
    ni = json.loads((NI_PARENT / 'summary.json').read_text())['new_wave']
    card = make_card(counts, unique, code, ni['source_audit'], ni['winner_audit']).replace('reports/cumulative-0004/', 'reports/cumulative-0006/').replace('NI selection and audit report', 'current source-contract exclusion report')
    card += f'''\n\n## Source-contract correction\n\nThe prior verified release contained {proof['active_new_episodes']:,} episodes. A full read-only scan found {diagnosis['matched_records']} records with an exact unresolved function-metadata fingerprint: the description says Sms77 account balance while parameter descriptions/defaults describe a token/address interface. The [primary balance documentation](https://docs.seven.io/en/rest-api/endpoints/account) documents no extra query parameters. The {len(holds['query_hashes'])} affected normalized questions are held in full, removing {len(holds['record_ids'])} records including other same-query views. The current active count is **{counts['accepted']:,}**. This is conservative curation of an unresolved source contract, not proof that all those answers are semantically wrong.\n\nEvery retained answer, PPL decision, full training text, token array, label and budget-proof value is unchanged. No second-choice answer replaces a newly held winner. Full remote byte verification is recorded separately in PUBLICATION.json after completion. The OpenCode addition remains 1,031,922 episodes; its source/winner diagnostics remain in reports/cumulative-0005/. No model was trained or newly benchmarked.\n'''
    (DEST / 'README.md').write_text(card); DatasetCard(card).validate()
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>來源工具規格隔離</title><style>body{{font:17px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>來源工具規格不一致：更新目前有效資料</h1><p>完整回查 {proof['active_new_episodes']:,} 筆，其中 {diagnosis['counts']['records_with_tools']:,} 筆包含工具規格。23 筆符合已知可疑 schema，分屬 21 個正規化問題。連同同題另外 10 個視圖，保守隔離 33 筆；新有效筆數為 {counts['accepted']:,}。</p><p>函式描述是 SMS 帳戶餘額，但參數描述與預設值是 token/address 餘額介面。這份來源契約尚不明確，不自行補造一份 schema。PPL 最低不會覆蓋這項品質門檻，也不以第二名答案替換。被保留的全部文字、tokens、labels、PPL 與推論預算證明，均與前一版逐值相同；歷史發布與首次判詞保留。</p><p>這不是 33 筆答案全部錯誤的證明。掃描只針對已知的精確工具指紋，不等於所有工具的語義驗證。沒有新增模型請求、沒有執行來源程式、沒有訓練權重。一千萬筆目標仍在進行。</p><pre>{html.escape(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False, indent=2))}</pre></html>'''
    (folder / 'report.html').write_text(body)
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    old = {f['path']: f['sha256'] for f in prior_plan['manifest']}
    base.save(STATE / 'publication-plan.json', {'parent_commit': proof['final_head'], 'parent_plan_sha256': base.file_sha(parent.STATE / 'publication-plan.json'),
        'library_commit': code, 'manifest': manifest, 'accepted': counts['accepted'],
        'incremental_add_paths': [f['path'] for f in manifest if old.get(f['path']) != f['sha256']],
        'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26; newly held source-contract queries excluded without fallback',
        'goal_phase': 'source_contract_query_holds_applied_continuing_to_10m'})
    print(json.dumps({'accepted': counts['accepted'], 'removed_rows': len(holds['record_ids']), 'changed_files': sum(old.get(f['path']) != f['sha256'] for f in manifest)}), flush=True)


def publish():
    if (STATE / 'publication-data-commit.json').exists(): raise ValueError('pruned release already committed')
    plan = json.loads((STATE / 'publication-plan.json').read_text()); api = HfApi()
    if base.file_sha(parent.STATE / 'publication-plan.json') != plan['parent_plan_sha256']:
        raise ValueError('parent manifest binding changed')
    if api.repo_info(transport.DATASET, repo_type='dataset').sha != plan['parent_commit']:
        raise ValueError('remote head changed before source-contract correction')
    actual = set(HfApi(token=False).list_repo_files(transport.DATASET, repo_type='dataset', revision=plan['parent_commit']))
    parent_plan = json.loads((parent.STATE / 'publication-plan.json').read_text())
    if actual != {f['path'] for f in parent_plan['manifest']} | {'.gitattributes', 'PUBLICATION.json'}:
        raise ValueError('parent tree differs from verified release')
    old = {f['path']: f['sha256'] for f in parent_plan['manifest']}
    changed = {f['path'] for f in plan['manifest'] if old.get(f['path']) != f['sha256']}
    if changed != set(plan['incremental_add_paths']): raise ValueError('incremental operation scope differs')
    operations = transport.replacement_deletions(actual, [f['path'] for f in plan['manifest']])
    for item in plan['manifest']:
        if base.file_sha(DEST / item['path']) != item['sha256']: raise ValueError('planned local bytes changed')
        if item['path'] in changed: operations.append(CommitOperationAdd(item['path'], DEST / item['path']))
    commit = api.create_commit(transport.DATASET, repo_type='dataset', operations=operations,
        parent_commit=plan['parent_commit'], commit_message='Exclude 33 episodes in 21 queries with unresolved source tool contracts; preserve retained answers and v26 selection', num_threads=8)
    base.save(STATE / 'publication-data-commit.json', {'commit': commit.oid, 'parent': plan['parent_commit'],
        'plan_sha256': base.file_sha(STATE / 'publication-plan.json')})
    print(json.dumps({'commit': commit.oid, 'accepted': plan['accepted']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'publish', 'verify'])
    args = parser.parse_args()
    with (FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.STATE = STATE; transport.DEST = DEST; transport.FIRST = FIRST
        if args.phase == 'prepare': prepare()
        elif args.phase == 'publish': publish()
        else: transport.verify()
