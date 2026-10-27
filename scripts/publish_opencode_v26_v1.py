"""Append source-qualified OpenCode winners to the fully verified NI release."""
import argparse
from collections import Counter
import fcntl
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from huggingface_hub import HfApi, DatasetCard, CommitOperationAdd
from scripts import build_memory_sft_v2 as base
from scripts import publish_memory_cumulative_v2 as transport
from scripts.publish_natural_instructions_v26_v1 import STATE as PARENT, DEST as PARENT_DEST, link_checked, make_card
from scripts.release_opencode_v26_v1 import STATE as SELECTED, SOURCE, PUBLIC

STATE = SELECTED / 'cumulative-publication-v1'
DEST = base.WORK / 'exports/memory-harness-v2-cumulative-0005'
FIRST = base.WORK / 'reports/memory-harness-v2'


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists(): raise ValueError('OpenCode publication plan already frozen')
    proof = json.loads((PARENT / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']: raise ValueError('parent not fully verified')
    parent_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    prior = json.loads((PARENT / 'summary.json').read_text())
    new = json.loads((SELECTED / 'public-package-summary.json').read_text())
    for key in ['all_three_configs_exact_readback', 'all_answer_free_qwen_chat_and_completion_budgets_measured',
                'all_native_budgets_include_4096_output_reserve', 'all_previous_folds_preserved',
                'no_published_question_overlap', 'first_nonaccept_query_holds_preserved', 'all_budget_proof_identities_equal']:
        if not new[key]: raise ValueError('new wave gate missing: ' + key)
    for item in parent_plan['manifest']:
        if item['path'] not in ['README.md', 'NOTICE.md', 'PUBLICATION.json']:
            link_checked(PARENT_DEST / item['path'], DEST / item['path'], item['sha256'])
    for item in new['files']: link_checked(PUBLIC / item['path'], DEST / item['path'], item['sha256'])
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('ATTACH DATABASE ? AS previous', (str(PARENT / 'public-index.sqlite'),))
    db.execute('ATTACH DATABASE ? AS incoming', (str(SELECTED / 'public-index.sqlite'),))
    if db.execute('SELECT 1 FROM incoming.records r JOIN previous.records p USING(query_hash) LIMIT 1').fetchone():
        raise ValueError('new coding question overlaps previous release')
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records'); db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)'); db.commit()
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative family/query leakage')
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
              for key in ['record_id', 'query_hash', 'family_id']}; db.close()
    counts = Counter(prior['counts']); counts.update(new['counts'])
    if counts['accepted'] != unique['record_id']: raise ValueError('count mismatch')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    summary = {'wave': 'cumulative-0005', 'counts': dict(counts), 'unique_counts': unique,
        'files': prior['files'] + new['files'], 'new_rows': new['counts']['accepted'],
        'prior_release': proof['final_head'], 'library_commit': code,
        'new_wave': {k: v for k, v in new.items() if k != 'files'},
        'prior_rows_preserved_byte_for_byte': True, 'all_cumulative_folds_preserved': True,
        'accepted_episode_target': 10000000, 'goal_complete': False, 'model_weights_trained': False,
        'model_efficacy_measured': False, 'status': 'LOCAL_VERIFIED_NOT_YET_PUBLISHED'}
    base.save(STATE / 'summary.json', summary)
    report_dir = DEST / 'reports/cumulative-0005'; report_dir.mkdir(parents=True, exist_ok=True)
    base.save(report_dir / 'summary.json', {k: v for k, v in summary.items() if k != 'files'})
    base.save(report_dir / 'data-files.json', summary['files'])
    shutil.copyfile(PARENT_DEST / 'PUBLICATION.json', report_dir / 'prior-publication.json')
    for source, name in [(SELECTED / 'ppl-score-binding.json', 'opencode-ppl-score-binding.json'),
        (SELECTED / 'release-scope.json', 'opencode-release-scope.json'),
        (SOURCE / 'semantic-audit-v1/summary.json', 'opencode-first-source-diagnosis.json'),
        (SELECTED / 'semantic-audit-v1/summary.json', 'opencode-first-winner-diagnosis.json'),
        (SOURCE / 'full-normalization-policy-v1.json', 'opencode-source-policy.json'),
        (FIRST / 'opencode-memory-adapter-tests-v1.log', 'opencode-memory-adapter-tests-v1.log'),
        (FIRST / 'opencode-static-gate-tests-v1.log', 'opencode-static-gate-tests-v1.log')]:
        shutil.copyfile(source, report_dir / name)
    lock = json.loads((SOURCE / 'build-source-lock.json').read_text())
    base.save(report_dir / 'opencode-source-attribution.json', {'files': [{k: v for k, v in s.items() if k != 'local'} for s in lock['files']],
        'license': 'cc-by-4.0', 'source_programs_executed': False})
    notice = (PARENT_DEST / 'NOTICE.md').read_text() + '''

## OpenCodeInstruct contextual coding addition

Source: https://huggingface.co/datasets/nvidia/OpenCodeInstruct/tree/8f3ba5bafe4d6e8db46082cf7ae6741bc370604d . License: CC-BY-4.0. Original coding tasks and public responses remain unchanged. The adapter places the task specification and supplied examples into causal front/back memory, adds coding skills, retains explicit source provenance and assistant-only targets, and chooses minimum native-v26 answer PPL among eligible complete alternatives. Recorded upstream tests must all pass and all three recorded rubric scores must be at least 4/5; these are not independent local execution evidence. Python code blocks are parsed as syntax without execution. Post-hoc unit tests and judge comments are not prompt memory. First non-accept source and winner queries remain excluded. See reports/cumulative-0005/opencode-source-attribution.json and each source_refs_json for file, row, source hashes, generation cohort and verification metadata.
'''
    (DEST / 'NOTICE.md').write_text(notice)
    ni = prior['new_wave']
    card = make_card(counts, unique, code, ni['source_audit'], ni['winner_audit'])
    card = card.replace('reports/cumulative-0004/', 'reports/cumulative-0005/').replace('NI selection and audit report', 'current release and OpenCode audit report')
    extra = f'''\nThe OpenCode addition contains **{new['counts']['accepted']:,}** new distinct coding questions with full task specifications available as memory. All source tests are recorded as passing with consistent counts and test score 1.0, and three recorded source judge scores are at least 4/5. Those tests were **not executed locally**; static Python parsing is not semantic correctness proof. Source first diagnosis: `{json.dumps(new['source_audit'], sort_keys=True)}` in 48 cases; winner first diagnosis: `{json.dumps(new['winner_audit'], sort_keys=True)}` in 32 cases. All first non-accept questions remain held, including ambiguous or contradictory criticisms. These are diagnostic counts, not corpus error rates. The new coding questions have no normalized-query overlap with the previous active release, and all previous family folds are preserved. No source test output or post-hoc judge text was inserted into prompt memory.\n\n'''
    card = card.replace('First NI source diagnosis:', extra + 'First NI source diagnosis:')
    (DEST / 'README.md').write_text(card); DatasetCard(card).validate()
    text = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>OpenCode v26 標準答案擴充</title>
<style>body{{font:17px/1.7 system-ui;max-width:1080px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>OpenCode 與 v26 最低 PPL 標準答案</h1><p>累計 {counts['accepted']:,} 筆新資料，本次新增 {new['counts']['accepted']:,} 個不同 coding 問題。目標一千萬筆仍在進行。</p>
<p>原始 5,000,000 筆先套用來源測試、來源評分、既有問題、評測重疊、同題不同字面上下文與首次審查隔離。保留 1,071,900 個候選答案；實際為 1,064,923 個不同答案、1,031,929 個完整輸入，15,843 組比較、48,837 個答案以凍結 v26 實際評分。每組完整比較後選最低答案 PPL；其餘單一答案沒有假裝跑過 PPL。</p>
<p>上游測試紀錄不等於本機重新執行。原始程式碼保持文字，僅解析 Python AST 語法。來源抽查 48 筆與選取答案抽查 32 筆分開報告，首輪非通過問題全部隔離。審查仍有矛盾與模糊判詞，不能直接拿否決比例當作程式錯誤率。</p>
<p>全部原文、tokens、labels、Chat／Completion 與原生 MiniCPM 長度均完整檢查；保留 4096 輸出 tokens，不裁切。輸入包含使用者的完整程式需求與範例，沒有洩漏後設測試或評審答案。沒有訓練模型，也沒有量測新模型能力提升。</p>
<pre>{html.escape(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False, indent=2))}</pre></html>'''
    (report_dir / 'report.html').write_text(text)
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    old_hashes = {x['path']: x['sha256'] for x in parent_plan['manifest']}
    additions = [x['path'] for x in manifest if old_hashes.get(x['path']) != x['sha256']]
    base.save(STATE / 'publication-plan.json', {'parent_commit': proof['final_head'], 'parent_plan_sha256': base.file_sha(PARENT / 'publication-plan.json'),
        'library_commit': code, 'manifest': manifest, 'accepted': counts['accepted'], 'incremental_add_paths': additions,
        'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26',
        'goal_phase': 'v26_ni_and_opencode_published_continuing_to_10m'})
    print(json.dumps({'accepted': counts['accepted'], 'new_rows': new['counts']['accepted'], 'files': len(manifest), 'changed_or_added_files': len(additions)}), flush=True)


def publish():
    if (STATE / 'publication-data-commit.json').exists(): raise ValueError('OpenCode wave already committed')
    plan = json.loads((STATE / 'publication-plan.json').read_text()); api = HfApi()
    if base.file_sha(PARENT / 'publication-plan.json') != plan['parent_plan_sha256']:
        raise ValueError('parent manifest binding changed')
    if api.repo_info(transport.DATASET, repo_type='dataset').sha != plan['parent_commit']:
        raise ValueError('remote head changed before append')
    actual = set(HfApi(token=False).list_repo_files(transport.DATASET, repo_type='dataset', revision=plan['parent_commit']))
    parent_plan = json.loads((PARENT / 'publication-plan.json').read_text())
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
        parent_commit=plan['parent_commit'], commit_message='Append source-qualified OpenCode with complete frozen-v26 answer selection and full format proofs', num_threads=8)
    base.save(STATE / 'publication-data-commit.json', {'commit': commit.oid, 'parent': plan['parent_commit'],
        'plan_sha256': base.file_sha(STATE / 'publication-plan.json')})
    print(json.dumps({'commit': commit.oid, 'accepted': plan['accepted']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'publish', 'verify']); args = p.parse_args()
    with (FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.STATE = STATE; transport.DEST = DEST; transport.FIRST = FIRST
        if args.phase == 'prepare': prepare()
        elif args.phase == 'publish': publish()
        else: transport.verify()
