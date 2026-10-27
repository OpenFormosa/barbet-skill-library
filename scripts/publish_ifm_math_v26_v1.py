"""Append verified IFM math winners, preserving the corrected current release."""
import argparse
from collections import Counter
import fcntl
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from huggingface_hub import DatasetCard, HfApi, CommitOperationAdd
from scripts import build_memory_sft_v2 as base
from scripts import publish_memory_cumulative_v2 as transport
from scripts.publish_natural_instructions_v26_v1 import link_checked, make_card, STATE as NI_PARENT
from scripts.publish_source_contract_pruned_v1 import STATE as PARENT, DEST as PARENT_DEST
from scripts.release_ifm_math_v26_v1 import STATE as SELECTED, PUBLIC, SOURCE
from scripts.normalize_ifm_math_memory_v1 import STATE as NORMALIZED

STATE = SELECTED / 'cumulative-publication-v1'
DEST = base.WORK / 'exports/memory-harness-v2-cumulative-0007'
FIRST = base.WORK / 'reports/memory-harness-v2'


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists(): raise ValueError('IFM publication plan already frozen')
    proof = json.loads((PARENT / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']: raise ValueError('corrected parent not fully verified')
    prior_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    prior = json.loads((PARENT / 'summary.json').read_text())
    new = json.loads((SELECTED / 'public-package-summary.json').read_text())
    source_scope = json.loads((SELECTED / 'final-source-scope-readback-v2.json').read_text())
    if (not source_scope['all_public_record_scope_gates_reread'] or source_scope['accepted'] != new['counts']['accepted']
            or source_scope['public_package_summary_sha256'] != base.file_sha(SELECTED / 'public-package-summary.json')):
        raise ValueError('prospective math source-scope gate incomplete')
    for key in ['all_three_configs_exact_readback', 'all_answer_free_qwen_chat_and_completion_budgets_measured',
                'all_native_budgets_include_4096_output_reserve', 'all_previous_folds_preserved',
                'no_published_question_overlap', 'first_nonaccept_query_holds_preserved', 'all_budget_proof_identities_equal']:
        if not new[key]: raise ValueError('new wave gate missing: ' + key)
    for item in prior_plan['manifest']:
        if item['path'] not in ['README.md', 'NOTICE.md', 'PUBLICATION.json']:
            link_checked(PARENT_DEST / item['path'], DEST / item['path'], item['sha256'])
    for item in new['files']: link_checked(PUBLIC / item['path'], DEST / item['path'], item['sha256'])
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('ATTACH DATABASE ? AS previous', (str(PARENT / 'public-index.sqlite'),))
    db.execute('ATTACH DATABASE ? AS incoming', (str(SELECTED / 'public-index.sqlite'),))
    if db.execute('SELECT 1 FROM incoming.records r JOIN previous.records p USING(query_hash) LIMIT 1').fetchone():
        raise ValueError('new math question overlaps previous release')
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records'); db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)'); db.commit()
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative question/family leakage')
    holds = json.loads((PARENT_DEST / 'reports/cumulative-0006/query-wide-holds.json').read_text())['query_hashes']
    if db.execute('SELECT 1 FROM records WHERE query_hash IN (' + ','.join('?' for _ in holds) + ') LIMIT 1', holds).fetchone():
        raise ValueError('previous source-contract query hold revived')
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
              for key in ['record_id', 'query_hash', 'family_id']}; db.close()
    counts = Counter(prior['counts']); counts.update(new['counts'])
    if counts['accepted'] != unique['record_id']: raise ValueError('cumulative count mismatch')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    summary = {'wave': 'cumulative-0007', 'counts': dict(counts), 'unique_counts': unique,
        'files': prior['files'] + new['files'], 'new_rows': new['counts']['accepted'],
        'prior_release': proof['final_head'], 'library_commit': code,
        'new_wave': {k: v for k, v in new.items() if k != 'files'},
        'new_math_source_scope': source_scope,
        'prior_rows_preserved_byte_for_byte': True, 'all_cumulative_folds_preserved': True,
        'all_previous_source_contract_query_holds_preserved': True,
        'accepted_episode_target': 10000000, 'goal_complete': False, 'model_weights_trained': False,
        'model_efficacy_measured': False, 'status': 'LOCAL_VERIFIED_NOT_YET_PUBLISHED'}
    base.save(STATE / 'summary.json', summary)
    folder = DEST / 'reports/cumulative-0007'; folder.mkdir(parents=True, exist_ok=True)
    base.save(folder / 'summary.json', {k: v for k, v in summary.items() if k != 'files'})
    base.save(folder / 'data-files.json', summary['files'])
    for source, name in [(SELECTED / 'ppl-score-binding.json', 'ifm-math-ppl-score-binding.json'),
        (SELECTED / 'release-scope.json', 'ifm-math-release-scope.json'),
        (SOURCE / 'semantic-audit-v1/summary.json', 'ifm-first-source-diagnosis.json'),
        (SELECTED / 'semantic-audit-v1/summary.json', 'ifm-math-first-winner-diagnosis.json'),
        (SELECTED / 'winner-diagnosis-policy-v1.json', 'ifm-math-winner-policy.json'),
        (NORMALIZED / 'normalization-policy.json', 'ifm-math-source-policy.json'),
        (NORMALIZED / 'v26-selection-v1/first-launch-import-failure-v1.json', 'ifm-native-first-import-failure.json'),
        (NORMALIZED / 'v26-selection-v1/score-launch-native-v2.json', 'ifm-native-prospective-entrypoint.json'),
        (NORMALIZED / 'v26-selection-v1/answer-selection-diagnostics-v1.json', 'ifm-ppl-diagnostics.json'),
        (SELECTED / 'source-scope-v2/policy.json', 'ifm-math-source-scope-policy.json'),
        (SELECTED / 'source-scope-v2/query-holds.json', 'ifm-math-additional-query-holds.json'),
        (SELECTED / 'source-scope-v2/exact-arithmetic-proofs.json', 'ifm-math-exact-arithmetic-proofs.json'),
        (SELECTED / 'final-source-scope-readback-v2.json', 'ifm-math-source-scope-readback.json'),
        (SELECTED / 'independent-graph-counterexample-v1.json', 'ifm-math-independent-graph-counterexample.json'),
        (PARENT_DEST / 'PUBLICATION.json', 'prior-publication.json'),
        (FIRST / 'ifm-math-candidate-tests-v1.log', 'ifm-math-candidate-tests-v1.log'),
        (FIRST / 'arithmetic-verifier-tests-v1.log', 'arithmetic-verifier-tests-v1.log'),
        (FIRST / 'math-audit-transport-tests-v2.log', 'math-audit-transport-tests-v2.log')]:
        shutil.copyfile(source, folder / name)
    jobs = json.loads((NORMALIZED / 'normalization-jobs.json').read_text())
    base.save(folder / 'ifm-math-source-attribution.json', {'files': [{k: v for k, v in job['source'].items() if k != 'local'} for job in jobs],
        'license': 'cc-by-4.0', 'upstream_original_dataset_not_provided_per_row': True,
        'source_programs_executed': False, 'separate_teacher_reasoning_fields_excluded': True})
    notice = (PARENT_DEST / 'NOTICE.md').read_text() + '''

## IFM TxT360 math addition

Source: https://huggingface.co/datasets/IFM/TxT360-3efforts/tree/bfc4a082d11967cd7810fe0b773be87bf54fb32e . License: CC-BY-4.0. The source is a regenerated response collection and does not provide original dataset attribution per row; no missing attribution is fabricated here. All seven downloaded math files and all effort variants are inventoried. Original task specifications become causal front/back memory with arithmetic and unit skills. Separate teacher think fields are excluded, and only assistant answers carry loss. Different effort levels are candidate answer versions, never extra questions. Eligible complete alternatives are selected by minimum native-v26 conditional answer PPL. Source and winner first non-accept queries remain held, including incomplete review output. These are source/structure checks and bounded diagnostic reviews, not whole-corpus semantic verification or execution evidence. See reports/cumulative-0007/ifm-math-source-attribution.json and per-row source_refs_json.
'''
    (DEST / 'NOTICE.md').write_text(notice)
    ni = json.loads((NI_PARENT / 'summary.json').read_text())['new_wave']
    card = make_card(counts, unique, code, ni['source_audit'], ni['winner_audit'])
    card = card.replace('reports/cumulative-0004/', 'reports/cumulative-0007/').replace('NI selection and audit report', 'current IFM math selection and audit report')
    card += f'''\n\n## Current IFM math addition\n\nThis wave adds **{new['counts']['accepted']:,}** distinct normalized math questions. Its 356,955 raw source rows yielded 339,279 eligible candidate versions and 339,195 distinct answers across 313,153 full input groups. All 24,933 competing groups, containing 50,975 candidate answers, use the unchanged frozen native v26 scorer before final export gates. Singletons are not represented as measured PPL. The original first launcher failed on an audit-only import dependency before any score; its evidence is retained, and the isolated prospective entrypoint leaves the numerical environment and scorer unchanged.\n\nThe IFM source pilot covers 54 coding/tool/math cases: `{json.dumps(new['source_audit'], sort_keys=True)}`. Its math subset has 8 ACCEPT, 1 REVISE and 9 incomplete-output HOLD across 18 cases. The new math winner diagnosis is `{json.dumps(new['winner_audit'], sort_keys=True)}` across {new['winner_audit_samples']} different, previously unreviewed questions, with an 8,192-token completion reserve measured before requests. The original incomplete reviews were not replayed; private reviewer reasoning is not saved or inserted into training memory. Diagnostics are not corpus accuracy or evidence of improved model ability.\n\nThe previously verified source-contract correction remains active: all 33 views of 21 held source-contract questions stay excluded. OpenCode's 1,031,922 coding episodes remain documented in reports/cumulative-0005/. Retained historical reports remain available. No model was trained or newly benchmarked.\n'''
    card += f'''\nAn additional all-selected-question scope scan holds {source_scope['scope_held_queries']} questions before public admission, without replacing their winners. It excludes the crossword cohort lacking supplied lexical evidence and independently established numerical defects. A bounded exact rational arithmetic verifier checks only its explicit whole-question integer-calculation grammar, leaving all unsupported forms unverified. For example, the source's sheep-purchase conditions imply a negative number of people; that question is held rather than silently corrected. A high-PPL but correct -17 arithmetic answer remains valid: no PPL threshold is used as a correctness test. See the additional query-hold policy and exact verification proofs in reports/cumulative-0007/.\n'''
    card += '\nOne first-rejected graph answer claims a minimum of 199 students. An independently authored 149-vertex bipartite graph was checked to contain every degree 1 through 99 and no triangles, disproving that answer. The finite counterexample receipt is included; the question stays held without replacement.\n'
    (DEST / 'README.md').write_text(card); DatasetCard(card).validate()
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>IFM 數學 v26 標準答案發布</title><style>body{{font:17px/1.7 system-ui;max-width:1080px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>IFM 數學：同題取 v26 最低答案 PPL</h1><p>本波新增 {new['counts']['accepted']:,} 個獨立正規化數學問題，累計 {counts['accepted']:,} 筆有效新資料。一千萬筆目標仍未完成。</p><p>完整來源 356,955 列，保留 339,279 候選，去除相同答案後 339,195 個版本、313,153 個完整輸入。24,933 組、50,975 個答案需比較；使用原本已校準的固定 native v26、完整不裁切輸入、batch size 1、assistant-only 條件 PPL，選擇每組最低者。所有其他發布門檻與首次隔離仍有效，不用次佳答案替換被隔離的問題。</p><p>來源數學抽查 8 ACCEPT、1 REVISE、9 HOLD；HOLD 是審核輸出未完成。新的選取答案抽查為 {html.escape(json.dumps(new['winner_audit']))}，共 {new['winner_audit_samples']} 個未審核問題，事前測量 8,192 個輸出 tokens 預算。不能以不同題目與預算宣稱受控準確率改善。首次失敗全部保留，教師私有推理沒有存入資料。</p><p>來源問題本身作為記憶，外加技能；不將答案或獨立 think 欄位放進記憶。全部文字、tokens、labels 與 answer-free Chat／Completion 預算完整核對。來源回應經結構檢查不等於逐題語義驗證，沒有執行來源程式，沒有訓練權重或聲稱追平大模型。</p><pre>{html.escape(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False, indent=2))}</pre></html>'''
    body = body.replace('<pre>', f'<p>追加來源範圍檢查隔離 {source_scope["scope_held_queries"]} 題，包括未提供字詞證據的填字題與已獨立證明條件矛盾的買羊題。精確算術檢查器只涵蓋明確的整數計算語法；本波 3 題通過，不能宣稱全部數學答案都經精確驗算。所有 PPL、首次判詞及保留資料原文維持原值，不以次低 PPL 取代被隔離的題目。</p><pre>', 1)
    (folder / 'report.html').write_text(body)
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    previous = {item['path']: item['sha256'] for item in prior_plan['manifest']}
    base.save(STATE / 'publication-plan.json', {'parent_commit': proof['final_head'], 'parent_plan_sha256': base.file_sha(PARENT / 'publication-plan.json'),
        'library_commit': code, 'manifest': manifest, 'accepted': counts['accepted'],
        'incremental_add_paths': [f['path'] for f in manifest if previous.get(f['path']) != f['sha256']],
        'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26',
        'goal_phase': 'ifm_math_v26_winners_published_continuing_to_10m'})
    print(json.dumps({'accepted': counts['accepted'], 'new_rows': new['counts']['accepted'], 'files': len(manifest)}), flush=True)


def publish():
    if (STATE / 'publication-data-commit.json').exists(): raise ValueError('IFM math wave already committed')
    plan = json.loads((STATE / 'publication-plan.json').read_text()); api = HfApi()
    if base.file_sha(PARENT / 'publication-plan.json') != plan['parent_plan_sha256']: raise ValueError('parent manifest changed')
    if api.repo_info(transport.DATASET, repo_type='dataset').sha != plan['parent_commit']: raise ValueError('remote head changed before append')
    actual = set(HfApi(token=False).list_repo_files(transport.DATASET, repo_type='dataset', revision=plan['parent_commit']))
    old_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    if actual != {f['path'] for f in old_plan['manifest']} | {'.gitattributes', 'PUBLICATION.json'}:
        raise ValueError('parent tree differs from verified release')
    old = {f['path']: f['sha256'] for f in old_plan['manifest']}
    changed = {f['path'] for f in plan['manifest'] if old.get(f['path']) != f['sha256']}
    if changed != set(plan['incremental_add_paths']): raise ValueError('incremental scope differs')
    operations = transport.replacement_deletions(actual, [f['path'] for f in plan['manifest']])
    for item in plan['manifest']:
        if base.file_sha(DEST / item['path']) != item['sha256']: raise ValueError('planned bytes changed')
        if item['path'] in changed: operations.append(CommitOperationAdd(item['path'], DEST / item['path']))
    commit = api.create_commit(transport.DATASET, repo_type='dataset', operations=operations,
        parent_commit=plan['parent_commit'], commit_message='Append full-context IFM math winners selected by frozen v26 minimum answer PPL; retain all prior query holds', num_threads=8)
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
