"""Append verified IFM coding winners while retaining every earlier query hold."""
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
from scripts.publish_ifm_math_v26_v1 import STATE as PARENT, DEST as PARENT_DEST
from scripts.release_ifm_code_v26_v1 import STATE as SELECTED, PUBLIC, SOURCE
from scripts.normalize_ifm_code_memory_v1 import STATE as NORMALIZED
from scripts.audit_ifm_code_scope_v2 import STATE as SCOPE

STATE = SELECTED / 'cumulative-publication-v1'
DEST = base.WORK / 'exports/memory-harness-v2-cumulative-0008'
FIRST = base.WORK / 'reports/memory-harness-v2'


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists():
        raise ValueError('coding publication plan already frozen')
    proof = json.loads((PARENT / 'publication-final-commit.json').read_text())
    if not proof['anonymous_full_file_sha256_readback']:
        raise ValueError('previous cumulative release not verified')
    prior_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    prior = json.loads((PARENT / 'summary.json').read_text())
    new = json.loads((SELECTED / 'public-package-summary.json').read_text())
    scope = json.loads((SELECTED / 'final-source-scope-readback-v1.json').read_text())
    if (not scope['all_public_record_scope_gates_reread']
            or scope['public_package_summary_sha256'] != base.file_sha(SELECTED / 'public-package-summary.json')
            or scope['source_scope_summary_sha256'] != base.file_sha(SCOPE / 'summary.json')):
        raise ValueError('coding static scope package proof missing or changed')
    for key in ['all_three_configs_exact_readback', 'all_answer_free_qwen_chat_and_completion_budgets_measured',
                'all_native_budgets_include_4096_output_reserve', 'all_previous_folds_preserved',
                'no_published_question_overlap', 'first_nonaccept_query_holds_preserved', 'all_budget_proof_identities_equal']:
        if not new[key]:
            raise ValueError('new coding release gate missing: ' + key)
    for item in prior_plan['manifest']:
        if item['path'] not in ['README.md', 'NOTICE.md', 'PUBLICATION.json']:
            link_checked(PARENT_DEST / item['path'], DEST / item['path'], item['sha256'])
    for item in new['files']:
        link_checked(PUBLIC / item['path'], DEST / item['path'], item['sha256'])
    policy = json.loads((NORMALIZED / 'normalization-policy.json').read_text())
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('ATTACH DATABASE ? AS previous', (str(PARENT / 'public-index.sqlite'),))
    db.execute('ATTACH DATABASE ? AS incoming', (str(SELECTED / 'public-index.sqlite'),))
    if db.execute('SELECT 1 FROM incoming.records JOIN previous.records USING(query_hash) LIMIT 1').fetchone():
        raise ValueError('incoming coding question overlaps the previous release')
    holds = policy['first_nonaccept_query_hashes'] + policy['unequal_literal_query_aliases']
    if holds and db.execute('SELECT 1 FROM incoming.records WHERE query_hash IN (' + ','.join('?' for _ in holds) + ') LIMIT 1', holds).fetchone():
        raise ValueError('previous source or condition hold reintroduced')
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records')
    db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)'); db.commit()
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative question or family split leakage')
    old_holds = json.loads((PARENT_DEST / 'reports/cumulative-0006/query-wide-holds.json').read_text())['query_hashes']
    if db.execute('SELECT 1 FROM records WHERE query_hash IN (' + ','.join('?' for _ in old_holds) + ') LIMIT 1', old_holds).fetchone():
        raise ValueError('previous tool-contract question hold reintroduced')
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
              for key in ['record_id', 'query_hash', 'family_id']}; db.close()
    counts = Counter(prior['counts']); counts.update(new['counts'])
    if counts['accepted'] != unique['record_id']:
        raise ValueError('cumulative record count differs')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    summary = {'wave': 'cumulative-0008', 'counts': dict(counts), 'unique_counts': unique,
        'files': prior['files'] + new['files'], 'new_rows': new['counts']['accepted'], 'prior_release': proof['final_head'],
        'library_commit': code, 'new_wave': {k: v for k, v in new.items() if k != 'files'},
        'prior_rows_preserved_byte_for_byte': True, 'all_cumulative_folds_preserved': True,
        'all_previous_source_contract_query_holds_preserved': True,
        'accepted_episode_target': 10000000, 'goal_complete': False,
        'model_weights_trained': False, 'model_efficacy_measured': False,
        'status': 'PAYLOAD_VALIDATED; REMOTE_READBACK_REPORTED_SEPARATELY_IN_PUBLICATION_JSON'}
    base.save(STATE / 'summary.json', summary)
    folder = DEST / 'reports/cumulative-0008'; folder.mkdir(parents=True, exist_ok=True)
    base.save(folder / 'summary.json', {k: v for k, v in summary.items() if k != 'files'})
    base.save(folder / 'data-files.json', summary['files'])
    copies = [(SELECTED / 'ppl-score-binding.json', 'ifm-code-ppl-score-binding.json'),
        (SELECTED / 'release-scope.json', 'ifm-code-release-scope.json'),
        (SOURCE / 'semantic-audit-v1/summary.json', 'ifm-first-source-diagnosis.json'),
        (SELECTED / 'semantic-audit-v1/summary.json', 'ifm-code-first-winner-diagnosis.json'),
        (SELECTED / 'winner-diagnosis-policy-v1.json', 'ifm-code-winner-policy.json'),
        (SELECTED / 'final-source-scope-readback-v1.json', 'ifm-code-static-scope-readback.json'),
        (SCOPE / 'summary.json', 'ifm-code-static-scope-summary.json'),
        (SCOPE / 'policy.json', 'ifm-code-static-scope-policy.json'),
        (SCOPE / 'prospective-correction-v2.json', 'ifm-code-static-scope-indentation-correction.json'),
        (SCOPE / 'query-holds.json', 'ifm-code-static-query-holds.json'),
        (SCOPE / 'independent-case-diagnosis.json', 'ifm-code-independent-case-diagnosis.json'),
        (FIRST / 'code-source-scope-tests-v2.log', 'code-source-scope-tests-v2.log'),
        (NORMALIZED / 'normalization-policy.json', 'ifm-code-source-policy.json'),
        (NORMALIZED / 'normalize-summary.json', 'ifm-code-source-summary.json'),
        (NORMALIZED / 'v26-selection-v1/candidate-policy-binding.json', 'ifm-code-framed-native-binding.json'),
        (NORMALIZED / 'v26-selection-v1/candidate-index-summary.json', 'ifm-code-candidate-summary.json'),
        (NORMALIZED / 'v26-selection-v1/episode-identity-v1/summary.json', 'ifm-code-episode-identity-summary.json'),
        (FIRST / 'ifm-code-candidate-tests-v1.log', 'ifm-code-candidate-tests-v1.log'),
        (FIRST / 'candidate-archive-tests-v1.log', 'candidate-archive-tests-v1.log'),
        (FIRST / 'framed-candidate-identity-tests-v1.log', 'framed-candidate-identity-tests-v1.log'),
        (base.WORK / 'reports/memory-harness-v2-candidate-archive-v1/summary.json', 'candidate-archive-native-render-equivalence.json'),
        (base.WORK / 'reports/memory-harness-v2-candidate-archive-v1/index-equivalence-v1/summary.json', 'candidate-archive-index-equivalence.json'),
        (base.REPO / 'configs/memory_sft_v2_scale_10m_resources_v2.json', 'prospective-resources-v2.json'),
        (PARENT_DEST / 'PUBLICATION.json', 'prior-publication.json')]
    for source, name in copies:
        shutil.copyfile(source, folder / name)
    jobs = json.loads((NORMALIZED / 'normalization-jobs.json').read_text())
    base.save(folder / 'ifm-code-source-attribution.json', {'files': [{k: v for k, v in j['source'].items() if k != 'local'} for j in jobs],
        'license': 'cc-by-4.0', 'upstream_original_dataset_not_provided_per_row': True,
        'source_programs_executed': False, 'separate_teacher_reasoning_fields_excluded': True})
    (DEST / 'NOTICE.md').write_text((PARENT_DEST / 'NOTICE.md').read_text() + '''

## IFM TxT360 coding addition

Source: https://huggingface.co/datasets/IFM/TxT360-3efforts/tree/bfc4a082d11967cd7810fe0b773be87bf54fb32e . License: CC-BY-4.0. This wave inventories all 38 downloaded code files and effort variants, retaining complete explicit Python responses that parse statically. The source's separate think fields are excluded. Original tasks become causal memory with code-contract and feedback skills. No downloaded code or tests were executed, and static parsing does not establish semantic correctness. The source does not provide original dataset attribution per row. Every eligible competing answer receives frozen native v26 conditional answer PPL; only its minimum is retained. First non-accept questions stay held without replacement. Lossless framed candidate storage is explicitly bound and does not change complete text, tokens, labels, scoring metric, model weights or prior cohort artifacts.
''')
    ni = json.loads((NI_PARENT / 'summary.json').read_text())['new_wave']
    card = make_card(counts, unique, code, ni['source_audit'], ni['winner_audit'])
    card = card.replace('reports/cumulative-0004/', 'reports/cumulative-0008/').replace('NI selection and audit report', 'current IFM coding selection and audit report')
    card += f'''\n\n## Current IFM coding addition\n\nThis wave adds **{new['counts']['accepted']:,}** coding episodes. All 1,865,716 raw rows across 38 files were inspected; 159,640 complete eligible candidates yielded 159,639 distinct answers and 149,150 full input groups. All 20,734 answers in the 10,245 competing groups received the same frozen native v26 scoring recipe. All saved minima are independently recomputed before export, and singletons remain explicitly unscored.\n\nThe first source coding subset had 11 ACCEPT, 6 REVISE and 1 REJECT across 18 diagnostic cases. The complete source audit covered 54 coding/tool/math cases. The new selected coding audit is `{json.dumps(new['winner_audit'], sort_keys=True)}` across {new['winner_audit_samples']} previously unreviewed questions; all prior source-reviewed queries are excluded from this new diagnostic. Every first non-ACCEPT query stays held without using a runner-up answer. These are bounded diagnostics, not corpus accuracy or whole-corpus correctness certification.\n\nExcluded non-Python-fence rows include tool-call trajectories outside this wave's response scope; their exclusion is not a claim that they are wrong. Complete Python fences were only parsed as syntax, never executed. Missing source programs or test outcomes are not fabricated. The storage adapter exactly reread 2,801,806,238 bytes of candidate payload from 1,188,257,364 bytes of independently checksummed frames. Native full render, token and label equivalence was also checked on 86 real source records, with separate prefix/observation fixtures. New reader and runner hashes are recorded explicitly; numerical scoring functions and prior frozen readers are unchanged.\n\nThe earlier 312,956 IFM math questions and all other retained rows remain byte-for-byte unchanged. Prior tool-contract and math defect holds remain active. No weights were trained and no new model capability improvement or 9B parity was measured.\n'''
    card += f'''\nA subsequent deterministic scan examined every saved coding winner, holding **{scope['scope_held_queries']:,}** questions for limited static scope signals: an unparseable additional explicit Python fence, duplicate literal dictionary keys, or a narrow past-execution claim without recorded observations. Categories overlap. These conservative holds can include illustrative fragments or intentional duplicate-key examples and are not proof that every held answer is wrong. Their union with the ten first non-ACCEPT questions excludes 1,432 questions. No saved answer, input, score, or prior verdict was rewritten. Two sampled defects were independently checked: an edge-time formula contradicting its first provided example and a duplicate-key expected word count. One reviewer's fabrication allegation was not independently established because the source described expected printed output; that first hold still remains. The original pre-scan Markdown indentation assertion failure, its prospective correction, and six passing regression tests are retained in the scope evidence.\n'''
    (DEST / 'README.md').write_text(card); DatasetCard(card).validate()
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>IFM coding v26 標準答案發布</title><style>body{{font:17px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>IFM coding：完整候選與最低 v26 PPL</h1><p>新增 {new['counts']['accepted']:,} 筆，累計 {counts['accepted']:,} 筆新資料；一千萬筆目標未完成。來源 38 檔、1,865,716 列，159,640 合格候選、159,639 不同答案、149,150 組完整輸入。10,245 個競爭組的 20,734 個答案都已實際計分，使用固定 native v26、未裁切輸入、batch size 1、assistant-only 條件 PPL。</p><p>來源 coding 首次抽查 11 通過、6 需修正、1 拒絕；新的選取答案首次抽查為 {html.escape(json.dumps(new['winner_audit']))}。全部首次非通過題目保留隔離，不改選次低答案。語法解析、有限抽查與低 PPL 均不是逐題正確性證明；未執行來源程式或測試，未訓練權重。</p><p>新候選封存格式逐 byte 還原所有完整 JSON；86 個實際樣本的新舊 native 文字、tokens、labels 及完整索引相等，另有前綴與不同觀察條件的測試。新儲存讀取器已獨立綁定，不混用舊 loader 身份。全部公開資料保留完整文字、兩種 tokenizer 的陣列與兩種推論請求的實測預算，跨版本切分與先前隔離不變。</p><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k != 'files'},ensure_ascii=False,indent=2))}</pre></html>'''
    (folder / 'report.html').write_text(body)
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    old = {f['path']: f['sha256'] for f in prior_plan['manifest']}
    base.save(STATE / 'publication-plan.json', {'parent_commit': proof['final_head'],
        'parent_plan_sha256': base.file_sha(PARENT / 'publication-plan.json'), 'library_commit': code,
        'manifest': manifest, 'accepted': counts['accepted'],
        'incremental_add_paths': [f['path'] for f in manifest if old.get(f['path']) != f['sha256']],
        'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26',
        'goal_phase': 'ifm_coding_v26_winners_published_continuing_to_10m'})
    print(json.dumps({'accepted': counts['accepted'], 'new_rows': new['counts']['accepted'], 'files': len(manifest)}), flush=True)


def publish():
    if (STATE / 'publication-data-commit.json').exists():
        raise ValueError('coding wave already committed')
    plan = json.loads((STATE / 'publication-plan.json').read_text()); api = HfApi()
    if base.file_sha(PARENT / 'publication-plan.json') != plan['parent_plan_sha256']:
        raise ValueError('parent manifest changed')
    if api.repo_info(transport.DATASET, repo_type='dataset').sha != plan['parent_commit']:
        raise ValueError('remote head changed before append')
    actual = set(HfApi(token=False).list_repo_files(transport.DATASET, repo_type='dataset', revision=plan['parent_commit']))
    prior_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    if actual != {f['path'] for f in prior_plan['manifest']} | {'.gitattributes', 'PUBLICATION.json'}:
        raise ValueError('parent tree differs from verified release')
    old = {f['path']: f['sha256'] for f in prior_plan['manifest']}
    changed = {f['path'] for f in plan['manifest'] if old.get(f['path']) != f['sha256']}
    if changed != set(plan['incremental_add_paths']):
        raise ValueError('incremental add scope differs')
    operations = transport.replacement_deletions(actual, [f['path'] for f in plan['manifest']])
    for item in plan['manifest']:
        if base.file_sha(DEST / item['path']) != item['sha256']:
            raise ValueError('planned payload bytes changed')
        if item['path'] in changed:
            operations.append(CommitOperationAdd(item['path'], DEST / item['path']))
    commit = api.create_commit(transport.DATASET, repo_type='dataset', operations=operations,
        parent_commit=plan['parent_commit'], commit_message='Append IFM coding winners selected by minimum frozen native-v26 answer PPL; retain full source and first-hold evidence', num_threads=8)
    base.save(STATE / 'publication-data-commit.json', {'commit': commit.oid, 'parent': plan['parent_commit'],
        'plan_sha256': base.file_sha(STATE / 'publication-plan.json')})
    print(json.dumps({'commit': commit.oid, 'accepted': plan['accepted']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'publish', 'verify']); args = parser.parse_args()
    with (FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.STATE = STATE; transport.DEST = DEST; transport.FIRST = FIRST
        if args.phase == 'prepare':
            prepare()
        elif args.phase == 'publish':
            publish()
        else:
            transport.verify()
