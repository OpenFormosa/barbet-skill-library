"""Append the exact NI package to the fully verified v26 release."""
import argparse
from collections import Counter
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

from huggingface_hub import DatasetCard
from scripts import build_memory_sft_v2 as base
from scripts import publish_memory_cumulative_v2 as transport
from scripts.recover_v26_publication_metadata_v1 import STATE as PARENT, DEST as PARENT_DEST
from scripts.prune_v26_query_aliases_v1 import STATE as PARENT_INDEX
from scripts.package_natural_instructions_scope_v2 import STATE as NI, SOURCE, PUBLIC, CONFIG as NI_SCOPE

STATE = NI / 'cumulative-publication-v2'
DEST = base.WORK / 'exports/memory-harness-v2-cumulative-0004-v2'
FIRST = base.WORK / 'reports/memory-harness-v2'


def link_checked(source, destination, expected):
    if base.file_sha(source) != expected: raise ValueError('source payload changed before publication')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists(): os.link(source, destination)
    if base.file_sha(destination) != expected: raise ValueError('destination conflicts with planned payload')


def make_card(counts, unique, code, source_audit, winner_audit):
    configs = '\n'.join(f'''- config_name: {config}
  data_files:
  - split: train
    path: data/wave-*/{config}/train-*.parquet
  - split: validation
    path: data/wave-*/{config}/validation-*.parquet''' for config in ['records', 'tokenized', 'minicpm5_tokenized'])
    return f'''---
license: other
license_name: mixed-source-licenses
license_link: https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/blob/main/NOTICE.md
task_categories:
- text-generation
language:
- en
- zh
configs:
{configs}
---

# Barbet memory, skill and harness SFT

This active release contains **{counts['accepted']:,} new episodes**, {unique['query_hash']:,} distinct normalized first queries and {unique['family_id']:,} conservative task families. Legacy active rows: **0**. The target remains **10,000,000 accepted, deduplicated new episodes**; the goal is unfinished.

For alternatives to the same complete input, the standard answer is the eligible candidate with the lowest conditional supervised-answer PPL under frozen [Barbet v26](https://huggingface.co/voidful/barbet-skill/tree/67131067ff1825179d5e38e91d499159669e9c66). The denominator contains assistant target tokens and assistant EOS only; prompts, memories, observations, unsupervised history and padding are excluded. Native unpadded batch size is one, with no truncation. All competitors must have finite scores. Different literal input conditions are held rather than compared. First negative quality decisions remain held. PPL measures model likelihood, not correctness.

{counts['selection:minimum_v26_conditional_answer_ppl']:,} retained episodes were chosen among scored alternatives; {counts['selection:only_eligible_answer']:,} had a single eligible answer and are not claimed to have measured PPL. Answer alternatives, memory positions and whitespace/case views do not add episodes.

| Count | Value |
|---|---:|
| Train | {counts['split:train']:,} |
| Validation | {counts['split:validation']:,} |
| Pangolin physical tokens | {counts['pangolin_tokens']:,} |
| Pangolin supervised tokens | {counts['pangolin_supervised_tokens']:,} |
| MiniCPM5 physical tokens | {counts['minicpm_tokens']:,} |
| MiniCPM5 supervised tokens | {counts['minicpm_supervised_tokens']:,} |
| Assistant turns | {counts['assistant_turns']:,} |
| Tool observations | {counts['tool_observations']:,} |

The Natural Instructions addition uses licensed official training tasks with the full task definition, different-input public worked examples and target input available as causal memory before the answer. Source outputs are targets, never inserted into target memory. Knowledge tasks lacking necessary evidence, worked-example target overlaps, previous questions and unequal literal input aliases are held. All reference alternatives remain available until actual v26 ranking. This addition teaches instruction and context use; it is not a new executed tool trajectory or measured retrieval improvement.

An NI output array can contain jointly required answer items rather than alternative complete answers. The final source-contract gate excludes unresolved Quoref/DROP/TORQUE/LIMIT and Hotpot support cohorts, additional bare-fact cohorts, intentional wrong-target POS tasks, and specified benchmark source families lacking verified original training provenance. Only explicitly reviewed multi-reference task contracts may supply answer alternatives. All retained inputs, targets and measured scores remain unchanged. The initial NI partial package was stopped before publication, and its original reviews remain preserved.

First NI source diagnosis: `{json.dumps(source_audit, sort_keys=True)}` across 80 diagnostic samples. First NI winner diagnosis: `{json.dumps(winner_audit, sort_keys=True)}`. Every first non-ACCEPT query is excluded. These diagnostics are not random corpus accuracy estimates or all-row semantic approvals; some first negative reviews contain contradictory or ambiguous complaints and remain conservative holds. Upstream answer provenance and verification tiers remain explicit.

The [library, harness and reproducible data code](https://github.com/OpenFormosa/barbet-skill-library/tree/{code}) support source-attributed front/back memory, skill retrieval, registered typed tool callbacks and bounded refinement using real verifier observations. No model weights were trained, and no new model capability gain or 9B parity was measured.

Use `records` for portable messages, tools, complete text, provenance and target spans. `tokenized` holds exact Pangolin IDs and assistant-only labels; `minicpm5_tokenized` holds exact native MiniCPM5-2B IDs and labels. Stored IDs are tokenizer-specific. `prepare_dataset_request` constructs answer-free Chat or exact-renderer Completion inputs and excludes later target turns. Native training and MiniCPM lengths include a 4,096-token output reserve within 131,072 tokens. Both answer-free Qwen request modes are measured against its actual served tokenizer and 1,000,000-token context limit. No truncation is used. These are format and budget checks, not Barbet backend certification.

All records and token/label arrays are read back in full. Budget-proof identities equal accepted record identities. Family and question folds are preserved across releases. The NI addition has no normalized-query overlap with the prior active release. Source code in examples and public test data was not executed locally.

See [source attribution and licenses](NOTICE.md), [current release counts](reports/cumulative-0004/summary.json), [NI selection and audit report](reports/cumulative-0004/report.html), and retained versioned prior reports. `PUBLICATION.json` is created only after full anonymous SHA-256 readback at the new data commit.
'''


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists(): raise ValueError('NI publication plan already frozen')
    prior = json.loads((PARENT / 'publication-final-commit.json').read_text())
    if not prior['anonymous_full_file_sha256_readback']: raise ValueError('prior release unverified')
    old = json.loads((PARENT / 'summary.json').read_text())
    new = json.loads((NI / 'public-package-summary.json').read_text())
    scope = json.loads((NI / 'final-scope-readback-v2.json').read_text())
    if not scope['all_public_record_scope_gates_reread'] or scope['scope_config_sha256'] != base.file_sha(NI_SCOPE) or scope['accepted'] != new['counts']['accepted']:
        raise ValueError('new reference-semantics and grounded-scope gate incomplete')
    for key in ['all_three_configs_exact_readback', 'all_answer_free_qwen_chat_and_completion_budgets_measured',
                'all_native_budgets_include_4096_output_reserve', 'all_previous_folds_preserved',
                'no_published_question_overlap', 'first_nonaccept_query_holds_preserved', 'all_budget_proof_identities_equal']:
        if not new[key]: raise ValueError('NI release gate missing: ' + key)
    old_plan = json.loads((PARENT / 'publication-plan.json').read_text())
    mutable = {'README.md', 'NOTICE.md', 'PUBLICATION.json'}
    for item in old_plan['manifest']:
        if item['path'] in mutable: continue
        link_checked(PARENT_DEST / item['path'], DEST / item['path'], item['sha256'])
    for item in new['files']:
        link_checked(PUBLIC / item['path'], DEST / item['path'], item['sha256'])
    db = sqlite3.connect(STATE / 'public-index.sqlite')
    db.execute('ATTACH DATABASE ? AS previous', (str(PARENT_INDEX / 'public-index.sqlite'),))
    db.execute('ATTACH DATABASE ? AS incoming', (str(NI / 'public-index.sqlite'),))
    db.execute('CREATE TABLE records AS SELECT * FROM previous.records')
    db.execute('INSERT INTO records SELECT * FROM incoming.records')
    db.execute('CREATE UNIQUE INDEX record_identity ON records(record_id)'); db.commit()
    for key in ['query_hash', 'family_id']:
        if db.execute(f'SELECT 1 FROM records GROUP BY {key} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('cumulative split leakage')
    unique = {key: db.execute(f'SELECT COUNT(DISTINCT {key}) FROM records').fetchone()[0]
              for key in ['record_id', 'query_hash', 'family_id']}; db.close()
    counts = Counter(old['counts']); counts.update(new['counts'])
    if unique['record_id'] != counts['accepted']:
        raise ValueError('cumulative count mismatch')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    summary = {'counts': dict(counts), 'unique_counts': unique, 'files': old['files'] + new['files'],
        'wave': 'cumulative-0004', 'prior_release': prior['final_head'], 'library_commit': code,
        'prior_rows_preserved_byte_for_byte': True, 'new_rows': new['counts']['accepted'],
        'new_wave': {k: v for k, v in new.items() if k != 'files'},
        'accepted_episode_target': 10000000, 'goal_complete': False, 'model_weights_trained': False,
        'model_efficacy_measured': False, 'status': 'LOCAL_VERIFIED_NOT_YET_PUBLISHED'}
    base.save(STATE / 'summary.json', summary)
    reports = DEST / 'reports/cumulative-0004'; reports.mkdir(parents=True, exist_ok=True)
    base.save(reports / 'summary.json', {k: v for k, v in summary.items() if k != 'files'})
    base.save(reports / 'data-files.json', summary['files'])
    shutil.copyfile(PARENT_DEST / 'PUBLICATION.json', reports / 'prior-publication.json')
    for source, name in [
        (NI / 'ppl-score-binding.json', 'ni-ppl-score-binding.json'),
        (NI / 'release-scope.json', 'ni-release-scope.json'),
        (NI / 'additional-quality-holds-v1.json', 'ni-additional-quality-holds.json'),
        (NI / 'semantic-audit-v1/summary.json', 'ni-first-winner-diagnosis.json'),
        (SOURCE / 'semantic-audit-v1/summary.json', 'ni-first-source-diagnosis.json'),
        (NI / 'final-scope-policy-v2.json', 'ni-final-scope-policy-v2.json'),
        (NI / 'final-scope-readback-v2.json', 'ni-final-scope-readback-v2.json'),
        (NI_SCOPE, 'ni-reference-semantics-policy.json'),
        (FIRST / 'ni-reference-semantics-tests-v1.log', 'ni-reference-semantics-tests-v1.log'),
        (FIRST / 'ni-release-tests-v1.log', 'ni-release-tests-v1.log')]:
        shutil.copyfile(source, reports / name)
    active_tasks = set(scope['by_task'])
    source_files = [item for item in json.loads((SOURCE / 'build-source-lock.json').read_text())['files']
                    if Path(item['path']).name in active_tasks]
    base.save(reports / 'ni-source-attribution.json', {
        'repo': 'allenai/natural-instructions',
        'official_test_tasks_excluded': True,
        'files': [{k: v for k, v in item.items() if k != 'local'} for item in source_files],
        'license_rule': 'Per-task admitted licenses only; never infer every underlying source is licensed from the repository license.'})
    notice = (PARENT_DEST / 'NOTICE.md').read_text() + '''

## Natural Instructions contextual-memory addition

Source: https://github.com/allenai/natural-instructions/tree/55a365637381ce7f3748fa2eac7aef1a113bbb82 . Only eligible official training tasks with explicit MIT, Apache-2.0, CC-BY-4.0 or CC-BY-SA-4.0 source licenses were staged; unknown licenses and official test tasks are excluded. See reports/cumulative-0004/ni-source-attribution.json and per-record source_refs_json for upstream dataset names, URLs, pinned task files, hashes, references and actual license. CC-BY-SA terms continue to apply to those adapted records. Transformations retain task definitions, select different-input public worked examples, package the full target input as causal front/back memory, add skill instructions, select among unchanged reference answers with native v26 PPL, and produce assistant-only training labels. No new answer generation or source-code execution.
'''
    (DEST / 'NOTICE.md').write_text(notice)
    card = make_card(counts, unique, code, new['source_audit'], new['winner_audit'])
    (DEST / 'README.md').write_text(card); DatasetCard(card).validate()
    report = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>v26 最低 PPL 與 NI 上下文資料更新</title>
<style>body{{font:17px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>v26 最低 PPL 標準答案與上下文資料擴充</h1>
<p>目前累計 {counts['accepted']:,} 筆，其中本次新增 {new['counts']['accepted']:,} 筆。不同正規化首問 {unique['query_hash']:,} 個、保守題目家族 {unique['family_id']:,} 個。目標一千萬筆尚未完成。</p>
<p>同一完整輸入的所有合格答案由凍結 v26 實際評分，再選最低 assistant-token PPL。已判定有問題的答案、不同輸入的大小寫或空白視圖、示範題與目標重疊均持續隔離。PPL 低代表模型較容易預測，不能證明答案正確。</p>
<p>NI 的定義、不同輸入示範與目標內容在答案產生前可見。目標答案不放入 memory。這批以短輸出和上下文理解為主；新增題數與有效受監督 tokens 分開呈現，沒有把大量背景 tokens 當作答案訓練量。某些上游 output 列表儲存的是多個必要答案項目，不能選最低 PPL 單項當成完整答案；相關未釐清任務、缺乏目標證據的事實問答與未確認原始訓練切分的評測來源均已隔離。較早的 NI 暫存包未發布。</p>
<p>來源首次抽查 80 筆與本次選取答案抽查分開記錄。首輪非通過全部保留隔離，矛盾判詞不改寫成通過，也不宣稱每筆資料都經過語義驗證。未訓練模型、未量測新模型效能，未宣稱追平大型模型。</p>
<pre>{html.escape(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False, indent=2))}</pre></html>'''
    (reports / 'report.html').write_text(report)
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': base.file_sha(p)}
                for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'PUBLICATION.json']
    base.save(STATE / 'publication-plan.json', {'parent_commit': prior['final_head'], 'library_commit': code,
        'manifest': manifest, 'accepted': counts['accepted'],
        'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26',
        'goal_phase': 'v26_and_natural_instructions_published_continuing_to_10m',
        'commit_message': 'Append contextual Natural Instructions with frozen-v26 minimum-PPL references and full format proofs'})
    print(json.dumps({'accepted': counts['accepted'], 'new_rows': new['counts']['accepted'], 'files': len(manifest)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'publish', 'verify']); args = p.parse_args()
    with (FIRST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.STATE = STATE; transport.DEST = DEST; transport.FIRST = FIRST
        if args.phase == 'prepare': prepare()
        elif args.phase == 'publish': transport.publish()
        else: transport.verify()
