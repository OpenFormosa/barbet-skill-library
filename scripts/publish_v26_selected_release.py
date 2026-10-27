"""Publish the fully verified cumulative v26-ranked replacement and read it back."""
import argparse
import fcntl
import html
import json
import shutil
import subprocess
from scripts import build_memory_sft_v2 as base
from scripts import publish_memory_cumulative_v2 as transport
from scripts.compact_v26_selected_release import STATE,DEST,SELECTED
from scripts.prepare_v26_answer_candidates import STATE as SCORE
from scripts.build_memory_sft_wave2 import STATE as SECOND

FIRST=base.WORK/'reports/memory-harness-v2'
DATASET=transport.DATASET


def configure():
    transport.STATE=STATE;transport.DEST=DEST;transport.FIRST=FIRST


def prepare():
    if (STATE/'publication-data-commit.json').exists():raise ValueError('committed payload is immutable')
    s=json.loads((STATE/'summary.json').read_text());c=s['counts'];u=s['unique_counts']
    if not all(s[k] for k in ['all_retained_records_and_token_arrays_equal','all_budget_proofs_preserved','all_known_first_wave_tool_holds_absent','all_terminal_completion_holds_absent']):raise ValueError('missing data release gates')
    for f in s['files']:
        if base.file_sha(DEST/f['path'])!=f['sha256']:raise ValueError('compacted payload changed')
    parent=json.loads((FIRST/'publication-final-commit.json').read_text())['final_head']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.REPO,text=True).strip()
    reports=DEST/'reports/v26-selected';reports.mkdir(parents=True,exist_ok=True)
    base.save(reports/'summary.json',{k:v for k,v in s.items() if k!='files'})
    base.save(reports/'data-files.json',s['files'])
    for src,name in [(SCORE/'candidate-index-summary.json','candidate-index.json'),
                     (SCORE/'canonical-single-v3-validation.json','canonical-ppl-validation.json'),
                     (SCORE/'live-calibration-diagnostic-v3.json','retained-numerical-diagnostic.json'),
                     (SCORE/'episode-identity-v1/summary.json','episode-identity-summary.json'),
                     (SCORE/'episode-identity-v1/policy.json','episode-identity-policy.json'),
                     (SELECTED/'dedup-summary.json','selected-before-token-budget.json'),
                     (SELECTED/'ppl-score-binding.json','ppl-score-binding.json'),
                     (SELECTED/'selection-policy.json','selection-policy.json')]:shutil.copyfile(src,reports/name)
    shutil.copyfile(SELECTED/'terminal-completion-audit-v1/summary.json',reports/'terminal-completion-quality-scan.json')
    shutil.copyfile(FIRST/'terminal-completion-quality-tests-v1.log',reports/'terminal-completion-quality-tests-v1.log')
    shutil.copyfile(FIRST/'winner-audit-resume-tests-v1.log',reports/'winner-audit-resume-tests-v1.log')
    audit=json.loads((SELECTED/'semantic-audit-v1/summary.json').read_text())
    scopes={}
    for label,state in [('wave-0001',FIRST),('wave-0002',SECOND),('v26-winner-diagnosis',SELECTED)]:
        a=json.loads((state/'semantic-audit-v1/summary.json').read_text())
        scopes[label]={'samples':len(a['results']),'verdicts':a['counts'],'scope':a['scope'],
            'valid_semantic_verdicts':a.get('valid_semantic_verdicts',sum(a['counts'].get(v,0) for v in ['ACCEPT','REVISE','REJECT'])),
            'orchestration_holds':a.get('orchestration_holds',0),
            'first_nonaccept_record_ids':a['excluded_record_ids'],'all_rows_semantically_verified':False}
    base.save(reports/'semantic-audit-scopes.json',scopes)
    for name in ['v26-answer-selection-tests-v3.log','v26-candidate-adapter-tests.log','v26-episode-identity-tests.log',
                 'v26-finalization-tests-v2.log','v26-compaction-tests.log','cumulative-publisher-tests-v2.log','public-readback-v2-tests.log',
                 'context-agent-v2-tests.log','contract-tests.log','dataset-adapter-tests.log','existing-inference-tests.log']:
        shutil.copyfile(FIRST/name,reports/name)
    recipe=DEST/'recipe';recipe.mkdir(exist_ok=True)
    for pattern in ['memory_sft_v2*.json','openorca_canonical_teacher_styles_v1.json']:
        for path in (base.REPO/'configs').glob(pattern):shutil.copyfile(path,recipe/path.name)
    base.save(recipe/'code.json',{'repo':'OpenFormosa/barbet-skill-library','commit':code,'model_weights_trained':False,
        'model':'voidful/barbet-skill','model_revision':'67131067ff1825179d5e38e91d499159669e9c66',
        'scoring':'Frozen native conditional assistant-token PPL; complete unpadded sequences.'})
    source_files=json.loads((SELECTED/'build-source-lock.json').read_text())['files']
    base.save(DEST/'provenance/source-files.json',[{k:v for k,v in f.items() if k!='local'} for f in source_files])
    first_public=base.WORK/'exports/memory-harness-v2-public'
    notice=(first_public/'NOTICE.md').read_text()+'\nAdditional wave-2 sources retained in this cumulative release:\n\n'
    revisions={f['repo']:f['revision'] for f in source_files}
    for sid in ['Open-Orca/OpenOrca','Agent-Ark/Toucan-1.5M']:
        notice+=f'- [{sid}](https://huggingface.co/datasets/{sid}/tree/{revisions[sid]}) — `{revisions[sid]}`.\n'
        src=base.RAW/'scale10m-reference'/(sid.replace('/','--')+'.README.md')
        dest=DEST/'provenance/upstream-notices'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    if (first_public/'provenance/upstream-notices').exists():
        shutil.copytree(first_public/'provenance/upstream-notices',DEST/'provenance/upstream-notices',dirs_exist_ok=True)
    (DEST/'NOTICE.md').write_text(notice)
    head=['---','pretty_name: Barbet Memory, Skills and Tool SFT v2','license: other','license_name: mixed-upstream-licenses',
        f'license_link: https://huggingface.co/datasets/{DATASET}/blob/main/NOTICE.md','language:','- en','- zh','task_categories:','- text-generation',
        'tags:','- tool-use','- retrieval-augmented-generation','- long-context','configs:']
    for config in ['records','tokenized','minicpm5_tokenized']:
        head+=[f'- config_name: {config}']
        if config=='records':head+=['  default: true']
        head+=['  data_files:','  - split: train',f'    path: data/wave-*/{config}/train-*.parquet','  - split: validation',f'    path: data/wave-*/{config}/validation-*.parquet']
    head+=['---','']
    md=f'''# Barbet Memory, Skills and Tool SFT v2

The active cumulative release contains **{c['accepted']:,} accepted new episodes**, **{u['query_hash']:,} distinct normalized first-user queries**, and **{u['family_id']:,} conservative split families**. Train: {c['split:train']:,}; validation: {c['split:validation']:,}. Legacy active rows: **0**. These are different scale measures, not interchangeable claims of independent natural problems.

The target remains **10,000,000 accepted new episodes**; this release is **{c['accepted']/100000:.2f}%** of that target. It is an intermediate publication, not completion of the expansion goal.

## Minimum v26 answer PPL

Competing eligible answers to an identical complete input are ranked by frozen `voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66`. This release contains **{c.get('selection:minimum_v26_conditional_answer_ppl',0):,}** selected records with measured minimum answer PPL and **{c.get('selection:only_eligible_answer',0):,}** records with only one eligible answer, for which no ranking score is claimed.

PPL is exp(mean negative log probability over supervised assistant tokens, including EOS). Prompt, memory, tool observations, unsupervised history and padding are excluded from the loss denominator. The full causal input remains visible. Scoring uses native v26 formatting, original mixed-precision weights, batch size one, FlashAttention, sequence-aligned 128-token vocabulary projections, FP32 cross entropy and FP64 summation. No context truncation or model training occurs. Per-record `source_refs_json.answer_selection` binds the score, answer-token count, input hash and protocol.

An earlier padded-batch calibration exceeded its preset tolerance and remains recorded. Canonical unbatched controls repeat exactly and match the same-kernel published loss within 2.5e-8 mean NLL. This is numerical consistency, not accuracy or an efficacy benchmark. **Lower PPL is not proof that an answer is correct.** Source, chronology, schema, first-hold and budget gates remain separate.

Exact dialogue-prefix training views do not add episodes. Same static tasks with conflicting observation conditions are held. Known generic OpenOrca teacher-style prompts are canonicalized using a fixed whitelist; task-specific constraints remain intact. The minimum is recomputed from saved scores after these exclusions, with no first-source fallback. Earlier source-order compaction was superseded before publication.

## Training views and measured scope

| View | Physical tokens | Assistant-supervised tokens |
|---|---:|---:|
| Pangolin | {c['pangolin_tokens']:,} | {c['pangolin_supervised_tokens']:,} |
| MiniCPM5-2B | {c['minicpm_tokens']:,} | {c['minicpm_supervised_tokens']:,} |

`records` contains complete messages, tools, memory, source provenance, full rendered texts and target spans. Token arrays store unshifted assistant-only labels with -100 elsewhere. Apply the causal shift once in training. The two tokenized views and memory positions do not count as extra episodes. All full arrays are read back; exact compaction changes file boundaries only.

Use the library's `prepare_dataset_request` adapter for answer-free Chat or exact-renderer Completion inputs. Do not feed stored training labels as inference requests. Both served-Qwen request modes and native MiniCPM/Pangolin budgets are measured for every row, with a 4,096-token output reserve and no truncation. These are format/budget checks, not Barbet vLLM backend certification.

Most imported answers retain their explicit upstream verification tier; they have not all been independently semantically verified. The new winner diagnostic covers {len(audit['results'])} planned samples, with outcome counts `{json.dumps(audit['counts'],sort_keys=True)}` and {audit.get('valid_semantic_verdicts',0)} complete semantic verdicts. Incomplete model output and first-attempt orchestration failures remain HOLD, without replaying those requests or pretending they are model verdicts. First non-accepts and their same-query variants are excluded. This purposive diagnostic is not a whole-corpus accuracy estimate. Historical `recorded_error_followup` only detects error-related words in tool text and does not establish a successful repair. Imported tool observations are recorded or simulated upstream as labeled, not locally re-executed source programs.

A new source-specific guard excludes {c.get('excluded:unresolved_terminal_error_before_completion_claim',0):,} compiled Nemotron Terminal trajectories whose final JSON declares `task_complete: true` while the last `run_terminal` observation contains an explicit Python exception line. The trigger was a first-review case that claimed JSON validation had succeeded despite a recorded `SyntaxError`. This is a conservative unresolved-validation hold; it neither proves that every printed error invalidates a task nor proves that older errors were resolved by a later observation. Diagnostic errors without a completion claim remain eligible under the other gates.

The library provides causal memory, skill retrieval, registered typed callbacks and bounded verifier-driven refinement. No model weights were trained, and this release does not establish small-model parity with a 9B model. MiniCPM5-2B is the pinned native formatting/context reference; the exact private training mixture is not claimed reproduced.

## Provenance and history

The active tree contains only new v2 data. All {c['excluded:new_declared_argument_guard']:,} known wave-1 function-parameter holds are absent. The previous public state is preserved at `{parent}`. Source revisions, upstream licenses, tests, score bindings, diagnostic scopes and data hashes accompany the release. This is a mixed-license collection; follow [NOTICE.md](NOTICE.md) and each row's source license.

Library code: https://github.com/OpenFormosa/barbet-skill-library/tree/{code}

Anonymous SHA256 readback is recorded in `PUBLICATION.json` only after every file has been fetched from the pinned public data commit.
'''
    (DEST/'README.md').write_text('\n'.join(head)+md)
    report=f'''<h1>v26 PPL 重選後的累積資料</h1><p>合格新軌跡 {c['accepted']:,}，不同首題 {u['query_hash']:,}，家族 {u['family_id']:,}。一千萬筆目標完成 {c['accepted']/100000:.2f}%，仍在擴充。</p><p>同一完整條件的合格答案，以固定 v26 的答案條件 PPL 最低者作為標準。題目、技能、記憶、觀察不計入 loss 分母。批次數值差異的原始失敗保留，正式採逐筆完整脈絡評分；不訓練權重。</p><p>同一對話的前綴片段不另計資料量；同一任務但觀察條件不同者隔離。最低 PPL 在品質排除後重新核對，不回退為第一個來源。PPL 衡量可預測性，不保證正確；首次負面審查及來源品質規則仍有效。</p><p>所有文字、三種資料視圖、labels 與推論預算均完整讀回檢查。現役舊版資料為零，舊版提交仍可重現。新抽查 {len(audit['results'])} 筆，結果 {html.escape(json.dumps(audit['counts']))}，不能當作全體正確率。</p><p>尚未訓練模型，沒有小模型已等同 9B 的結論。後續仍依外置記憶、基本邏輯、RAG、工具、程式與真實回饋的原則擴充。</p>'''
    (reports/'report.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v26 PPL 重選資料報告</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:0 24px}</style>'+report+'</html>')
    manifest=[{'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':base.file_sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='PUBLICATION.json']
    base.save(STATE/'publication-plan.json',{'parent_commit':parent,'library_commit':code,'manifest':manifest,'accepted':c['accepted'],
        'answer_selection':'minimum eligible conditional answer PPL under frozen native Barbet v26',
        'commit_message':'Publish cumulative new-only SFT with minimum-v26-PPL answer selection and exact budget proofs',
        'goal_phase':'v26_selected_release_published_continuing_toward_10m'})
    print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'accepted':c['accepted']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','publish','verify']);args=p.parse_args();configure()
    with (FIRST/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='prepare':prepare()
        elif args.phase=='publish':transport.publish()
        else:transport.verify()
