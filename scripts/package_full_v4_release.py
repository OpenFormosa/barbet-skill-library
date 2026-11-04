"""Package only a fully processed, anonymously verified v4 corpus."""
from pathlib import Path
from collections import Counter
import html
import json
import subprocess
import yaml
from barbet_pipeline.common import read_json, save_new, sha_file
from barbet_pipeline.full_corpus import STATE, REPO
from barbet_pipeline.full_publication import BRANCH, MAIN, BASE
from barbet_pipeline.full_release import collect_complete_release, configuration_specs, exposure_policy
from barbet_pipeline.gpu_release import collect_complete_gpu_release
from barbet_pipeline.gpu_report import gpu_report_html


def main():
    summary, statistics, data = collect_complete_release(STATE)
    gpu_summary, gpu_files = collect_complete_gpu_release(STATE, data)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Commit the implementation before packaging')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    output = STATE / 'release-package'
    output.mkdir(exist_ok=False)
    metadata = list(gpu_files)
    def public(value):
        if isinstance(value, dict): return {k: public(v) for k, v in value.items()}
        if isinstance(value, list): return [public(v) for v in value]
        if isinstance(value, str) and value.startswith('/root/'):
            return 'local-path-omitted/' + Path(value).name
        return value
    def add(path, remote):
        metadata.append({'local': str(path), 'remote': remote, 'bytes': path.stat().st_size, 'sha256': sha_file(path)})
    def write(name, value):
        path = output / 'metadata' / name
        save_new(path, public(value)); add(path, 'full-v4/metadata/' + name)
    for name in ['processing-summary', 'statistics-summary', 'split-preflight', 'live-tokenizer-controls',
                 'actual-source-integration-controls-v2', 'transport-server-stopped',
                 'resource-amendment-before-full-run', 'initial-attempt-closure',
                 'ranked-input-amendment', 'ranked-input-controls', 'source-visibility-summary',
                 'source-visibility-tie-amendment', 'actual-retrieval-tie-control',
                 'source-visibility-cutoff-amendment', 'actual-retrieval-cutoff-control',
                 'graph-visibility-summary','actual-graph-visibility-controls','actual-changes-summary',
                 'bounded-readback-interruption','bounded-readback-amendment','bounded-readback-control',
                 'ranged-readback-interruption','ranged-readback-amendment','ranged-readback-control']:
        write(name + '.json', read_json(STATE / (name + '.json')))
    write('processing-plan.json', read_json(STATE / 'plan.json'))
    write('exposure-policy.json', exposure_policy(statistics))
    write('source-processing-receipts.json', [read_json(p) for p in sorted((STATE / 'receipts').glob('*.json'))])
    write('anonymous-data-readback.json', [read_json(p) for p in sorted((STATE / 'publication/readback').glob('*.json'))])
    write('source-visible-prefix-receipts.json', [read_json(p) for p in sorted((STATE / 'source-visibility').glob('*.json'))])
    write('graph-visible-query-receipts.json', [read_json(p) for p in sorted((STATE / 'graph-visibility').glob('*.json'))])
    write('actual-text-change-receipts.json', [read_json(p) for p in sorted((STATE / 'actual-changes').glob('*.json'))])
    gpu = STATE / 'gpu-quality'
    for name in ['plan', 'full-score-plan', 'numerical-controls', 'anchors', 'anchor-scores',
                 'context-probe-plan', 'context-probe-scores', 'context-probe-summary', 'context-probe-controls',
                 'provenance-index', 'provenance-index-restart', 'complete-summary', 'curriculum-policy',
                 'measurement-files', 'launch', 'workers-exited']:
        write('gpu-' + name + '.json', read_json(gpu / (name + '.json')))
    write('gpu-score-receipts.json', [read_json(p) for p in sorted((gpu / 'score-receipts').glob('*.json'))])
    write('gpu-aggregation-receipts.json', [read_json(p) for p in sorted((gpu / 'aggregation-receipts').glob('*.json'))])
    write('gpu-worker-completion.json', [read_json(gpu / 'workers' / (str(i) + '.json')) for i in range(8)])
    for name in ['numerical-controls.log', 'context-probe-controls-rerun.log', 'release-tests.log']:
        add(gpu / name, 'full-v4/metadata/gpu-' + name)
    for name in ['semantic_v3_sources.lock.json', 'semantic_v3_environment.lock.json',
                 'semantic_v3_model_manifest.json', 'taiwan-civic-policy-v1.json',
                 'value-alignment-policy-v1.json', 'source_tool_contract_holds_v1.json']:
        write(name, read_json(REPO / 'configs' / name))
    for name in ['unit-tests-complete-v5.log', 'request-adapter-tests.log', 'visibility-and-release-tests.log',
                 'source-visibility-v3-controls.log','final-exposure-policy-tests.log','graph-visibility-tests.log',
                 'final-accounting-tests.log','anchored-config-tests.log','bounded-readback-tests.log','ranged-readback-tests.log']:
        path = STATE / name
        if not path.exists(): raise FileNotFoundError(path)
        add(path, 'full-v4/metadata/' + name)
    configurations = configuration_specs(data)
    configurations.append({'config_name': 'full_v4_gpu_measurements', 'data_files': [
        {'split': split, 'path': f'full-v4/gpu-quality/measurements/*/{split}-*.parquet'}
        for split in ['train', 'validation']]})
    # Existing semantic contrasts are additional authored roots, not another copy
    # of the SQuAD material that v4 already includes.
    for kind in ['records', 'tokenized']:
        configurations.append({'config_name': 'semantic_v3_' + kind, 'data_files': [
            {'split': 'train', 'path': f'semantic-v3/pilot/{kind}/train-*.parquet'},
            {'split': 'validation', 'path': f'semantic-v3/pilot/{kind}/dev-*.parquet'}]})
    config_counts = []
    for status in ['verified', 'source_reference', 'reference_only']:
        for split in ['train', 'validation']:
            n = sum(f['rows'] for f in data if f['remote'].startswith(f'full-v4/{status}/records/{split}-'))
            config_counts.append({'status': status, 'split': split, 'rows': n})
    write('configuration-counts.json', config_counts)
    c = summary['counts']
    changes=read_json(STATE/'actual-changes-summary.json')['counts']
    rows = '\n'.join(f"| `{status}` | {next(r['rows'] for r in config_counts if r['status']==status and r['split']=='train'):,} | {next(r['rows'] for r in config_counts if r['status']==status and r['split']=='validation'):,} |"
                     for status in ['verified', 'source_reference', 'reference_only'])
    card = f'''# Barbet long-context SFT — full corpus reprocessing v4

**All {c['active_rows_processed']:,} active source records have been processed.** This release contains {c['exported']:,} fully re-exported records and {c['quarantined']:,} new quarantine dispositions across {summary['source_shards']:,} source shards. It is a complete corpus transformation, with no row sampling limit. It is not a new model training run or proof of capability improvement.

This branch preserves `main` at [{MAIN}](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/{MAIN}) and the earlier semantic release at [{BASE}](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/{BASE}). New v4 files are under `full-v4/`. A complete release requires `full-v4/release-manifest.json`, `full-v4/publication-complete.json`, and absence of `FULL_V4_STAGING_INCOMPLETE.txt`.

| Quality scope | Train records | Validation records |
|---|---:|---:|
{rows}

Each status has separate `full_v4_STATUS_records` and `full_v4_STATUS_tokenized` configurations. These are two representations of the **same** examples. The default `full_v4_dispositions` config has a `corpus` split accounting for all {c['active_rows_processed']:,} parents; it is not a training set. **`reference_only` is excluded from automatic training admission.** No mixture of the other statuses has been validated as a healthy training curriculum.

In full-v4 records, `root_task_id` conservatively preserves the inherited `family_id` split group. This may group multiple questions sharing an article or passage; it is not an independent-question count. Use `source` + `query_hash` / `parent_record_id` for individual question identity and keep the inherited family together for splitting. The authored v3 semantic config retains its separate root contract.

The compatibility fields `input_view_changed` and `skill_prefill_updated` mark the transformation/refresh path, which can be idempotent. A separate complete original-source census found **{changes['actual_serialized_text_changed']:,} serialized texts actually changed** and **{changes['actual_serialized_text_identical']:,} remained byte-identical**. Of the skill refreshes, {changes.get('skill_refresh_already_byte_identical',0):,} already matched the current library. These are reprocessed existing questions, not newly invented independent tasks. See [actual text-change census](full-v4/metadata/actual-changes-summary.json).

```python
from datasets import load_dataset
data = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "full_v4_verified_records",
    revision="{BRANCH}",  # pin the completed immutable revision for experiments
    split="train", streaming=True,
)
```

## What changed across the complete corpus

- Every active parent was reconstructed from typed messages, tools and loss spans. Its original Pangolin IDs and labels were recomputed and compared, before export. Saved output arrays were then independently read back and recomputed.
- {c.get('skill_prefills_updated',0):,} singleton-target skill prefixes were refreshed from the versioned library. Multi-candidate frozen-v26 inputs and their selected answers remain exactly unchanged. Earlier PPL is not presented as the score of a changed input; fresh frozen-v26 measurements for every exported row are provided separately and do not change target selection.
- {c.get('natural_views_replaced',0):,} SQuAD learner views use the previously accepted natural source-only answer renderer, with front/back evidence and optional skills. Source gold answers and source splits are retained. This is not new source content.
- All {c.get('decision_points',0):,} accepted assistant decisions have answer-free Chat and exact-renderer Completion prefix hashes and measured token budgets. Complete native Pangolin sequences use at most 131,072 tokens. Every inference prefix reserves 4,096 output tokens under both Pangolin and the actual Qwen served tokenizer; no truncation. The 4,096-token reserve is an inference request budget, not a claim that every reference answer can finish within it; no free-generation completion test was run. Qwen tokenization is only a transport check. **Barbet training uses Pangolin; no MiniCPM tokenizer is used.**
- {c.get('tool_reexecutions',0):,} finite-graph lookup calls were actually reexecuted from public state. A separate graph algorithm checked final answers. Source programs, arbitrary code and remote actions were not executed.
- Every finite-graph row additionally binds the evaluated operation, direction and explicit entity IDs to the actual visible question. Inline edge records must match the registered state and actual memory rendering exactly; tool observations were checked by reexecution. Display labels and current-world facts are not independently certified. [Complete graph-input audit](full-v4/metadata/graph-visibility-summary.json).
- All exported QA and Natural Instructions visible questions, definitions, examples, passages and actual retrieval observations were additionally reconstructed from pinned source documents. See the [complete source-visible prefix audit](full-v4/metadata/source-visibility-summary.json). This checks source fidelity beyond final-answer matching.
- 343 corpus program tests and 112 subtests passed, plus 15 bounded-transfer, range and resumable-publication controls. Historical fork tests emitted 34 Python deprecation warnings; these are disclosed in the test log.
- All pre-existing holds remain excluded. New holds have parent IDs and reasons. Root/query/source split checks found no conflicts in the complete active corpus. This does not assert semantic-neighbor decontamination against unavailable remote DEV outputs.

## Verification scope matters

`verified` combines explicit, recorded scopes. Finite graph answers have independent executable checks. SQuAD uses pinned gold/span/sufficiency labels. Hotpot uses pinned answers and supporting sentence IDs. Civic extracts and quotations are checked against attributed sources. The latter checks do not establish universal factual truth, legal applicability, or model learning. Inspect `verification_scope` and `semantic_correctness_independently_certified`; the boolean is true only for independently checked finite graph semantics.

`source_reference` preserves pinned Natural Instructions target/input references and previously accepted owner-authored normative cases. It is source fidelity, **not an independent correctness judgment**.

`reference_only` includes other math, coding, general instruction and tool targets without a suitable independent verifier. They receive the same full structural/token/inference-prefix processing, but their semantic correctness is not certified. They are provided separately for traceability and further research, not relabeled as accepted TRAIN merely because they are well formed.

## Exhaustive frozen-v26 GPU measurements

All **{gpu_summary['totals']['rows']:,} exported rows** were scored on the actual frozen `voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66`, using at most eight local H200 GPUs and no training updates or truncation. The `full_v4_gpu_measurements` config contains per-record assistant NLL, teacher-forced top-1 counts, a detached-readout gradient sketch norm, seven TRAIN semantic-prototype cosines and exact public provenance. Join by `record_id`; measurement rows are not additional training examples. All IDs match the full export exactly once.

Global supervised-token-weighted NLL is **{gpu_summary['totals']['token_weighted_nll']:.6f}**. Teacher-forced token prediction is not free-generation task success. Source verification statuses are unchanged by model scores. The gradient projection borrows from [LESSER](https://arxiv.org/abs/2610.03702); it measures the readout copy, not the full tied-weight or model gradient. No training benefit is inferred from cosine similarity.

An additional 15,430 candidate scores cover all 2,000 authored TRAIN roots, with independently verified public-prefix construction. Of discriminating first-action pairs, comparison is 0/287 and ordering 0/286 for preferring the correct answer in **both** conditions, despite low average NLL. These are conditional candidate preferences on TRAIN, not a held-out generation benchmark. Keep both valid semantic conditions; do not select only the easier one. Numerical, length-normalization and missing-evidence caveats are in the [GPU report](full-v4/gpu-report.html), [complete measurements](full-v4/metadata/gpu-complete-summary.json) and [measured curriculum policy](full-v4/metadata/gpu-curriculum-policy.json).

## Inference and loss contract

Use `barbet_pipeline.full_requests.prepare_full_request` from the [pinned implementation](https://github.com/OpenFormosa/barbet-skill-library/tree/{commit}). Select a measured `target_turn`; the request includes only preceding visible messages. No target answer, private grading metadata, execution receipt, or future observation is serialized. Portable tool feedback uses an explicit `user` continuation prefixed `TOOL_OBSERVATION:` to match the frozen renderer. Completion uses the exact Pangolin rendering with `add_special_tokens=False`.

Tokenized labels are **unshifted**, with `-100` on system/user/skills/documents/tool observations and unaccepted attempts; only accepted assistant spans and necessary EOS receive loss. The trainer must shift once and normalize across actual supervised tokens. Training arrays are not portable requests for a model using another tokenizer.

## Exposure and limitations

The complete export has {c['input_tokens']:,} Pangolin input tokens and {c['supervised_tokens']:,} supervised tokens. Read the [measured exposure policy](full-v4/metadata/exposure-policy.json) before designing a mixture. Most independently verified rows are graph operations; uniform sampling would preserve that imbalance. Source-gold fidelity, low PPL and a large row count do not establish teaching value. Reference-only data is not a substitute for independently verified general math, coding and natural-instruction coverage.

The inherited `semantic_v3_records` / `semantic_v3_tokenized` configs retain 6,286 TRAIN views / 2,000 roots and 629 DEV views / 200 roots with counterfactuals, invariance and faded binding. They are separately identified, not multiplied into millions of nominally independent tasks. Private calibration/TEST and private oracles remain unpublished. Do not append the inherited v3 natural-reading config to v4; those source parents are already represented.

This export is mixed length up to 128K, **not a completed 1M curriculum**. There is no new CPT/SFT/RL, no trained-model improvement, no proof of learnability, new-composition transfer or retention, and no MiniCPM/Qwen capability equivalence. Training-parent identity, trainer numerical parity and model evaluations remain separate evidence requirements. Dataset processing itself is complete.

## Values and licensing

Existing Taiwan sovereignty, people's self-determination and universal human-rights policies remain explicit, with factual and quotation fidelity. Existing civic records are reprocessed under their precise source or normative scope; no new political facts or demonstrated trained alignment are claimed. See [civic policy](full-v4/metadata/taiwan-civic-policy-v1.json) and [value policy](full-v4/metadata/value-alignment-policy-v1.json).

All individual source licenses and attributions are retained in each record. SQuAD derivatives retain CC BY-SA 4.0. New authored semantic cases retain Apache-2.0. This multi-source release is **not** relicensed as a single Apache-2.0 corpus. Old immutable versions remain available.

## Audit and reproducibility

[Full manifest](full-v4/release-manifest.json) · [All source receipts](full-v4/metadata/source-processing-receipts.json) · [Anonymous file readback](full-v4/metadata/anonymous-data-readback.json) · [Processing totals](full-v4/metadata/processing-summary.json) · [Exposure and lengths](full-v4/metadata/statistics-summary.json) · [Traditional Chinese report](full-v4/report.html)

All newly published data bytes were anonymously downloaded and SHA-256 checked before removing only the task-created local copy. Original sources, checkpoints and receipts were retained. The local transport server made no model-generation requests and has stopped; no training job was started.
'''
    front = {'language': ['zh','en'], 'license': 'other', 'license_name': 'per-source-licenses-preserved',
             'license_link': f'https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/blob/{BRANCH}/README.md#values-and-licensing',
             'task_categories': ['text-generation'], 'tags': ['barbet','skill','tool-use','audited-data'], 'configs': configurations}
    path = output / 'README.md'; path.write_text('---\n' + yaml.safe_dump(front,sort_keys=False,allow_unicode=True) + '---\n\n' + card)
    add(path,'README.md')
    src_rows=''.join('<tr><td>'+html.escape(k[10:])+'</td><td>'+f'{v:,}'+'</td></tr>'
                     for k,v in sorted(summary['by_source'].items()) if k.startswith('processed:'))
    status_rows=''.join(f'<tr><td>{r["status"]}</td><td>{r["split"]}</td><td>{r["rows"]:,}</td></tr>' for r in config_counts)
    report=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Barbet 全量資料重處理 v4</title><style>body{{max-width:1080px;margin:40px auto;padding:0 22px;font:17px/1.7 system-ui;color:#263732}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #bbb}}code{{overflow-wrap:anywhere}}a{{color:#176e58}}</style><h1>Barbet 全量資料重處理 v4</h1><p>2026-11-04。全部 {c['active_rows_processed']:,} 筆 active parent、{summary['source_shards']:,} 個來源分片已完成處理。輸出 {c['exported']:,} 筆，新增隔離 {c['quarantined']:,} 筆。沒有用抽樣結果代替全量驗證，也沒有啟動訓練。</p><h2>交付範圍</h2><table><tr><th>範圍</th><th>Split</th><th>筆數</th></tr>{status_rows}</table><p>verified 包含獨立圖運算驗證與來源 gold／引述核對，確切範圍逐筆保存；source_reference 是來源答案忠實性，不是獨立語義裁判。reference_only 未獨立確認答案語義，另行匯出，不自動納入訓練。預設 configuration 是完整處置索引，避免無意把所有來源等量混入 TRAIN。</p><h2>實際改動與驗證</h2><ul><li>{c.get('skill_prefills_updated',0):,} 筆經過 singleton skill 更新路徑，其中部分原本已符合最新版。全資料實際文字改變 {changes['actual_serialized_text_changed']:,} 筆，{changes['actual_serialized_text_identical']:,} 筆保持逐 byte 相同；多候選輸入及 frozen-v26 勝出答案保持原樣。</li><li>{c.get('natural_views_replaced',0):,} 筆自然閱讀改用來源限定的自然答案，保留前／後證據位置與來源 gold。</li><li>所有原始及輸出 Pangolin IDs／labels 逐筆重算；所有 {c.get('decision_points',0):,} 個 assistant prefix 通過 Chat／Completion 身分與實際 tokenizer 預算檢查。</li><li>{c.get('tool_reexecutions',0):,} 次可信有限圖工具實際回放；任意來源程式沒有在 host 執行。</li><li>輸出 {c['input_tokens']:,} input tokens、{c['supervised_tokens']:,} supervised tokens；不使用 MiniCPM tokenizer。</li><li>4,096-token 輸出保留是推論請求預算，沒有宣稱每個參考答案都能在此預算內自然完成。</li><li>所有新發布資料逐檔匿名 SHA-256 回讀；只有經證實遠端備份的本次輸出才釋放本地副本，原始資料與 checkpoint 保留。</li></ul><h2>全部來源對帳</h2><table><tr><th>來源</th><th>處理筆數</th></tr>{src_rows}</table><h2>不能由這次處理推論的事</h2><p>來源答案正確性、語義覆蓋、學生可學習性、遷移與能力保留仍是不同問題。有限圖操作仍占多數；不得直接把完整資料均勻混合，再用 loss 下降宣稱能力改善。數學／程式／自然指令的一部分只能維持 reference_only。沒有重新生成數百萬個 teacher 答案，沒有新模型分數，也沒有完成 1M 課程。</p><p>既有 6,286 TRAIN／629 DEV 的語義對照仍可單獨使用，沒有複製湊量。私有 calibration、TEST、oracle 未公開。舊 516 題遠端逐題輸出未取得，不能聲稱已完成其語義近鄰除汙。</p><p>台灣主權、人民自決、普世人權與來源忠實性規則保留；來源事實、歷史引述、作者規範立場分開。新課程是否能保留通用能力，仍須模型實驗回答。</p><p><a href="https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/{BRANCH}">完整 HF 分支</a> · <a href="https://github.com/OpenFormosa/barbet-skill-library/tree/{commit}">固定程式版本</a> · <a href="metadata/processing-summary.json">處理總帳</a> · <a href="metadata/statistics-summary.json">全部 token 曝光與長度分布</a></p></html>'''
    path=output/'report.html'; path.write_text(report); add(path,'full-v4/report.html')
    # The corpus report and GPU appendix have separate evidence scopes.
    report = report.replace('沒有新模型分數', '沒有新模型訓練增益')
    report = report.replace('</body>', '') if '</body>' in report else report
    report = report.replace('</html>', '<p><a href="gpu-report.html">全量 GPU 評分、語義弱項與文獻報告</a>：全部新版資料均已用凍結 v26 測量；這不是訓練後能力提升的證明。</p></html>')
    path.write_text(report)
    metadata[-1].update(bytes=path.stat().st_size, sha256=sha_file(path))
    gpu_html = gpu_report_html(gpu_summary, read_json(gpu / 'context-probe-summary.json'), read_json(gpu / 'numerical-controls.json'))
    path = output / 'gpu-report.html'; path.write_text(gpu_html); add(path, 'full-v4/gpu-report.html')
    all_items=data+metadata
    manifest={'schema_version':'barbet-full-release-v4','status':'FULL_CORPUS_DATA_PROCESSED_NOT_MODEL_TRAINING',
              'source_active_rows':c['active_rows_processed'],'exported_rows':c['exported'],'quarantined_rows':c['quarantined'],
              'source_shards':summary['source_shards'],'parent_main':MAIN,'inherited_semantic_release':BASE,'candidate_branch':BRANCH,
              'implementation_commit':commit,'training_ready':False,'all_sources_processed':True,'no_sample_limit':True,
              'reference_only_is_not_training_admission':True,'requires_staging_marker_absent':True,
              'all_exported_rows_gpu_measured':True,'gpu_measurement_rows':gpu_summary['totals']['rows'],
              'gpu_measurements_are_not_training_validation':True,
              'requires_publication_complete_receipt':True,'private_oracle_or_heldout_records_published':False,
              'files':[{'path':x['remote'],'sha256':x['sha256'],'bytes':x['bytes']} for x in all_items],
              'hash_scope':'new full-v4 data and metadata, excluding this manifest and later completion receipt'}
    path=output/'release-manifest.json';save_new(path,manifest);add(path,'full-v4/release-manifest.json')
    save_new(output/'upload-plan.json',{'items':metadata})
    print(json.dumps({'source_rows':c['active_rows_processed'],'data_files':len(data),'metadata_files':len(metadata),'implementation_commit':commit}))


if __name__=='__main__':main()
