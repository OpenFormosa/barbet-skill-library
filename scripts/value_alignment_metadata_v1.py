"""Current counts, transparent values evidence and filtered sampling sidecars."""
from collections import Counter
import html
import json
import shutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import DatasetCard
from scripts import publish_value_alignment_v1 as release


def metadata(plan):
    s=release.read(release.STATE/'summary.json');c=s['counts'];u=s['unique_counts'];a=release.read(release.STATE/'admission.json')
    d=release.DEST;folder=d/release.FOLDER;folder.mkdir(parents=True,exist_ok=True)
    removed={rid for j in plan['jobs'] for rid in j['removed_ids']}
    recipe=release.read(release.prior.DEST/'recipe/barbet-training-v2.json')
    old_sidecar=release.prior.DEST/recipe['refinement_evidence_sidecar'];t=pq.read_table(old_sidecar)
    retained=t.filter(pc.invert(pc.is_in(t['record_id'],value_set=pa.array(sorted(removed)))))
    relative='proofs/value-alignment-v1/refinement-recorded-evidence-v1.parquet';target=d/relative;target.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(retained,target,compression='zstd',compression_level=3)
    if not pq.read_table(target).equals(retained):raise ValueError('Refinement values changed')
    refinement={'previous':len(t),'removed':len(t)-len(retained),'retained':len(retained),'path':relative,'sha256':release.sha(target),'recorded_signals_not_execution_success_or_reward':True}
    release.save(folder/'current-refinement-evidence.json',refinement)
    civic_source=release.prior.DEST/'curriculum/taiwan-civic-v1.parquet';civic=pq.read_table(civic_source)
    civic=civic.filter(pc.invert(pc.is_in(civic['record_id'],value_set=pa.array(sorted(removed)))))
    civic_relative='curriculum/value-alignment-v1/civic-records.parquet';p=d/civic_relative;p.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(civic,p,compression='zstd')
    if len(civic)!=s['retained_civic_episodes'] or not pq.read_table(p).equals(civic):raise ValueError('Civic active index differs')
    civic_tokens=sum(civic['pangolin_supervised_tokens'].to_pylist())
    recipe.update(refinement_evidence_sidecar=relative,civic_record_id_sidecar=civic_relative,
        civic_raw_supervised_token_share=civic_tokens/c['pangolin_supervised_tokens'],value_alignment_policy='VALUE_ALIGNMENT_POLICY.json',
        value_alignment_report=release.FOLDER+'/report.html',value_alignment_gate='Sensitive inputs/targets require contextual acceptance or an exact source-bound contract; unknown and first nonaccept queries excluded',
        value_alignment_exhaustive_semantic_certification=False,value_alignment_efficacy_measured=False)
    recipe.pop('dataset_revision',None);release.save(d/'recipe/barbet-training-v2.json',recipe)
    shutil.copyfile(release.REPO/'configs/value-alignment-policy-v1.json',d/'VALUE_ALIGNMENT_POLICY.json')
    shutil.copyfile(release.prior.DEST/'NOTICE.md',d/'NOTICE.md')
    training=(release.prior.DEST/'TRAINING.md').read_text()
    training+='\n\nCurrent values admission is documented in VALUE_ALIGNMENT_POLICY.json and reports/value-alignment-v1/report.html. Use the CURRENT sidecar paths in recipe/barbet-training-v2.json; historical sidecars remain provenance and may contain held IDs. Quality/values eligibility precedes native-v26 PPL ranking. Screening and sampled audits do not establish all-row semantic alignment. Do not treat audit verdicts, policy cards or target construction traces as external factual RAG evidence. No training was performed by this release.\n'
    (d/'TRAINING.md').write_text(training)
    for name in ['summary.json','contract-summary.json','contract-policy.json','system-binding-summary.json']:
        release.save(folder/name,release.read(release.STATE/name))
    public_admission={k:v for k,v in a.items() if k!='query_hashes'}
    release.save(folder/'admission-summary.json',public_admission)
    release.save(folder/'all-source-scan-summary.json',release.read(release.STATE/'tool-registry-scan-v1/summary.json'))
    release.save(folder/'previous-message-scan-summary.json',release.read(release.STATE/'context-scan-v2/scan-summary.json'))
    release.save(folder/'controls-summary.json',{'passed':release.read(release.STATE/'controls.json')['passed'],'control_cases':12,'scope':'First live local-Qwen values-policy calibration; positive and negative cases; not student-model performance'})
    # Public evidence contains identities/verdicts, never private audit prompts,
    # control packets, grader policy injection or full source repair payloads.
    results=release.read(release.STATE/'semantic-summary.json')['results'];rows=[]
    for r in results:
        receipt=release.STATE/'semantic-audit-v1'/r['id']/'receipt.json';meta=release.read(receipt) if receipt.exists() else {}
        rows.append({'record_id':r['record_id'],'query_hash':r['query_hash'],'first_verdict':r['verdict'],'request_sha256':meta.get('request_sha256'),'review_model':meta.get('model'),'review_revision':meta.get('revision'),'scope':'values-policy contextual audit; diagnostic sample'})
    pq.write_table(pa.Table.from_pylist(rows),folder/'first-semantic-dispositions.parquet',compression='zstd')
    shutil.copyfile(release.STATE/'candidate-dispositions.parquet',folder/'candidate-dispositions.parquet')
    release.save(folder/'held-queries.json',{'query_hashes':a['query_hashes'],'first_negative_and_pending_queries_held':True,'not_all_held_rows_claimed_wrong':True})
    contracts=release.read(release.STATE/'contract-summary.json');audit_counts=', '.join(f'{k} {v}' for k,v in a['semantic_counts'].items())
    text=(release.prior.DEST/'README.md').read_text();front=text[:text.index('\n---',4)+4]
    sources='\n'.join(f'| {k[7:]} | {v:,} |' for k,v in sorted(c.items()) if k.startswith('source:'))
    buckets='\n'.join(f'| ≤ {bound:,} | {c.get("length_bucket:"+str(bound),0):,} |' for bound in release.prior.prior.BOUNDS)
    scored=sum(v for k,v in c.items() if k.startswith('selection:') and k!='selection:only_eligible_answer')
    card=f'''{front}

# Barbet memory, skills and harness SFT

**{c['accepted']:,} retained new episodes**, **{u['query_hash']:,} distinct normalized first queries**, **{u['family_id']:,} task families**. Train **{c['split:train']:,}**; validation **{c['split:validation']:,}**. Active legacy rows: **0**.

The training target is **Barbet**. Default `tokenized` contains existing Barbet/Pangolin `input_ids` and assistant-only `labels`; `records` contains messages, tools, causal memory and source provenance. No MiniCPM tokenizer is required. Historical auxiliary mini fields are provenance only. Tokenizer SHA-256: `f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176`. Use [TRAINING.md](TRAINING.md) and [the recipe](recipe/barbet-training-v2.json). Pin the verified revision in [PUBLICATION.json](PUBLICATION.json).

```python
from datasets import load_dataset
train = load_dataset("{release.DATASET}", "tokenized", split="train", streaming=True)
```

## Explicit values and admission

The dataset owner adopts **Taiwan sovereignty and the free democratic self-determination of Taiwan's people**, democracy, liberty, human rights, rule of law, equality and pluralism. This is a disclosed normative policy, not universal diplomatic consensus. No party loyalty is prescribed. Peaceful dissent and criticism of Taiwan's government remain protected. Chinese people and all other groups have equal human dignity. Accurate translations, quotations, historical accounts and attributed opposing views must not be distorted to fit a position. See [VALUE_ALIGNMENT_POLICY.json](VALUE_ALIGNMENT_POLICY.json).

This revision screens **10,321,901 parent episodes** across every source, including actual scenario instructions, conversations and tool registries. **{a['candidate_rows']:,}** rows match broad topic routing. Matching is not a violation. **241,552** routed identifier-query rows passed per-row recomputation and complete visible-question, target, evidence and read-only-tool checks. **920** civic article extracts passed exact original-source and quotation-boundary checks. These establish bounded query/extraction contracts, not current-world truth or endorsement of quoted source claims.

Twelve first live auditor controls passed, including Taiwan self-determination, Chinese people's dignity, criticism of Taiwan's government, faithful opposing quotations, and rejection of persecution and dehumanization. **{a['semantic_first_reviews']}** fixed diagnostic records received full-context local-Qwen reviews: **{audit_counts}**. These are diagnostic counts, not corpus accuracy estimates. Unreviewed sensitive material and first non-ACCEPT results are held query-wide. New holds remove **{s['new_held_rows']:,} episodes** across **{s['new_held_queries']:,} queries**, with no runner-up replacement. Cumulative documented correction holds: **{s['cumulative_removed_rows']:,} episodes / {s['cumulative_held_queries']:,} queries**. Quarantine does not assert that every removed answer is wrong. [Full audit report]({release.FOLDER}/report.html).

**No exhaustive semantic-alignment guarantee is claimed.** The lexical router can miss implicit bias, paraphrases and other languages; screen-negative rows are not individually semantically certified. Sampled auditors can make mistakes. Earlier first holds remain immutable, and this gate never replays a failed review to obtain approval.

## Standard answers and continuation

Quality and values eligibility precede answer selection. Among eligible alternatives to an identical complete input, the frozen lowest conditional supervised-answer PPL under native Barbet v26 remains the criterion. Assistant tokens and EOS are scored; prompts, memory, observations and padding are excluded. All competitors require finite native scores; ties use record IDs. There are **{scored:,}** retained episodes selected from scored alternatives and **{c.get('selection:only_eligible_answer',0):,}** unscored singletons. Low PPL proves neither truth nor values alignment. This revision changes no surviving target, score, split or token array.

Training text uses **barbet_portable_v1**, distinct from the checkpoint's native Jinja tool format. Keep the supplied IDs and unshifted labels; use `BarbetSFTCollator` and one causal shift. Do not apply another chat template or enable cross-example packing before testing attention and Mamba state boundaries. Matching inference uses `barbet_completion` or `ContextAgent(..., prompt_format="barbet_portable_v1")`. Frozen native PPL is not portable-template PPL. No weights were trained.

| Measure | Count |
|---|---:|
| Complete Barbet tokens | {c['pangolin_tokens']:,} |
| Supervised Barbet tokens | {c['pangolin_supervised_tokens']:,} |
| Assistant turns | {c['assistant_turns']:,} |
| Tool observations | {c['tool_observations']:,} |

Every row fits its smallest Barbet-token bucket and retains 4,096 output tokens within 131,072 total. No truncation or unrelated padding is used.

| Complete Barbet token upper bound | Episodes |
|---|---:|
{buckets}

| Source | Episodes |
|---|---:|
{sources}

## Memory, skills and sampling

The [skill library and harness](https://github.com/OpenFormosa/barbet-skill-library/tree/{plan['library_commit']}) support answer-free requests, bounded retrieval, explicitly registered tools and verifier-guided refinement. Context and observations are evidence, not authority. Exclude audit reports, source_refs, gold answers and construction traces from RAG indexes. All original causal masks and budget proofs remain bound to unchanged retained records.

There are **{len(civic)}** retained civic episodes. The original 933-row cohort comprised 920 law extracts, 12 authored normative cases and one dated attributed official quotation. Current sampling IDs are in `{civic_relative}`. Their raw supervised-token share is **{100*recipe['civic_raw_supervised_token_share']:.6f}%**. This small normative curriculum does not establish broad alignment; sampling weights remain uncalibrated and must be evaluated. Do not duplicate examples to claim independent data. Laws are quoted as provided versions, not assurances of present individual applicability.

The active refinement sidecar is `{relative}`, containing **{len(retained):,}** retained recorded-error/same-tool-retry tags. Those are upstream observations, not local execution, successful repair certification or an RL reward. Historical sidecars remain provenance and are not the current recipe target.

## Verification and limits

Changed Parquets preserve every retained column and full array exactly, then receive anonymous SHA-256 readback on immutable staging commits. Main copies those exact objects. Unchanged files inherit the prior verified-object chain; new metadata is independently downloaded and hashed. This is composed verification, not a fresh download of all historical bytes.

Structured tasks use a limited set of schemas across distinct source anchors. SFT graph-family splits can share other context facts and are not entity-disjoint evaluation. Source programs were not executed. Prior v26 short padded forward validation is wiring evidence only, not gradient, long-context, packing, alignment or capability evaluation. No small-model/9B equivalence or training improvement is established.

Per-source licenses and attribution remain in [NOTICE.md](NOTICE.md). Earlier [readiness reports](reports/v26-readiness-v1/report.html) and [civic reports](reports/taiwan-civic-v1/report.html) describe historical revisions; the values report and PUBLICATION.json describe this active subset.
'''
    DatasetCard(card).validate();(d/'README.md').write_text(card)
    status_rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>' for k,v in sorted(a['candidate_dispositions'].items()))
    report=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>價值對齊資料納入查核</title>
<style>body{{font:17px/1.8 system-ui;max-width:1080px;margin:40px auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>價值對齊資料納入查核</h1>
<p>依資料集所有者明示立場：台灣主權、人民自由民主自決、普世人權、法治、平等、多元。保護和平異議與對任何政府的批評，不預設政黨；中國人民與其他群體享有同等尊嚴。不得為符合立場竄改來源、引文、翻譯、歷史或法律文字。</p>
<h2>實際資料變更</h2><p>從 10,321,901 筆現役資料，新增整題隔離 {s['new_held_queries']:,} 個 query、{s['new_held_rows']:,} 筆 episode；保留 <strong>{c['accepted']:,}</strong> 筆新資料，train {c['split:train']:,}、validation {c['split:validation']:,}，標準化首問 {u['query_hash']:,}、家族 {u['family_id']:,}。沒有新增重複資料充數，沒有把次低 PPL 答案補上，也未修改任何保留答案、分割、分數或 tokens/labels。</p>
<h2>全量篩查與可證明的界線</h2><p>先逐筆掃描實際對話、來源情境指令，再掃描所有模型可見工具定義，合計 {a['candidate_rows']:,} 個敏感主題候選。來源名稱和 JSON 格式不構成豁免。逐筆重算 241,552 個 ID/計數查詢，核對可見問題、所有 assistant 文字、唯讀工具、完整觀察與原始有限紀錄；920 個法規摘錄另對原始 XML、法規名稱、條號與引用範圍核對。這些只證明有限查詢或忠實摘錄，不代表來源內容是當今世界真相，也不代表模型背書其價值。</p>
<table><tr><th>候選的初次處理</th><th>筆數</th></tr>{status_rows}</table>
<p>若同一 query 中仍有待審或不通過紀錄，其他正向紀錄也隨整題隔離；因此上述初次分類不等於最終保留筆數。所有既有第一次不通過紀錄繼續隔離。</p>
<h2>語境審核</h2><p>本機 Qwen/Qwen3.8-Flash-Next-FP8，revision 236dfdf285828023ca3bcd3f37366c58a3469b13；最多兩個工作槽。12 個正反控制全通過，測試包含台灣自決、反族群貶抑、批評台灣政府、忠實翻譯不同立場，以及拒絕迫害、竄改引文與未授權侵害。審核先固定樣本、完整輸入與嚴格 JSON schema，不截斷、不重試負面結果求通過，完整量測 served tokenizer 與預留輸出。</p>
<p>{a['semantic_first_reviews']} 筆固定樣本得到 {html.escape(audit_counts)}。來源分層採各來源最低 hash 8 筆，另含原對話篩查未命中的固定 256 筆，以及全部 13 個規範／官方引述案例。工具定義補掃後，實際仍未命中的樣本為 {a['screen_negative_sample']['remaining_screen_negative_after_registry_scan']} 筆，其結果 {html.escape(json.dumps(a['screen_negative_sample']['results']))}。這是診斷樣本，不是全語料錯誤率或價值合格率估計。負面模型判定也可能有誤；保留第一判定並隔離，不把它直接宣稱為已證實違規。</p>
<h2>價值與 PPL 的先後</h2><p>先符合品質與價值納入條件，再使用已凍結的原生 v26 最低 PPL 排名。PPL 只衡量模型熟悉程度，不證明正確、誠實或價值合格。保留的多答案 winner 維持原排名，不因這次審核改換答案。訓練仍使用 Barbet tokenizer、assistant-only labels、barbet_portable_v1；不需要 MiniCPM tokenizer。</p>
<h2>抽樣與效能限制</h2><p>現役公民資料 {len(civic)} 筆，監督 token 原始占比 {100*recipe['civic_raw_supervised_token_share']:.6f}%；少量規範案例仍不足以證明泛化。現役 refinement 索引 {len(retained):,} 筆，排除 {len(t)-len(retained):,} 個新隔離 ID；標記是上游錄製的錯誤／重試訊號，不是本機驗證修復或 RL reward。來源、任務、長度與 token 比例應在另外的受控訓練實驗中校準。</p>
<p><strong>沒有宣稱全部一千萬筆已逐筆完成語義價值認證。</strong> 詞彙篩查仍可能漏掉隱含偏見、改寫及其他語言；未命中的大多數資料只有既有品質檢查與本輪診斷抽樣。禁止把這份資料納入報告說成 v26 已對齊：本輪沒有訓練，也沒有測得模型行為或能力提升。</p>
<h2>發布驗證</h2><p>僅重寫受影響 Parquet 的保留子集，逐欄逐陣列相等；固定 staging commit 完整匿名 SHA-256 回讀後才複製到 main。未變更檔案沿用先前已驗證物件；新 metadata 獨立回讀。本輪是完整可追溯的組合驗證，不是假稱重新下載所有歷史 bytes。</p>
<p>凍結程式版本 {plan['library_commit']}。現役 counts 及首次處理索引在本資料夾，最終版本見根目錄 PUBLICATION.json。所有評審提示與完整稽核封包保留本機，不可作為 RAG 證據。</p></html>'''
    (folder/'report.html').write_text(report)
    release.save(folder/'metadata-binding.json',{'metadata_driver_sha256':release.sha(release.REPO/'scripts/value_alignment_metadata_v1.py'),'library_commit':plan['library_commit'],'admission_sha256':plan['admission_sha256'],'refinement':refinement,'civic_index_sha256':release.sha(d/civic_relative),'civic_raw_supervised_token_share':recipe['civic_raw_supervised_token_share'],'public_audit_packets_redacted':True})
