"""Render the main dataset card from the completed immutable v4 receipts."""
from pathlib import Path
import json
import yaml
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, sha_file

STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
OUTPUT = STATE / 'main-promotion-20261105'
V4 = '7103d1b1bc1fb4cda1298c084a66bda542d61151'
OLD_MAIN = 'b88f4ff7171bb6fb8c1b775ae21de814ea2123fb'
BACKUP = 'archive/main-before-v4-20261105'
ROOT = 'https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|'] +
        ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows])


def main():
    cpu = read_json(STATE / 'processing-summary.json')
    gpu = read_json(STATE / 'gpu-quality/complete-summary.json')
    stats = read_json(STATE / 'statistics-summary.json')
    exposure = read_json(STATE / 'training-exposure-evidence.json')
    probes = read_json(STATE / 'gpu-quality/context-probe-summary.json')
    changes = read_json(STATE / 'actual-changes-summary.json')['counts']
    controls = read_json(STATE / 'gpu-quality/numerical-controls.json')
    runtime = read_json(STATE / 'gpu-quality/runtime-environment.json')
    recovery = read_json(STATE / 'gpu-quality/parallel-recovery-20261105/completion.json')
    recovery_control = read_json(STATE / 'gpu-quality/parallel-recovery-controls-20261105/result.json')
    plan = read_json(STATE / 'gpu-quality/full-score-plan.json')
    counts = cpu['counts']; totals = gpu['totals']
    assert counts['exported'] == totals['rows'] == 10018961
    assert cpu['source_shards'] == 1504 and gpu['files'] == 1643
    original_card = (STATE / 'release-package/README.md').read_text()
    front = yaml.safe_load(original_card.split('---', 2)[1])
    original_configs = json.loads(json.dumps(front['configs']))
    front['license_link'] = ROOT + '/blob/main/README.md#values-and-licensing'
    scoped = read_json(STATE / 'release-package/metadata/configuration-counts.json')
    scope_table = table(['品質範圍', 'TRAIN 筆數', 'Validation 筆數', '驗證意義'], [
        [f'§{status}§',
         f"{next(r['rows'] for r in scoped if r['status']==status and r['split']=='train'):,}",
         f"{next(r['rows'] for r in scoped if r['status']==status and r['split']=='validation'):,}", description]
        for status, description in [
            ('verified', '依逐筆 scope 通過有限圖獨立計算，或來源 gold／證據／引述核對'),
            ('source_reference', '來源題目與參考答案忠實性；尚非獨立語義裁判'),
            ('reference_only', '完整結構、token 與前綴檢查；答案語義仍待獨立驗證')]])
    source_table = table(['來源識別', '處理', 'verified', 'source_reference', 'reference_only', '新增隔離'], [
        [f'§{source}§'] + [f"{cpu['by_source'].get(prefix+source,0):,}" for prefix in
            ['processed:', 'verified:', 'source_reference:', 'reference_only:', 'quarantined:']]
        for source in sorted(k[len('processed:'):] for k in cpu['by_source'] if k.startswith('processed:'))])
    length_rows = []
    previous = 0
    for bound, n in sorted(totals['length_bucket_upper_bounds'].items(), key=lambda p: int(p[0])):
        length_rows.append([f'{previous+1:,}–{int(bound):,}', f'{n:,}', f"{n/totals['rows']:.2%}"])
        previous = int(bound)
    length_table = table(['完整 Pangolin sequence tokens', '筆數', '占比'], length_rows)
    token_scope_table = table(['TRAIN 品質範圍', '筆數', 'Input tokens', 'Supervised tokens', '監督 token 占比'], [
        [f'§{name}§', f"{v['rows']:,}", f"{v['input_tokens']:,}", f"{v['supervised_tokens']:,}",
         f"{v['supervised_tokens']/exposure['totals']['supervised_tokens']:.2%}"]
        for name, v in exposure['by_quality_scope'].items()])
    domain_table = table(['繼承的 domain 標籤', 'TRAIN 筆數', 'Supervised tokens', '監督 token 占比'], [
        [f'§{name}§', f"{v['rows']:,}", f"{v['supervised_tokens']:,}",
         f"{v['supervised_tokens']/exposure['totals']['supervised_tokens']:.2%}"]
        for name, v in exposure['by_inherited_domain_tag'].items()])
    family_table = table(['語義能力', 'TRAIN roots', '有區分力的配對', '總 log-prob 兩側皆正確', '長度正規化兩側皆正確', '原題平均 NLL'], [
        [f'§{v["family"]}§', v['roots'], v['discriminating_first_action_pairs'],
         f"{v['sum_logprob_pairs_correct']}/{v['discriminating_first_action_pairs']}",
         f"{v['mean_logprob_pairs_correct']}/{v['discriminating_first_action_pairs']}",
         f"{v['full_nll']:.6f}"] for v in probes['families']])
    gains_table = table(['語義能力', 'Skill 增益（nats）', '證據增益（nats）', 'IFD-like ratio', '不變改寫的 NLL 變化'], [
        [f'§{v["family"]}§', f"{v['skill_gain_nats']:+.6f}", f"{v['evidence_gain_nats']:+.6f}",
         f"{v['ifd_like_ratio']:.6f}", f"{v['invariance_nll_change']:+.6f}"] for v in probes['families']])
    source_gpu_table = table(['品質範圍', '來源', 'Split', '筆數', 'Token 加權 NLL', '逐筆平均 NLL'], [
        [f'§{key.split("|")[0]}§', f'§{key.split("|")[1]}§', key.split('|')[2],
         f"{v['rows']:,}", f"{v['token_weighted_nll']:.6f}", f"{v['example_mean_nll']:.6f}"]
        for key, v in sorted(gpu['source_groups'].items())])
    histogram_rows = []
    previous = 0.
    for bound in [.125, .25, .5, 1., 2., 4., 8., 16., 32.]:
        key = str(int(bound)) if bound.is_integer() else str(bound)
        n = totals['nll_histogram_upper_bounds'].get(key, 0)
        histogram_rows.append([f'({previous:g}, {bound:g}]', f'{n:,}'])
        previous = bound
    histogram_rows.append(['>32', str(totals['nll_histogram_upper_bounds'].get('>32', 0))])
    histogram = table(['每列平均 assistant NLL 範圍', '筆數'], histogram_rows)
    prototype_table = table(['TRAIN 語義原型', '全資料平均 cosine'], [
        [f'§{k}§', f'{v:.6f}'] for k, v in totals['mean_semantic_family_cosines'].items()])
    config_table = table(['Configuration', 'Split', '用途'], [
        ['§full_v4_verified_records§ / §full_v4_verified_tokenized§', 'train / validation', '依明確 scope 核對的資料；仍須設計能力平衡與保留課程'],
        ['§full_v4_source_reference_records§ / §full_v4_source_reference_tokenized§', 'train / validation', '來源參照資料；須區分參照忠實性與答案正確性'],
        ['§full_v4_reference_only_records§ / §full_v4_reference_only_tokenized§', 'train / validation', '待進一步驗證的獨立資料池，不自動納入訓練'],
        ['§full_v4_dispositions§（預設）', 'corpus', '全部 active parents 的處置索引，包含隔離；不是 TRAIN'],
        ['§full_v4_gpu_measurements§', 'train / validation', '逐筆模型測量 sidecar，依 record_id 連接；不是新增訓練列'],
        ['§semantic_v3_records§ / §semantic_v3_tokenized§', 'train / validation', '2,000 TRAIN roots／200 DEV roots 的獨立語義課程']])
    record_fields = table(['Records 欄位', '內容與用途'], [
        ['§record_id§ / §parent_record_id§', '新匯出列與原始 parent 的精確身分；GPU sidecar 以 record_id 連接'],
        ['§family_id§ / §root_task_id§ / §query_hash§', '繼承的來源群組及問題身分；root_task_id 可能涵蓋同文章的多個問題'],
        ['§split§ / §domain§ / §source§ / §upstream_source§ / §license§', '來源、原始切分、繼承分類及逐筆授權'],
        ['§processing_status§ / §verification_scope§', '品質層級及實際驗證範圍'],
        ['§semantic_correctness_independently_certified§', '僅對獨立核對的有限圖語義設為 true；來源 gold 或引述核對不冒充獨立語義證明'],
        ['§messages_json§ / §tools_json§ / §memory_json§ / §skills_json§', 'Typed 訊息、工具、記憶與 skill 的 JSON 字串'],
        ['§text§ / §target_spans_json§ / §renderer_version§', '精確 Pangolin rendering、合格 assistant 監督區間與版本'],
        ['§checks_json§ / §execution_receipts_json§ / §source_refs_json§', '驗證證據、已發生的執行回執與來源定位'],
        ['§inference_decisions_json§', '每個合格 assistant 決策的 answer-free 前綴雜湊及實測推論預算'],
        ['§pangolin_tokens§ / §pangolin_supervised_tokens§ / §output_reserve_tokens§', '完整輸入、監督與推論輸出保留的 token 計量'],
        ['§target_selection_scope§ / §parent_record_sha256§', '既有答案選擇範圍及原 parent 的雜湊'],
        ['§input_view_changed§ / §natural_answer_rewritten§ / §skill_prefill_updated§', '處理路徑旗標；是否真的改 byte 要看完整文字差異盤點']])
    sample_path = Path('/root/barbet-context-sft-data/reports/semantic-curriculum-v3/pilot/records/train-0000.parquet')
    sample = next(pq.ParquetFile(sample_path).iter_batches(batch_size=1)).to_pylist()[0]
    public = json.loads(sample['public_episode_json'])
    messages = json.loads(sample['messages_json'])
    data_rows = json.loads(public['context'][0]['text'])['rows']
    sample_table = table(['實際項目 ID', 'score（Decimal 字串）'], [
        [f'§{r["id"]}§', f'§"{r["score"]}"§'] for r in data_rows])
    target = next(m for m in messages if m['role'] == 'assistant')
    example_call = json.dumps(target['tool_calls'][0]['function'], ensure_ascii=False, indent=2)
    fields_example = json.dumps({'record_id': 'fe4edfdd14caddf2ea8831a5323a014c32b61597818503d687a9b47ad4910299',
        'source': 'ConceptNet/5.7-original-lexical-english', 'processing_status': 'verified',
        'verification_scope': 'independent_finite_graph_query_and_tool_reexecution',
        'pangolin_tokens': 2310, 'pangolin_supervised_tokens': 34}, ensure_ascii=False, indent=2)
    weights_table = table(['模型檔案', 'SHA-256'], [[f'§{name}§', f'§{value}§'] for name, value in plan['model_files'].items()])
    text = f'''# Barbet long-context SFT v4：完整資料、GPU 評量與使用報告

本資料集的 **§main§ 已採用完整驗收的 v4**。本頁集中收錄資料規模、內容、來源、品質範圍、全量處理方式、凍結 Barbet v26 的 GPU 結果、語義弱項、使用契約、文獻與可重現證據。

全部 **{counts['active_rows_processed']:,} 筆 active parents**、{cpu['source_shards']:,} 個來源分片已處理完成。匯出 **{counts['exported']:,} 筆**，新增隔離 {counts['quarantined']:,} 筆；全部 {gpu['files']:,} 個 tokenized 分片均完成 GPU 評分、逐筆來源與唯一 ID 對帳。

**資料處理與發布已完成；§training_ready=false§。** 這是一個依驗證範圍分層的資料池。低 loss、來源 gold、大量列數及合法格式，各自都不能證明學生能學會、能遷移或能保住一般能力。

## 版本與主分支

| 項目 | 固定版本或位置 |
|---|---|
| 目前主要入口 | [main]({ROOT}/tree/main) |
| 完整驗收的 v4 資料版本 | [§{V4}§]({ROOT}/tree/{V4}) |
| 保留的 v4 分支 | [full-corpus-v4-20261104]({ROOT}/tree/full-corpus-v4-20261104) |
| 升版前 main | [§{OLD_MAIN}§]({ROOT}/tree/{OLD_MAIN}) |
| 升版前 main 備份 | [{BACKUP}]({ROOT}/tree/{BACKUP}) |
| 既有語義課程版本 | [§4b38655b336aa2d0c01f6f0047701e7eeec23ed6§]({ROOT}/tree/4b38655b336aa2d0c01f6f0047701e7eeec23ed6) |
| 本次主分支升版與 README 紀錄 | [main-promotion-20261105.json](full-v4/main-promotion-20261105.json) |

本次升版直接沿用已驗收的資料與測量檔，更新 README 並新增升版紀錄。原始 [release manifest](full-v4/release-manifest.json) 與 [publication receipt](full-v4/publication-complete.json) 保留原樣，記錄的是上表固定 v4 revision；其中舊 README 的雜湊對應該封存版本。新版 README 的雜湊與允許變動路徑另記於升版紀錄，不回寫歷史收據。

實驗請固定 §revision="{V4}"§，或固定本次 main 的完整 commit。§main§ 方便取得目前版本，本身不是不可變的實驗識別。

## 一、資料規模與品質分層

{scope_table}

- TRAIN：**{counts['split:train']:,}** 筆；validation：**{counts['split:validation']:,}** 筆。
- 完整 Pangolin input tokens：**{totals['input_tokens']:,}**。
- Assistant supervised tokens：**{totals['supervised_tokens']:,}**。
- 合格 assistant 決策前綴：**{counts['decision_points']:,}** 個。
- 原有隔離／拒絕資料：**{counts['prior_holds_excluded']:,}** 筆，維持排除。
- 本次新增隔離：5 筆超出完整推論預算而不截斷；1 筆保留既有自然閱讀視圖的 HOLD。

§verified§ 包含不同證據範圍，不能解讀成「每一列都由獨立世界知識 oracle 證明」。有限圖題有獨立可執行語義核對；SQuAD 使用固定來源 gold／span／資訊充分性標籤；HotpotQA 使用來源答案及支持句；公民資料則區分來源引述核對與作者規範立場。請逐筆閱讀 §verification_scope§。

§source_reference§ 保留 Natural Instructions 的原始題目、定義、例子、輸入與目標，以及先前合格的作者規範案例。它證明來源忠實性，尚非獨立答案正確性裁判。

§reference_only§ 包含尚無合適獨立驗證器的數學、程式、一般指令與工具答案。它們接受相同的完整結構、token、mask 與推論前綴檢查，但**不因低 NLL 或格式合法而自動加入訓練**。

## 二、所有來源的實際處理數量

{source_table}

來源識別不一律代表存在同名 HF repository。逐筆授權、原始 revision、文件／題號與來源定位保留於 §license§、§source_refs_json§ 及 [source receipts](full-v4/metadata/source-processing-receipts.json)。

這些是重新處理的既有問題與 episode；列數不等於獨立語義家族數，也不等於新增了一千萬個不同推理問題。

## 三、全量實際改動與驗證

1. **逐筆重建原資料。** 從 typed messages、工具與 target spans 重建原始文字，重新計算原有 Pangolin IDs／labels 並比較；匯出後再讀回、重新計算輸出 arrays。
2. **Skill 更新。** {counts['skill_prefills_updated']:,} 筆 singleton-target 走版本化 library 更新路徑。實際文字改變 **{changes['actual_serialized_text_changed']:,}** 筆，**{changes['actual_serialized_text_identical']:,}** 筆逐 byte 相同；其中 {changes['skill_refresh_already_byte_identical']:,} 筆原本已符合新版 skill。路徑旗標不是實際文字差異的證明。
3. **既有多候選答案保留。** 凍結 v26 已選定的多候選完整輸入與目標保持原樣；不把舊 PPL 當成修改後輸入的分數。新 GPU 測量另行提供，沒有重新選答案。
4. **自然閱讀。** {counts['natural_views_replaced']:,} 筆 SQuAD 視圖採已接受的自然、來源限定答案 renderer，保留來源 gold、原 split、前／後證據位置及可選 skill。沒有憑空增加來源內容。
5. **完整推論前綴。** 全部 {counts['decision_points']:,} 個合格 assistant 決策都驗證 answer-free Chat 與 exact-renderer Completion 的身分、雜湊及實測 token 預算。
6. **實際工具回放。** {counts['tool_reexecutions']:,} 次可信有限圖查詢由公開 state 重新執行，另以獨立圖演算法檢查答案；來源任意程式與遠端動作沒有在 host 執行。
7. **圖題語義連結。** 完整核對實際可見題目中的 operation、方向與明確 entity IDs，inline edges 必須對得上登錄 state 與實際 memory rendering。Display labels 及當代世界事實不因此被獨立認證。
8. **QA／NI 的可見輸入。** 所有匯出 SQuAD、HotpotQA 與 Natural Instructions 的題目、定義、例子、passage 與實際檢索觀察，都由固定來源重建。這比只比對最後答案多了一層來源忠實性檢查。
9. **Split 與隔離。** 全 active corpus 的 record ID、root 與 query hash 跨 split 沒有衝突；原有 HOLD 保留。未取得遠端舊 516 題逐題輸出，因此不聲稱完成其語義近鄰除汙。
10. **公開 bytes。** 全部新資料、測量與 metadata 逐檔匿名下載並比對 SHA-256。只釋放本次產生且已證實遠端備份的本地輸出；原始來源、checkpoint、歷史收據保留。

[文字實際差異](full-v4/metadata/actual-changes-summary.json) · [來源可見前綴](full-v4/metadata/source-visibility-summary.json) · [圖題語義核對](full-v4/metadata/graph-visibility-summary.json) · [完整處理總帳](full-v4/metadata/processing-summary.json)

## 四、長度分布與真正的長上下文範圍

下表是**完整序列的實際 Pangolin token 數**，包含 assistant target，各 bucket 互斥：

{length_table}

- 實際最長完整序列：**{stats['maximum_total_pangolin_tokens']:,} tokens**。
- 本 release 的完整 sequence 上限：131,072 tokens。
- 每個推論前綴保留 4,096 output tokens，並以 Pangolin 與實際 Qwen served tokenizer 檢查完整預算，沒有靜默截斷。
- 4,096 是請求的輸出保留，不代表每個參考答案都經自由生成證明能在此額度自然完成。
- Qwen tokenizer 僅作 transport／budget 相容性檢查。**Barbet 訓練使用 Pangolin，沒有使用 MiniCPM tokenizer。**

這是最長 128K 的 mixed-length 匯出，**尚非完成的 1M 訓練課程**。模型有原生長 context 配置，與本資料集涵蓋哪些實際長度、能否生成正確長文答案，是不同證據。

## 五、TRAIN 的監督分布：為何不能整包均勻續訓

TRAIN 合計 {exposure['totals']['rows']:,} 筆、{exposure['totals']['input_tokens']:,} input tokens、{exposure['totals']['supervised_tokens']:,} supervised tokens：

{token_scope_table}

**{exposure['reference_only_supervised_fraction']:.2%} 的 TRAIN 監督 tokens 屬於 §reference_only§。** 這表示獨立語義驗證仍有缺口，不表示其中每個答案都錯。

有限圖題有 {exposure['finite_graph']['rows']:,} 個 TRAIN episodes、{exposure['finite_graph']['supervised_tokens']:,} supervised tokens，占所有 TRAIN 監督量 **{exposure['finite_graph_supervised_fraction_all_train']:.2%}**；但在 §verified§ 加 §source_reference§ 的監督量中，占 **{exposure['finite_graph_supervised_fraction_of_verified_plus_source_reference']:.2%}**。這個有明確驗證或參照的子集，只剩 {exposure['non_graph_with_verification_or_source_reference']['supervised_tokens']:,} 個非圖題監督 tokens。

因此，全量池的問題是多數監督尚未獨立確認；改成只取 scoped 子集，則會嚴重偏向圖操作。兩者都不能直接當成已驗證的一般能力課程。

繼承的 domain 標籤如下；標成 coding、math 或 tool_use 不保證題目有足夠難度、語義覆蓋或正確性：

{domain_table}

[完整 TRAIN 監督對帳](full-v4/metadata/training-exposure-evidence.json) · [長度與曝光統計](full-v4/metadata/statistics-summary.json) · [資料使用規則](full-v4/metadata/exposure-policy.json)

## 六、資料格式與實際內容

### Configurations

{config_table}

Records 與 tokenized 是同一批例子的兩種表示，不能把兩者相加當成更多資料。GPU measurements 是同一批例子的 sidecar，也不增加訓練題數。

§full_v4_dispositions§ 預設讀取的是完整處置索引；訓練與研究請明確選擇 config 及 split。

### Full-v4 records

{record_fields}

Full-v4 的 §root_task_id§ 保守沿用原 §family_id§，可能把同一文章或專案的多題分在同一組。獨立問題可同時參考 §source§、§query_hash§ 及 §parent_record_id§；切分時保留這個群組。另附 authored semantic v3 使用自己的根任務契約。

這是從固定 v4 的 §full_v4_verified_records/train§ 實際讀出的欄位摘錄：

§§§json
{fields_example}
§§§

### Tokenized

核心欄位包含 §record_id§、§input_ids§、§labels§、§pangolin_tokens§、§pangolin_supervised_tokens§。§input_ids§ 與 §labels§ 長度相同；§labels§ 保持 **unshifted**。

- System、user、skill、文件、tool observation、padding 與未接受 attempt 都是 §-100§。
- 只監督合格 assistant 行動／回答及必要終止 token。
- Trainer 只 shift 一次，loss 依全域實際 supervised-token 數正規化；不能先各 rank 平均再平均。
- Assistant-only mask 不表示可以 detach 整個 prefix。答案仍需透過梯度學會使用前文。
- 已存 token arrays 不是另一種 tokenizer 的可直接服務請求。

### 實際語義例子：嚴格小於與等於邊界

以下來自 §semantic_v3_records/train§ 的原題，record ID 為 §{sample['record_id']}§，root 為 §{sample['root_task_id']}§，授權 §{sample['license']}§。為方便閱讀，只投影表格的 §id§ 與 §score§；原列另保留來源／目的 ID、缺值與型別等欄位。

原始使用者請求：

> {public['task']}

{sample_table}

經驗證的第一個 assistant 工具動作：

§§§json
{example_call}
§§§

實際工具回執之後的答案為：§{messages[-1]['content']}§

等於門檻的 A、B 必須排除；不能把 §lt§ 寫成 §le§。同 root 的反事實與不變改寫必須放在同一 split，共同保留。這個例子示範資料的語義契約，不代表 v26 已能正確完成新問題。

## 七、凍結 Barbet v26 的全量 GPU 測量

全部 {totals['rows']:,} 筆、{gpu['files']:,} 個 tokenized 分片，以固定模型 [§voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66§](https://huggingface.co/voidful/barbet-skill/tree/67131067ff1825179d5e38e91d499159669e9c66) 完成計算。最多使用同一台 8 張本地 H200；沒有模型更新或序列截斷。

| 指標 | 全量結果 |
|---|---:|
| Supervised-token 加權 NLL | {totals['token_weighted_nll']:.6f} |
| 每列平均 NLL 再取平均 | {totals['example_mean_nll']:.6f} |
| Teacher-forced top-1 命中 | {totals['teacher_forced_top1_correct']:,} / {totals['supervised_tokens']:,}（{totals['teacher_forced_top1_fraction']:.2%}） |
| 每列平均 NLL 的最小／最大值 | {totals['nll_min']:.6f} / {totals['nll_max']:.6f} |
| 平均 readout gradient sketch norm | {totals['mean_readout_gradient_sketch_norm']:.6f} |
| 重複／遺漏 scored IDs | 0 / 0 |
| 因 GPU 分數改變來源品質／目標答案 | 0 / 0 |

Token 命中率不是自由生成任務成功率。低 NLL 可能反映熟悉的格式、重複結構或答案風格；新 NLL 覆蓋完整合格 assistant targets，除非 loss 範圍一致，不能直接與歷史候選選擇 PPL 比較。

每列平均 NLL 分布如下，各 bucket 互斥：

{histogram}

### 梯度投影的實際範圍

使用固定 seed 20261104、雙側 Rademacher 投影，將 detached readout-copy 梯度投影到 64 × 128＝8,192 維；逐筆輸出保存其 norm，以及與七種 TRAIN 語義原型的 cosine。原型只使用全部 2,000 個 authored TRAIN roots，各取原題一次，沒有 DEV／calibration／TEST。

它不是 tied embeddings 的完整梯度，也不是全模型、optimizer-aware 或經訓練驗證的 influence。高 cosine 可能來自共同輸出格式，不是語義正確或可學習性分數。

{prototype_table}

### 模型、tokenizer 與環境

{weights_table}

實際環境包括 Python 3.12.3、PyTorch 2.11.0／CUDA 13.0、Transformers 5.14.1、tokenizers 0.22.2、mamba-ssm 2.3.2.post1、causal-conv1d 1.6.2.post1、PyArrow 25.0.0；保留 native loader 必要的 FP32 參數，使用 BF16 與 Flash Attention，沒有整模型強制 cast。

完整 package／wheel metadata、driver、硬體、attention backend、精度及 worker 設定均有 [實際執行環境紀錄](full-v4/metadata/gpu-runtime-environment.json)，模型／程式／anchor 的雜湊綁在 [固定計畫](full-v4/metadata/gpu-full-score-plan.json)。

## 八、語義對照：低 NLL 下仍存在的弱點

對全部 2,000 個 authored TRAIN roots，建立 **15,430 個候選評分輸入**：原題、改變語義的反事實、不變改寫、移除 skill、移除證據、answer-only，以及相反條件的錯誤候選。

其中 1,715 組第一動作確實不同；285 組第一動作相同，因此不適合用這個首動作配對指標。只有原題與反事實都偏好各自正確候選，才算兩側皆正確：

{family_table}

§0/0§ 代表沒有可計分的不同首動作，不能解讀成該能力零分。這是 **TRAIN 上的條件候選偏好**，不是自由生成準確率、封存 TEST 或未見任務遷移成績。

比較 0/287、排序 0/286、版本規則 4/284，與偏低的 NLL 同時存在。獨立覆核重建所有 9,430 個原題／反事實／不變候選的實際 prefix 與 target，並核對全部 15,430 個評分身分；比較、排序與版本題的候選只差 operation。這支持保留語義條件對照，不能只留下低 NLL 的一側。

增益定義為「移除該內容後的 NLL − 完整輸入 NLL」，正值表示加入該內容提高這個參考答案的機率：

{gains_table}

- 移除證據後仍評相同 gold，只測機率敏感度；真正正確動作可能變成補查或澄清，不能當成缺證據作答準確率。
- §evidence_state§ 的總 log-prob 與長度正規化配對結果不同，必須同時報告。
- IFD-like ratio 是完整輸入與 answer-only NLL 的凍結模型診斷，不是原 IFD 完整訓練流程，也沒有套用門檻刪資料。
- 完整輸入下正確的兩個語義條件，不是「同一題的不同答案版本」。既有 frozen-v26 最低合格 PPL 規則只適用於同一完整輸入的答案候選。

[完整語義結果](full-v4/metadata/gpu-context-probe-summary.json) · [候選建構核對](full-v4/metadata/gpu-context-probe-controls.json) · [資料使用決策](full-v4/metadata/gpu-curriculum-policy.json)

## 九、逐來源與 split 的完整 GPU 統計

以下 NLL 都來自同一份固定 v26 權重；按 supervised tokens 加權的結果與逐筆平均分開：

{source_gpu_table}

來源 NLL 不是資料品質排行榜，也不能跨來源直接推論教學價值或難度。保留原品質範圍、source、split、root、parent 與 query 身分，再解讀模型測量。

## 十、如何載入與使用

### 取得目前 main 的資料

§§§python
from datasets import load_dataset

records = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "full_v4_verified_records",
    revision="main",
    split="train",
    streaming=True,
)
row = next(iter(records))
print(row["record_id"], row["verification_scope"])
§§§

### 固定不可變版本，讀取 Pangolin arrays

§§§python
from datasets import load_dataset

V4_DATA_REVISION = "{V4}"
tokenized = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "full_v4_verified_tokenized",
    revision=V4_DATA_REVISION,
    split="train",
    streaming=True,
)
row = next(iter(tokenized))
assert len(row["input_ids"]) == len(row["labels"]) == row["pangolin_tokens"]
assert sum(label != -100 for label in row["labels"]) == row["pangolin_supervised_tokens"]
§§§

此例只示範讀取與檢查，不是建議對全部 §verified§ 均勻續訓。

### 讀取 GPU sidecar

§§§python
measurements = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "full_v4_gpu_measurements",
    revision=V4_DATA_REVISION,
    split="train",
    streaming=True,
)
measurement = next(iter(measurements))
print(measurement["record_id"], measurement["nll"])
§§§

以 §record_id§ 與 records／tokenized 作精確 join，不依賴列順序或 §zip§。量測欄位包括 §nll_sum§、§nll§、§teacher_forced_top1_correct§、§readout_gradient_sketch_norm§、§semantic_family_cosines§，並保留 source／split／品質 scope；§automatic_training_admission=False§ 表示分數本身不自動批准訓練。

### 建立不含目標答案的推論請求

使用 [固定版本的 full_requests adapter](https://github.com/OpenFormosa/barbet-skill-library/blob/3aabda94f26389d4f356701f48b0efa825a8fe12/barbet_pipeline/full_requests.py)，在該 repository 的 Python path 下：

§§§python
from barbet_pipeline.full_requests import prepare_full_request
row_from_records = next(iter(records))

chat = prepare_full_request(row_from_records, model="YOUR_SERVED_MODEL", mode="chat")
completion = prepare_full_request(row_from_records, model="YOUR_SERVED_MODEL", mode="completion")
# 需要較晚的決策時，明確指定已量測的 target_turn。
§§§

Adapter 只序列化所選 assistant 決策之前可見的內容，不含 target answer、私有評分欄位或未來 observation。Portable tool feedback 使用帶 §TOOL_OBSERVATION:§ 的明確 user continuation，以匹配固定 renderer；Completion 使用精確 Pangolin rendering 與 §add_special_tokens=False§。

工具結果只能接續實際執行的相同行動，不能把已存 teacher 回執接到新的 student action。資料格式／token-budget 控制通過，也不等於某個 vLLM 或 Barbet serving backend 的 cache／工具續接已認證。

## 十一、建議的資料使用原則與尚待實驗的部分

1. 正確性與來源忠實性先於模型偏好。§reference_only§ 不因低 loss 或高 cosine 升級。
2. 同 root 的反事實、不變改寫與不同資訊狀態共同保留；限制同根重播，避免模板量冒充能力覆蓋。
3. 配比按實際 supervised tokens 與獨立 roots／decision points 管理，另設圖題占比限制；不依 rows 均勻取樣一千萬筆。
4. 一般能力／保留資料與新語義課程持續混合。50%／50% 僅是候選起點，尚未經 Barbet 訓練證明最佳。
5. 新能力、未見措辭／schema／組合遷移、一般能力與長 context 保留，必須在同一 checkpoint 驗證。
6. 分開報告模型原始提案、host 檢查、真實執行效果與最終任務成功；不把外部程式修正算成模型學會。
7. 缺證據時訓練補查、澄清或有限未知；證據恢復後應恢復回答，避免只學固定拒答。
8. 不因學生分數或意見不同刪除已合格的台灣主權、人民自決、普世人權資料；事實、引述與作者規範立場仍各自核對。

| 能力訓練證據層 | 本次可證明的範圍 |
|---|---|
| 答案／來源正確性 | 依每列明確 scope；沒有把所有來源宣稱為獨立語義認證 |
| 覆蓋 | 完整來源／長度／曝光分布及七類語義對照；平衡一般能力課程仍待建立 |
| 學得會 | 本次未執行訓練，沒有以合成分數代替可學習性實驗 |
| 能遷移 | 未見模板／schema／組合與自然來源的訓練後測試尚待完成 |
| 保得住 | 同 checkpoint 的一般與長 context 保留尚待模型實驗 |

既有 §semantic_v3_records§／§semantic_v3_tokenized§ 另含 **6,286 TRAIN views／2,000 roots** 與 **629 DEV views／200 roots**。不是百萬個獨立新任務；不要再附加舊 v3 natural-reading config，因為其來源 parents 已包含在 v4。私有 calibration／TEST 與私有 oracle 未公開，也未用於產生或選擇 TRAIN。

本次沒有 CPT／SFT／RL、模型權重修改、訓練後能力提升、Qwen／MiniCPM 等價結論或 1M 自由生成證明。資料處理與模型能力建構須分別驗收。

## 十二、工程控制、恢復與發布驗收

- CPU corpus program suite：343 tests、112 subtests；另有 15 項 bounded-transfer／range／resumable-publication controls。歷史 fork tests 的 34 個 Python deprecation warnings 保留於原 log，沒有隱藏或當成新失敗。
- 實際 Barbet dense NLL 與 selected-position chunked NLL 都是 **{controls['dense_nll']:.9f}**；單筆／batch 及梯度投影方向控制通過。小矩陣 analytic readout gradient 與 autograd 另做獨立數值核對。
- 初次 rank 0 下載遇到 curl exit 7；其他七個原 worker 完成。原失敗收據、日誌與下載重試證據保留。
- 網路續跑恢復後，為使用七張閒置 GPU，對自有單張續跑作明確記錄的 SIGTERM（退出碼 -15）。保留 **{recovery['retained_rows']:,} 筆、{recovery['retained_files']:,} 個已完成分片**，只重新分配 **{recovery['new_rows']:,} 筆、{recovery['new_files']:,} 個缺少分片**。
- 原始 scorer、權重、tokenizer、每檔批次及計算不變，核心 Parquet scoring loop 的 AST 完全一致。32 筆實際 v26 平行恢復控制 NLL 差異 **{recovery_control['maximum_nll_absolute_error']:.1f}**、最低 readout cosine **{recovery_control['minimum_readout_feature_cosine']:.9f}**，token 命中數相同。
- 恢復相關 15 項檢查涵蓋互斥完整分派、來源變動、歷史 log／成功收據竄改、缺少完成、不同 PID 及非零實際退出碼；不把這些工程測試當作模型成績。
- 八個實際補跑程序全部成功退出，才確認原邏輯 assignment 完成。全部 {totals['rows']:,} IDs 在公開 provenance 與實際 scored seen set 中一一對應，沒有遺漏或重複。
- 原始 v4 發布的 4,790 個新資料檔已匿名完整回讀；1,738 個新 metadata／GPU sidecar 檔也逐檔匿名 SHA-256 核對。封存 v4 的 6,529 個檔案樹路徑與身分通過最終核對，未完成標記已移除。
- 10 個 configurations、19 組 config／split 以匿名 §datasets.load_dataset§ 實際載入第一筆，並核對 schema、records／tokenized ID 與 adapter。這是介面整合控制；全列 tokens／labels／前綴與來源已由處理程序另作全量驗證。
- 本次 main 升版另核對全部 payload 路徑與身分，只允許 README 及升版紀錄變動；歷史 v4 分支與舊 main 備份保留。

[原始網路恢復紀錄](full-v4/metadata/gpu-network-resume-history.json) · [平行恢復與實際退出碼](full-v4/metadata/gpu-parallel-recovery-history.json) · [全部 GPU receipts](full-v4/metadata/gpu-score-receipts.json) · [全部來源 join receipts](full-v4/metadata/gpu-aggregation-receipts.json)

## 十三、文獻依據與本次採用範圍

| 文獻／官方資料 | 本次採用 | 明確界線 |
|---|---|---|
| [LESSER（2026 預印本）](https://arxiv.org/abs/2610.03702) | Forward-only、雙側 Rademacher 的 readout gradient sketch | 只測 detached readout-copy，沒有複現完整模型梯度或其訓練選料收益 |
| [LESS](https://arxiv.org/abs/2402.04333) | 借鑑目標能力相關性 | 沒有重現 warmup／optimizer-aware influence；cosine 不等於因果效益 |
| [Instruction-Following Difficulty](https://arxiv.org/abs/2308.12032) | 完整輸入與 answer-only NLL 的 IFD-like ratio | 凍結 v26 診斷；未套用原方法的訓練流程或門檻篩選 |
| [RHO-LOSS](https://arxiv.org/abs/2206.07137) | 區分已會、值得學、尚未學與可學 | 不用 loss 單獨判斷真實性或教學價值 |
| [Sufficient Context](https://arxiv.org/abs/2411.06037) | 區分資訊充分與模型能使用資訊 | Context 內有答案，不構成 Barbet 自由生成正確的證明 |
| [UltraData-SFT-2605 官方資料卡](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) | 區分答案品質、難度／可學習性與實際小型訓練用途驗證 | 本次沒有重現其訓練驗證，也不跨 tokenizer／loss 口徑計算追平倍數 |

## Values and licensing

台灣主權、人民自決、普世人權、民主與人格尊嚴等既有原則維持明確；來源事實、歷史引述與作者規範立場分開處理。不能因學生 NLL 高、cosine 低或模型意見不同，就刪除已合格的價值對齊資料。

公民來源 §OpenFormosa/taiwan-civic-v1§ 共 933 筆：921 筆為 §verified§、12 筆為 §source_reference§。來源引述與作者規範依各列 §verification_scope§ 分別處理；這不是新增政治事實或已訓練價值對齊效果的宣稱。

[公民資料政策](full-v4/metadata/taiwan-civic-policy-v1.json) · [價值對齊政策](full-v4/metadata/value-alignment-policy-v1.json) · [來源工具契約 HOLD](full-v4/metadata/source_tool_contract_holds_v1.json)

本 release 為多來源資料，逐列保留原授權與署名；**沒有整包改授權為 Apache-2.0**。SQuAD 衍生資料保留 CC BY-SA 4.0，新增 authored semantic cases 為 Apache-2.0。§license: other§ 表示請依各來源的實際授權使用。

## 完整可重現入口

- [資料與 metadata 原始 manifest](full-v4/release-manifest.json)
- [原始完整發布收據](full-v4/publication-complete.json)
- [本次 main／README 升版收據](full-v4/main-promotion-20261105.json)
- [完整 CPU 處理總帳](full-v4/metadata/processing-summary.json)
- [逐來源處理收據](full-v4/metadata/source-processing-receipts.json)
- [原始資料匿名回讀](full-v4/metadata/anonymous-data-readback.json)
- [完整 GPU 計畫](full-v4/metadata/gpu-full-score-plan.json)
- [完整 GPU 統計](full-v4/metadata/gpu-complete-summary.json)
- [語義 probe 計畫](full-v4/metadata/gpu-context-probe-plan.json)
- [語義 probe 結果](full-v4/metadata/gpu-context-probe-summary.json)
- [TRAIN 監督分布](full-v4/metadata/training-exposure-evidence.json)
- [資料使用政策](full-v4/metadata/gpu-curriculum-policy.json)
- [來源與環境版本鎖](full-v4/metadata/semantic_v3_sources.lock.json)
- [凍結模型 manifest](full-v4/metadata/semantic_v3_model_manifest.json)
- [資料實作版本](https://github.com/OpenFormosa/barbet-skill-library/tree/3aabda94f26389d4f356701f48b0efa825a8fe12)
- [完成報告與程式版本](https://github.com/OpenFormosa/barbet-skill-library/tree/144d199d471c9f07a0938be3aa0aa002b0dae5dc)
- [原 CPU HTML 報告](full-v4/report.html)／[原 GPU HTML 報告](full-v4/gpu-report.html)

本 README 已包含報告的資料、方法、結果、使用契約、恢復歷程及限制；HTML 與 JSON 保留作原始封存證據及機器讀取用途。
'''
    assert front['configs'] == original_configs
    output = OUTPUT / 'README.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write('---\n' + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + '---\n\n' + text.replace('§', chr(96)))
    print(json.dumps({'readme_bytes': output.stat().st_size, 'lines': len(output.read_text().splitlines()),
        'sha256': sha_file(output), 'configs_unchanged': True, 'output': str(output)}))


if __name__ == '__main__':
    main()
