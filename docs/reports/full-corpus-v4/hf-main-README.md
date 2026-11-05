---
language:
- zh
- en
license: other
license_name: per-source-licenses-preserved
license_link: https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/blob/main/README.md#values-and-licensing
task_categories:
- text-generation
tags:
- barbet
- skill
- tool-use
- audited-data
configs:
- config_name: full_v4_verified_records
  data_files:
  - split: train
    path: full-v4/verified/records/train-*.parquet
  - split: validation
    path: full-v4/verified/records/validation-*.parquet
- config_name: full_v4_verified_tokenized
  data_files:
  - split: train
    path: full-v4/verified/tokenized/train-*.parquet
  - split: validation
    path: full-v4/verified/tokenized/validation-*.parquet
- config_name: full_v4_source_reference_records
  data_files:
  - split: train
    path: full-v4/source_reference/records/train-*.parquet
  - split: validation
    path: full-v4/source_reference/records/validation-*.parquet
- config_name: full_v4_source_reference_tokenized
  data_files:
  - split: train
    path: full-v4/source_reference/tokenized/train-*.parquet
  - split: validation
    path: full-v4/source_reference/tokenized/validation-*.parquet
- config_name: full_v4_reference_only_records
  data_files:
  - split: train
    path: full-v4/reference_only/records/train-*.parquet
  - split: validation
    path: full-v4/reference_only/records/validation-*.parquet
- config_name: full_v4_reference_only_tokenized
  data_files:
  - split: train
    path: full-v4/reference_only/tokenized/train-*.parquet
  - split: validation
    path: full-v4/reference_only/tokenized/validation-*.parquet
- config_name: full_v4_dispositions
  default: true
  data_files:
  - split: corpus
    path: full-v4/dispositions/*.parquet
- config_name: full_v4_gpu_measurements
  data_files:
  - split: train
    path: full-v4/gpu-quality/measurements/*/train-*.parquet
  - split: validation
    path: full-v4/gpu-quality/measurements/*/validation-*.parquet
- config_name: semantic_v3_records
  data_files:
  - split: train
    path: semantic-v3/pilot/records/train-*.parquet
  - split: validation
    path: semantic-v3/pilot/records/dev-*.parquet
- config_name: semantic_v3_tokenized
  data_files:
  - split: train
    path: semantic-v3/pilot/tokenized/train-*.parquet
  - split: validation
    path: semantic-v3/pilot/tokenized/dev-*.parquet
---

# Barbet long-context SFT v4：完整資料、GPU 評量與使用報告

本資料集的 **`main` 已採用完整驗收的 v4**。本頁集中收錄資料規模、內容、來源、品質範圍、全量處理方式、凍結 Barbet v26 的 GPU 結果、語義弱項、使用契約、文獻與可重現證據。

全部 **10,018,967 筆 active parents**、1,504 個來源分片已處理完成。匯出 **10,018,961 筆**，新增隔離 6 筆；全部 1,643 個 tokenized 分片均完成 GPU 評分、逐筆來源與唯一 ID 對帳。

**資料處理與發布已完成；`training_ready=false`。** 這是一個依驗證範圍分層的資料池。低 loss、來源 gold、大量列數及合法格式，各自都不能證明學生能學會、能遷移或能保住一般能力。

## 版本與主分支

| 項目 | 固定版本或位置 |
|---|---|
| 目前主要入口 | [main](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/main) |
| 完整驗收的 v4 資料版本 | [`7103d1b1bc1fb4cda1298c084a66bda542d61151`](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/7103d1b1bc1fb4cda1298c084a66bda542d61151) |
| 保留的 v4 分支 | [full-corpus-v4-20261104](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/full-corpus-v4-20261104) |
| 升版前 main | [`b88f4ff7171bb6fb8c1b775ae21de814ea2123fb`](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/b88f4ff7171bb6fb8c1b775ae21de814ea2123fb) |
| 升版前 main 備份 | [archive/main-before-v4-20261105](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/archive/main-before-v4-20261105) |
| 既有語義課程版本 | [`4b38655b336aa2d0c01f6f0047701e7eeec23ed6`](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/4b38655b336aa2d0c01f6f0047701e7eeec23ed6) |
| 本次主分支升版與 README 紀錄 | [main-promotion-20261105.json](full-v4/main-promotion-20261105.json) |

本次升版直接沿用已驗收的資料與測量檔，更新 README 並新增升版紀錄。原始 [release manifest](full-v4/release-manifest.json) 與 [publication receipt](full-v4/publication-complete.json) 保留原樣，記錄的是上表固定 v4 revision；其中舊 README 的雜湊對應該封存版本。新版 README 的雜湊與允許變動路徑另記於升版紀錄，不回寫歷史收據。

實驗請固定 `revision="7103d1b1bc1fb4cda1298c084a66bda542d61151"`，或固定本次 main 的完整 commit。`main` 方便取得目前版本，本身不是不可變的實驗識別。

## 一、資料規模與品質分層

| 品質範圍 | TRAIN 筆數 | Validation 筆數 | 驗證意義 |
|---|---|---|---|
| `verified` | 5,486,313 | 112,738 | 依逐筆 scope 通過有限圖獨立計算，或來源 gold／證據／引述核對 |
| `source_reference` | 608,930 | 12,693 | 來源題目與參考答案忠實性；尚非獨立語義裁判 |
| `reference_only` | 3,722,424 | 75,863 | 完整結構、token 與前綴檢查；答案語義仍待獨立驗證 |

- TRAIN：**9,817,667** 筆；validation：**201,294** 筆。
- 完整 Pangolin input tokens：**30,174,631,335**。
- Assistant supervised tokens：**3,951,574,820**。
- 合格 assistant 決策前綴：**17,308,873** 個。
- 原有隔離／拒絕資料：**310,662** 筆，維持排除。
- 本次新增隔離：5 筆超出完整推論預算而不截斷；1 筆保留既有自然閱讀視圖的 HOLD。

`verified` 包含不同證據範圍，不能解讀成「每一列都由獨立世界知識 oracle 證明」。有限圖題有獨立可執行語義核對；SQuAD 使用固定來源 gold／span／資訊充分性標籤；HotpotQA 使用來源答案及支持句；公民資料則區分來源引述核對與作者規範立場。請逐筆閱讀 `verification_scope`。

`source_reference` 保留 Natural Instructions 的原始題目、定義、例子、輸入與目標，以及先前合格的作者規範案例。它證明來源忠實性，尚非獨立答案正確性裁判。

`reference_only` 包含尚無合適獨立驗證器的數學、程式、一般指令與工具答案。它們接受相同的完整結構、token、mask 與推論前綴檢查，但**不因低 NLL 或格式合法而自動加入訓練**。

## 二、所有來源的實際處理數量

| 來源識別 | 處理 | verified | source_reference | reference_only | 新增隔離 |
|---|---|---|---|---|---|
| `Agent-Ark/Toucan-1.5M` | 201,939 | 0 | 0 | 201,938 | 1 |
| `ConceptNet/5.7-original-lexical-english` | 867,454 | 867,454 | 0 | 0 | 0 |
| `IFM/TxT360-3efforts` | 458,332 | 0 | 0 | 458,332 | 0 |
| `Open-Orca/OpenOrca` | 1,077,547 | 0 | 0 | 1,077,547 | 0 |
| `OpenFormosa/taiwan-civic-v1` | 933 | 921 | 12 | 0 | 0 |
| `Wikidata5M/original-inductive-train` | 4,547,338 | 4,547,338 | 0 | 0 | 0 |
| `allenai/natural-instructions` | 621,611 | 0 | 621,611 | 0 | 0 |
| `hotpotqa/hotpot_qa` | 69,213 | 69,213 | 0 | 0 | 0 |
| `microsoft/orca-math-word-problems-200k` | 191,149 | 0 | 0 | 191,149 | 0 |
| `nvidia/Nemotron-SFT-Agentic-v2` | 735,322 | 0 | 0 | 735,318 | 4 |
| `nvidia/OpenCodeInstruct` | 1,031,318 | 0 | 0 | 1,031,318 | 0 |
| `rajpurkar/squad_v2` | 114,126 | 114,125 | 0 | 0 | 1 |
| `voidful/agent-sft` | 102,685 | 0 | 0 | 102,685 | 0 |

來源識別不一律代表存在同名 HF repository。逐筆授權、原始 revision、文件／題號與來源定位保留於 `license`、`source_refs_json` 及 [source receipts](full-v4/metadata/source-processing-receipts.json)。

這些是重新處理的既有問題與 episode；列數不等於獨立語義家族數，也不等於新增了一千萬個不同推理問題。

## 三、全量實際改動與驗證

1. **逐筆重建原資料。** 從 typed messages、工具與 target spans 重建原始文字，重新計算原有 Pangolin IDs／labels 並比較；匯出後再讀回、重新計算輸出 arrays。
2. **Skill 更新。** 9,374,082 筆 singleton-target 走版本化 library 更新路徑。實際文字改變 **8,290,103** 筆，**1,728,858** 筆逐 byte 相同；其中 1,198,070 筆原本已符合新版 skill。路徑旗標不是實際文字差異的證明。
3. **既有多候選答案保留。** 凍結 v26 已選定的多候選完整輸入與目標保持原樣；不把舊 PPL 當成修改後輸入的分數。新 GPU 測量另行提供，沒有重新選答案。
4. **自然閱讀。** 114,091 筆 SQuAD 視圖採已接受的自然、來源限定答案 renderer，保留來源 gold、原 split、前／後證據位置及可選 skill。沒有憑空增加來源內容。
5. **完整推論前綴。** 全部 17,308,873 個合格 assistant 決策都驗證 answer-free Chat 與 exact-renderer Completion 的身分、雜湊及實測 token 預算。
6. **實際工具回放。** 3,461,420 次可信有限圖查詢由公開 state 重新執行，另以獨立圖演算法檢查答案；來源任意程式與遠端動作沒有在 host 執行。
7. **圖題語義連結。** 完整核對實際可見題目中的 operation、方向與明確 entity IDs，inline edges 必須對得上登錄 state 與實際 memory rendering。Display labels 及當代世界事實不因此被獨立認證。
8. **QA／NI 的可見輸入。** 所有匯出 SQuAD、HotpotQA 與 Natural Instructions 的題目、定義、例子、passage 與實際檢索觀察，都由固定來源重建。這比只比對最後答案多了一層來源忠實性檢查。
9. **Split 與隔離。** 全 active corpus 的 record ID、root 與 query hash 跨 split 沒有衝突；原有 HOLD 保留。未取得遠端舊 516 題逐題輸出，因此不聲稱完成其語義近鄰除汙。
10. **公開 bytes。** 全部新資料、測量與 metadata 逐檔匿名下載並比對 SHA-256。只釋放本次產生且已證實遠端備份的本地輸出；原始來源、checkpoint、歷史收據保留。

[文字實際差異](full-v4/metadata/actual-changes-summary.json) · [來源可見前綴](full-v4/metadata/source-visibility-summary.json) · [圖題語義核對](full-v4/metadata/graph-visibility-summary.json) · [完整處理總帳](full-v4/metadata/processing-summary.json)

## 四、長度分布與真正的長上下文範圍

下表是**完整序列的實際 Pangolin token 數**，包含 assistant target，各 bucket 互斥：

| 完整 Pangolin sequence tokens | 筆數 | 占比 |
|---|---|---|
| 1–2,048 | 4,762,268 | 47.53% |
| 2,049–4,096 | 3,633,591 | 36.27% |
| 4,097–8,192 | 1,336,412 | 13.34% |
| 8,193–16,384 | 197,641 | 1.97% |
| 16,385–32,768 | 59,148 | 0.59% |
| 32,769–65,536 | 25,194 | 0.25% |
| 65,537–131,072 | 4,707 | 0.05% |

- 實際最長完整序列：**126,844 tokens**。
- 本 release 的完整 sequence 上限：131,072 tokens。
- 每個推論前綴保留 4,096 output tokens，並以 Pangolin 與實際 Qwen served tokenizer 檢查完整預算，沒有靜默截斷。
- 4,096 是請求的輸出保留，不代表每個參考答案都經自由生成證明能在此額度自然完成。
- Qwen tokenizer 僅作 transport／budget 相容性檢查。**Barbet 訓練使用 Pangolin，沒有使用 MiniCPM tokenizer。**

這是最長 128K 的 mixed-length 匯出，**尚非完成的 1M 訓練課程**。模型有原生長 context 配置，與本資料集涵蓋哪些實際長度、能否生成正確長文答案，是不同證據。

## 五、TRAIN 的監督分布：為何不能整包均勻續訓

TRAIN 合計 9,817,667 筆、29,570,896,855 input tokens、3,872,323,168 supervised tokens：

| TRAIN 品質範圍 | 筆數 | Input tokens | Supervised tokens | 監督 token 占比 |
|---|---|---|---|---|
| `reference_only` | 3,722,424 | 12,915,043,760 | 3,454,355,358 | 89.21% |
| `source_reference` | 608,930 | 1,535,759,209 | 7,562,488 | 0.20% |
| `verified` | 5,486,313 | 15,120,093,886 | 410,405,322 | 10.60% |

**89.21% 的 TRAIN 監督 tokens 屬於 `reference_only`。** 這表示獨立語義驗證仍有缺口，不表示其中每個答案都錯。

有限圖題有 5,306,336 個 TRAIN episodes、405,637,194 supervised tokens，占所有 TRAIN 監督量 **10.48%**；但在 `verified` 加 `source_reference` 的監督量中，占 **97.05%**。這個有明確驗證或參照的子集，只剩 12,330,616 個非圖題監督 tokens。

因此，全量池的問題是多數監督尚未獨立確認；改成只取 scoped 子集，則會嚴重偏向圖操作。兩者都不能直接當成已驗證的一般能力課程。

繼承的 domain 標籤如下；標成 coding、math 或 tool_use 不保證題目有足夠難度、語義覆蓋或正確性：

| 繼承的 domain 標籤 | TRAIN 筆數 | Supervised tokens | 監督 token 占比 |
|---|---|---|---|
| `coding` | 1,217,192 | 1,360,165,159 | 35.13% |
| `math` | 541,125 | 363,346,222 | 9.38% |
| `reading` | 5,309,064 | 302,373,713 | 7.81% |
| `search` | 1 | 601 | 0.00% |
| `tool_use` | 2,750,285 | 1,846,437,473 | 47.68% |

[完整 TRAIN 監督對帳](full-v4/metadata/training-exposure-evidence.json) · [長度與曝光統計](full-v4/metadata/statistics-summary.json) · [資料使用規則](full-v4/metadata/exposure-policy.json)

## 六、資料格式與實際內容

### Configurations

| Configuration | Split | 用途 |
|---|---|---|
| `full_v4_verified_records` / `full_v4_verified_tokenized` | train / validation | 依明確 scope 核對的資料；仍須設計能力平衡與保留課程 |
| `full_v4_source_reference_records` / `full_v4_source_reference_tokenized` | train / validation | 來源參照資料；須區分參照忠實性與答案正確性 |
| `full_v4_reference_only_records` / `full_v4_reference_only_tokenized` | train / validation | 待進一步驗證的獨立資料池，不自動納入訓練 |
| `full_v4_dispositions`（預設） | corpus | 全部 active parents 的處置索引，包含隔離；不是 TRAIN |
| `full_v4_gpu_measurements` | train / validation | 逐筆模型測量 sidecar，依 record_id 連接；不是新增訓練列 |
| `semantic_v3_records` / `semantic_v3_tokenized` | train / validation | 2,000 TRAIN roots／200 DEV roots 的獨立語義課程 |

Records 與 tokenized 是同一批例子的兩種表示，不能把兩者相加當成更多資料。GPU measurements 是同一批例子的 sidecar，也不增加訓練題數。

`full_v4_dispositions` 預設讀取的是完整處置索引；訓練與研究請明確選擇 config 及 split。

### Full-v4 records

| Records 欄位 | 內容與用途 |
|---|---|
| `record_id` / `parent_record_id` | 新匯出列與原始 parent 的精確身分；GPU sidecar 以 record_id 連接 |
| `family_id` / `root_task_id` / `query_hash` | 繼承的來源群組及問題身分；root_task_id 可能涵蓋同文章的多個問題 |
| `split` / `domain` / `source` / `upstream_source` / `license` | 來源、原始切分、繼承分類及逐筆授權 |
| `processing_status` / `verification_scope` | 品質層級及實際驗證範圍 |
| `semantic_correctness_independently_certified` | 僅對獨立核對的有限圖語義設為 true；來源 gold 或引述核對不冒充獨立語義證明 |
| `messages_json` / `tools_json` / `memory_json` / `skills_json` | Typed 訊息、工具、記憶與 skill 的 JSON 字串 |
| `text` / `target_spans_json` / `renderer_version` | 精確 Pangolin rendering、合格 assistant 監督區間與版本 |
| `checks_json` / `execution_receipts_json` / `source_refs_json` | 驗證證據、已發生的執行回執與來源定位 |
| `inference_decisions_json` | 每個合格 assistant 決策的 answer-free 前綴雜湊及實測推論預算 |
| `pangolin_tokens` / `pangolin_supervised_tokens` / `output_reserve_tokens` | 完整輸入、監督與推論輸出保留的 token 計量 |
| `target_selection_scope` / `parent_record_sha256` | 既有答案選擇範圍及原 parent 的雜湊 |
| `input_view_changed` / `natural_answer_rewritten` / `skill_prefill_updated` | 處理路徑旗標；是否真的改 byte 要看完整文字差異盤點 |

Full-v4 的 `root_task_id` 保守沿用原 `family_id`，可能把同一文章或專案的多題分在同一組。獨立問題可同時參考 `source`、`query_hash` 及 `parent_record_id`；切分時保留這個群組。另附 authored semantic v3 使用自己的根任務契約。

這是從固定 v4 的 `full_v4_verified_records/train` 實際讀出的欄位摘錄：

```json
{
  "record_id": "fe4edfdd14caddf2ea8831a5323a014c32b61597818503d687a9b47ad4910299",
  "source": "ConceptNet/5.7-original-lexical-english",
  "processing_status": "verified",
  "verification_scope": "independent_finite_graph_query_and_tool_reexecution",
  "pangolin_tokens": 2310,
  "pangolin_supervised_tokens": 34
}
```

### Tokenized

核心欄位包含 `record_id`、`input_ids`、`labels`、`pangolin_tokens`、`pangolin_supervised_tokens`。`input_ids` 與 `labels` 長度相同；`labels` 保持 **unshifted**。

- System、user、skill、文件、tool observation、padding 與未接受 attempt 都是 `-100`。
- 只監督合格 assistant 行動／回答及必要終止 token。
- Trainer 只 shift 一次，loss 依全域實際 supervised-token 數正規化；不能先各 rank 平均再平均。
- Assistant-only mask 不表示可以 detach 整個 prefix。答案仍需透過梯度學會使用前文。
- 已存 token arrays 不是另一種 tokenizer 的可直接服務請求。

### 實際語義例子：嚴格小於與等於邊界

以下來自 `semantic_v3_records/train` 的原題，record ID 為 `36679a14963bdd7410fb743925205e16c1f63bf62c6c11f8cdd1f93a649d5f88`，root 為 `33636ccdd2a4d0c388cf3f0971a1bd11fca55144c5bb91ec32f322d50ef6d30f`，授權 `apache-2.0`。為方便閱讀，只投影表格的 `id` 與 `score`；原列另保留來源／目的 ID、缺值與型別等欄位。

原始使用者請求：

> Hey, grab the IDs from table_33636ccd where score is strictly less than 8.

| 實際項目 ID | score（Decimal 字串） |
|---|---|
| `d2a4_A` | `"8"` |
| `d2a4_B` | `"8"` |
| `d2a4_C` | `"19"` |
| `d2a4_D` | `"6"` |
| `d2a4_E` | `"11"` |
| `d2a4_F` | `"-1"` |

經驗證的第一個 assistant 工具動作：

```json
{
  "arguments": {
    "field": "score",
    "operation": "lt",
    "table_id": "table_33636ccd",
    "threshold": "8",
    "tie_rule": "id_ascending"
  },
  "name": "table_select"
}
```

實際工具回執之後的答案為：`Matching item IDs: d2a4_D, d2a4_F.`

等於門檻的 A、B 必須排除；不能把 `lt` 寫成 `le`。同 root 的反事實與不變改寫必須放在同一 split，共同保留。這個例子示範資料的語義契約，不代表 v26 已能正確完成新問題。

## 七、凍結 Barbet v26 的全量 GPU 測量

全部 10,018,961 筆、1,643 個 tokenized 分片，以固定模型 [`voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66`](https://huggingface.co/voidful/barbet-skill/tree/67131067ff1825179d5e38e91d499159669e9c66) 完成計算。最多使用同一台 8 張本地 H200；沒有模型更新或序列截斷。

| 指標 | 全量結果 |
|---|---:|
| Supervised-token 加權 NLL | 1.093813 |
| 每列平均 NLL 再取平均 | 1.503019 |
| Teacher-forced top-1 命中 | 3,013,224,463 / 3,951,574,820（76.25%） |
| 每列平均 NLL 的最小／最大值 | 0.006864 / 16.085125 |
| 平均 readout gradient sketch norm | 6.435919 |
| 重複／遺漏 scored IDs | 0 / 0 |
| 因 GPU 分數改變來源品質／目標答案 | 0 / 0 |

Token 命中率不是自由生成任務成功率。低 NLL 可能反映熟悉的格式、重複結構或答案風格；新 NLL 覆蓋完整合格 assistant targets，除非 loss 範圍一致，不能直接與歷史候選選擇 PPL 比較。

每列平均 NLL 分布如下，各 bucket 互斥：

| 每列平均 assistant NLL 範圍 | 筆數 |
|---|---|
| (0, 0.125] | 2,577 |
| (0.125, 0.25] | 37,454 |
| (0.25, 0.5] | 669,167 |
| (0.5, 1] | 3,481,592 |
| (1, 2] | 3,620,281 |
| (2, 4] | 1,848,936 |
| (4, 8] | 287,533 |
| (8, 16] | 71,420 |
| (16, 32] | 1 |
| >32 | 0 |

### 梯度投影的實際範圍

使用固定 seed 20261104、雙側 Rademacher 投影，將 detached readout-copy 梯度投影到 64 × 128＝8,192 維；逐筆輸出保存其 norm，以及與七種 TRAIN 語義原型的 cosine。原型只使用全部 2,000 個 authored TRAIN roots，各取原題一次，沒有 DEV／calibration／TEST。

它不是 tied embeddings 的完整梯度，也不是全模型、optimizer-aware 或經訓練驗證的 influence。高 cosine 可能來自共同輸出格式，不是語義正確或可學習性分數。

| TRAIN 語義原型 | 全資料平均 cosine |
|---|---|
| `binding` | 0.132556 |
| `comparison` | 0.104446 |
| `evidence_state` | 0.053941 |
| `operation_state` | 0.078547 |
| `ordering` | 0.105531 |
| `rule_version` | 0.089673 |
| `units` | 0.064274 |

### 模型、tokenizer 與環境

| 模型檔案 | SHA-256 |
|---|---|
| `model.safetensors` | `4705731266f202ef031a2a70db88c777b4799a8b88e1671557b7062d1ecb6730` |
| `config.json` | `2b7b9f1cdfc39805084dcf8e668390c4e2aec390054957b227c31f9bf0b6e983` |
| `modeling_barbet.py` | `68101aa44ca98009e2ec1d9bb33c9110e77c76d1bd9d57ef6ddbe0a11e18d541` |
| `configuration_barbet.py` | `fdd956158d9f7c96bf3b5a90f95ddc4470a074032be99d32536964a0b6a5c042` |
| `load_barbet.py` | `803661d0387f98619222859e4f1120df212d19272451adf0229d9396b25197e2` |
| `tokenizer.json` | `f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176` |

實際環境包括 Python 3.12.3、PyTorch 2.11.0／CUDA 13.0、Transformers 5.14.1、tokenizers 0.22.2、mamba-ssm 2.3.2.post1、causal-conv1d 1.6.2.post1、PyArrow 25.0.0；保留 native loader 必要的 FP32 參數，使用 BF16 與 Flash Attention，沒有整模型強制 cast。

完整 package／wheel metadata、driver、硬體、attention backend、精度及 worker 設定均有 [實際執行環境紀錄](full-v4/metadata/gpu-runtime-environment.json)，模型／程式／anchor 的雜湊綁在 [固定計畫](full-v4/metadata/gpu-full-score-plan.json)。

## 八、語義對照：低 NLL 下仍存在的弱點

對全部 2,000 個 authored TRAIN roots，建立 **15,430 個候選評分輸入**：原題、改變語義的反事實、不變改寫、移除 skill、移除證據、answer-only，以及相反條件的錯誤候選。

其中 1,715 組第一動作確實不同；285 組第一動作相同，因此不適合用這個首動作配對指標。只有原題與反事實都偏好各自正確候選，才算兩側皆正確：

| 語義能力 | TRAIN roots | 有區分力的配對 | 總 log-prob 兩側皆正確 | 長度正規化兩側皆正確 | 原題平均 NLL |
|---|---|---|---|---|---|
| `binding` | 286 | 286 | 243/286 | 243/286 | 0.586110 |
| `comparison` | 287 | 287 | 0/287 | 0/287 | 0.511366 |
| `evidence_state` | 286 | 286 | 229/286 | 150/286 | 1.897568 |
| `operation_state` | 285 | 0 | 0/0 | 0/0 | 0.449451 |
| `ordering` | 286 | 286 | 0/286 | 0/286 | 0.643917 |
| `rule_version` | 284 | 284 | 4/284 | 4/284 | 0.652108 |
| `units` | 286 | 286 | 284/286 | 284/286 | 0.488129 |

`0/0` 代表沒有可計分的不同首動作，不能解讀成該能力零分。這是 **TRAIN 上的條件候選偏好**，不是自由生成準確率、封存 TEST 或未見任務遷移成績。

比較 0/287、排序 0/286、版本規則 4/284，與偏低的 NLL 同時存在。獨立覆核重建所有 9,430 個原題／反事實／不變候選的實際 prefix 與 target，並核對全部 15,430 個評分身分；比較、排序與版本題的候選只差 operation。這支持保留語義條件對照，不能只留下低 NLL 的一側。

增益定義為「移除該內容後的 NLL − 完整輸入 NLL」，正值表示加入該內容提高這個參考答案的機率：

| 語義能力 | Skill 增益（nats） | 證據增益（nats） | IFD-like ratio | 不變改寫的 NLL 變化 |
|---|---|---|---|---|
| `binding` | -0.034916 | +0.062457 | 0.183303 | -0.003685 |
| `comparison` | -0.012252 | -0.063021 | 0.137801 | +0.015079 |
| `evidence_state` | -0.000104 | +1.439017 | 0.408607 | -0.008345 |
| `operation_state` | +0.009728 | -0.033714 | 0.131232 | -0.000123 |
| `ordering` | -0.024509 | -0.034190 | 0.178946 | -0.012733 |
| `rule_version` | +0.069392 | +0.012667 | 0.180314 | -0.009667 |
| `units` | +0.006774 | +0.027723 | 0.148821 | +0.001030 |

- 移除證據後仍評相同 gold，只測機率敏感度；真正正確動作可能變成補查或澄清，不能當成缺證據作答準確率。
- `evidence_state` 的總 log-prob 與長度正規化配對結果不同，必須同時報告。
- IFD-like ratio 是完整輸入與 answer-only NLL 的凍結模型診斷，不是原 IFD 完整訓練流程，也沒有套用門檻刪資料。
- 完整輸入下正確的兩個語義條件，不是「同一題的不同答案版本」。既有 frozen-v26 最低合格 PPL 規則只適用於同一完整輸入的答案候選。

[完整語義結果](full-v4/metadata/gpu-context-probe-summary.json) · [候選建構核對](full-v4/metadata/gpu-context-probe-controls.json) · [資料使用決策](full-v4/metadata/gpu-curriculum-policy.json)

## 九、逐來源與 split 的完整 GPU 統計

以下 NLL 都來自同一份固定 v26 權重；按 supervised tokens 加權的結果與逐筆平均分開：

| 品質範圍 | 來源 | Split | 筆數 | Token 加權 NLL | 逐筆平均 NLL |
|---|---|---|---|---|---|
| `reference_only` | `Agent-Ark/Toucan-1.5M` | train | 197,952 | 1.227484 | 1.217405 |
| `reference_only` | `Agent-Ark/Toucan-1.5M` | validation | 3,986 | 1.205122 | 1.207323 |
| `reference_only` | `IFM/TxT360-3efforts` | train | 449,057 | 1.242953 | 1.286992 |
| `reference_only` | `IFM/TxT360-3efforts` | validation | 9,275 | 1.242521 | 1.286297 |
| `reference_only` | `Open-Orca/OpenOrca` | train | 1,056,085 | 0.776013 | 0.903071 |
| `reference_only` | `Open-Orca/OpenOrca` | validation | 21,462 | 0.781551 | 0.907438 |
| `reference_only` | `microsoft/orca-math-word-problems-200k` | train | 187,340 | 0.778241 | 0.792936 |
| `reference_only` | `microsoft/orca-math-word-problems-200k` | validation | 3,809 | 0.776858 | 0.793357 |
| `reference_only` | `nvidia/Nemotron-SFT-Agentic-v2` | train | 720,604 | 1.272244 | 1.355179 |
| `reference_only` | `nvidia/Nemotron-SFT-Agentic-v2` | validation | 14,714 | 1.267833 | 1.354367 |
| `reference_only` | `nvidia/OpenCodeInstruct` | train | 1,010,824 | 0.498734 | 0.521373 |
| `reference_only` | `nvidia/OpenCodeInstruct` | validation | 20,494 | 0.499147 | 0.521262 |
| `reference_only` | `voidful/agent-sft` | train | 100,562 | 0.867759 | 0.680435 |
| `reference_only` | `voidful/agent-sft` | validation | 2,123 | 0.853581 | 0.650022 |
| `source_reference` | `OpenFormosa/taiwan-civic-v1` | train | 12 | 2.931597 | 3.010313 |
| `source_reference` | `allenai/natural-instructions` | train | 608,918 | 2.106278 | 4.473552 |
| `source_reference` | `allenai/natural-instructions` | validation | 12,693 | 2.099858 | 4.480541 |
| `verified` | `ConceptNet/5.7-original-lexical-english` | train | 850,189 | 0.901079 | 1.143266 |
| `verified` | `ConceptNet/5.7-original-lexical-english` | validation | 17,265 | 0.900630 | 1.146204 |
| `verified` | `OpenFormosa/taiwan-civic-v1` | train | 908 | 0.373435 | 0.560652 |
| `verified` | `OpenFormosa/taiwan-civic-v1` | validation | 13 | 0.553101 | 0.793146 |
| `verified` | `Wikidata5M/original-inductive-train` | train | 4,456,147 | 1.198383 | 1.606436 |
| `verified` | `Wikidata5M/original-inductive-train` | validation | 91,191 | 1.196835 | 1.605774 |
| `verified` | `hotpotqa/hotpot_qa` | train | 67,810 | 1.731910 | 2.189029 |
| `verified` | `hotpotqa/hotpot_qa` | validation | 1,403 | 1.737053 | 2.175339 |
| `verified` | `rajpurkar/squad_v2` | train | 111,259 | 1.232653 | 2.299765 |
| `verified` | `rajpurkar/squad_v2` | validation | 2,866 | 1.525334 | 3.047885 |

來源 NLL 不是資料品質排行榜，也不能跨來源直接推論教學價值或難度。保留原品質範圍、source、split、root、parent 與 query 身分，再解讀模型測量。

## 十、如何載入與使用

### 取得目前 main 的資料

```python
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
```

### 固定不可變版本，讀取 Pangolin arrays

```python
from datasets import load_dataset

V4_DATA_REVISION = "7103d1b1bc1fb4cda1298c084a66bda542d61151"
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
```

此例只示範讀取與檢查，不是建議對全部 `verified` 均勻續訓。

### 讀取 GPU sidecar

```python
measurements = load_dataset(
    "OpenFormosa/barbet-long-context-sft",
    "full_v4_gpu_measurements",
    revision=V4_DATA_REVISION,
    split="train",
    streaming=True,
)
measurement = next(iter(measurements))
print(measurement["record_id"], measurement["nll"])
```

以 `record_id` 與 records／tokenized 作精確 join，不依賴列順序或 `zip`。量測欄位包括 `nll_sum`、`nll`、`teacher_forced_top1_correct`、`readout_gradient_sketch_norm`、`semantic_family_cosines`，並保留 source／split／品質 scope；`automatic_training_admission=False` 表示分數本身不自動批准訓練。

### 建立不含目標答案的推論請求

使用 [固定版本的 full_requests adapter](https://github.com/OpenFormosa/barbet-skill-library/blob/3aabda94f26389d4f356701f48b0efa825a8fe12/barbet_pipeline/full_requests.py)，在該 repository 的 Python path 下：

```python
from barbet_pipeline.full_requests import prepare_full_request
row_from_records = next(iter(records))

chat = prepare_full_request(row_from_records, model="YOUR_SERVED_MODEL", mode="chat")
completion = prepare_full_request(row_from_records, model="YOUR_SERVED_MODEL", mode="completion")
# 需要較晚的決策時，明確指定已量測的 target_turn。
```

Adapter 只序列化所選 assistant 決策之前可見的內容，不含 target answer、私有評分欄位或未來 observation。Portable tool feedback 使用帶 `TOOL_OBSERVATION:` 的明確 user continuation，以匹配固定 renderer；Completion 使用精確 Pangolin rendering 與 `add_special_tokens=False`。

工具結果只能接續實際執行的相同行動，不能把已存 teacher 回執接到新的 student action。資料格式／token-budget 控制通過，也不等於某個 vLLM 或 Barbet serving backend 的 cache／工具續接已認證。

## 十一、建議的資料使用原則與尚待實驗的部分

1. 正確性與來源忠實性先於模型偏好。`reference_only` 不因低 loss 或高 cosine 升級。
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

既有 `semantic_v3_records`／`semantic_v3_tokenized` 另含 **6,286 TRAIN views／2,000 roots** 與 **629 DEV views／200 roots**。不是百萬個獨立新任務；不要再附加舊 v3 natural-reading config，因為其來源 parents 已包含在 v4。私有 calibration／TEST 與私有 oracle 未公開，也未用於產生或選擇 TRAIN。

本次沒有 CPT／SFT／RL、模型權重修改、訓練後能力提升、Qwen／MiniCPM 等價結論或 1M 自由生成證明。資料處理與模型能力建構須分別驗收。

## 十二、工程控制、恢復與發布驗收

- CPU corpus program suite：343 tests、112 subtests；另有 15 項 bounded-transfer／range／resumable-publication controls。歷史 fork tests 的 34 個 Python deprecation warnings 保留於原 log，沒有隱藏或當成新失敗。
- 實際 Barbet dense NLL 與 selected-position chunked NLL 都是 **2.083169699**；單筆／batch 及梯度投影方向控制通過。小矩陣 analytic readout gradient 與 autograd 另做獨立數值核對。
- 初次 rank 0 下載遇到 curl exit 7；其他七個原 worker 完成。原失敗收據、日誌與下載重試證據保留。
- 網路續跑恢復後，為使用七張閒置 GPU，對自有單張續跑作明確記錄的 SIGTERM（退出碼 -15）。保留 **9,545,945 筆、1,496 個已完成分片**，只重新分配 **473,016 筆、147 個缺少分片**。
- 原始 scorer、權重、tokenizer、每檔批次及計算不變，核心 Parquet scoring loop 的 AST 完全一致。32 筆實際 v26 平行恢復控制 NLL 差異 **0.0**、最低 readout cosine **0.999999821**，token 命中數相同。
- 恢復相關 15 項檢查涵蓋互斥完整分派、來源變動、歷史 log／成功收據竄改、缺少完成、不同 PID 及非零實際退出碼；不把這些工程測試當作模型成績。
- 八個實際補跑程序全部成功退出，才確認原邏輯 assignment 完成。全部 10,018,961 IDs 在公開 provenance 與實際 scored seen set 中一一對應，沒有遺漏或重複。
- 原始 v4 發布的 4,790 個新資料檔已匿名完整回讀；1,738 個新 metadata／GPU sidecar 檔也逐檔匿名 SHA-256 核對。封存 v4 的 6,529 個檔案樹路徑與身分通過最終核對，未完成標記已移除。
- 10 個 configurations、19 組 config／split 以匿名 `datasets.load_dataset` 實際載入第一筆，並核對 schema、records／tokenized ID 與 adapter。這是介面整合控制；全列 tokens／labels／前綴與來源已由處理程序另作全量驗證。
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

公民來源 `OpenFormosa/taiwan-civic-v1` 共 933 筆：921 筆為 `verified`、12 筆為 `source_reference`。來源引述與作者規範依各列 `verification_scope` 分別處理；這不是新增政治事實或已訓練價值對齊效果的宣稱。

[公民資料政策](full-v4/metadata/taiwan-civic-policy-v1.json) · [價值對齊政策](full-v4/metadata/value-alignment-policy-v1.json) · [來源工具契約 HOLD](full-v4/metadata/source_tool_contract_holds_v1.json)

本 release 為多來源資料，逐列保留原授權與署名；**沒有整包改授權為 Apache-2.0**。SQuAD 衍生資料保留 CC BY-SA 4.0，新增 authored semantic cases 為 Apache-2.0。`license: other` 表示請依各來源的實際授權使用。

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
