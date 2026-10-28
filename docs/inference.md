# Harness 與 inference

`barbet_inference` 是可安裝的 Python API 與 CLI。它把技能選擇、文件檢索、
模型生成、公開測試與有限次修復接起來，不需要先準備 benchmark 或答案檔。
預設採用已在驗證集選出的路由設定；自訂問題、技能及參數的效果仍須另外評估。
`recommended` 源自 MiniCPM 的驗證集，不代表每個模型的最佳策略。

目前提供 MiniCPM5-1B 與 Qwen3.5-9B 的固定版本預設，支援本機 vLLM，或已有
vLLM 相容服務的 `/v1/completions`。不是任意 chat-only API 的通用轉接器。
不訓練模型，不啟動大模型 fallback。

## 安裝

在 repository 目錄內，使用獨立 Python 環境；測試環境為 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[vllm]'
```

`vllm` extra 固定本次驗證的 vLLM 0.23.0、transformers 5.12.1。本機生成需要
相容的 GPU／CUDA 環境。若只連遠端推論服務，安裝 `pip install -e '.[tokenizer]'`；
仍需載入與伺服器一致的 tokenizer，才能準確渲染模板與限制上下文。
只讀技能或接自己的生成後端，可用不含額外依賴的 `pip install -e .`。

Python 執行回饋需要 Linux 與 `libseccomp2`（Debian/Ubuntu：
`sudo apt-get install libseccomp2`）。缺少隔離能力時會回傳失敗，不會降級成未隔離執行。
可用 `--no-execution` 關閉執行與修復。安裝包包含全部技能定義與短卡，
安裝後不必依賴原始 checkout 的 `skills/` 路徑。

## 一個問題

```bash
barbet-infer run --family mini \
  --prompt '一本筆記本 35 元，買 4 本，付 200 元，應找回多少元？' \
  --contract number \
  --output outputs/answer.jsonl \
  --trace outputs/calls.jsonl
```

預設下載並使用 `profiles.json` 中固定的模型版本。`--model /path/to/snapshot`
可指定本機模型；自訂 HF ID 可配 `--revision COMMIT`。`--local-files-only`
要求使用已快取的 snapshot。家族參數控制原生聊天／思考及工具介面，不能當作
任意模型都相容的保證。GPU 選擇交由 `CUDA_VISIBLE_DEVICES`。

`--prompt` 會附上該輸出契約的格式要求。JSON/JSONL 請求保留呼叫端提供的 prompt，
應自行寫清楚答案形式；程式的 `public_tests` 會另行附在第一輪 prompt，
讓模型在執行前就能看到公開例子。`output_contract` 可為：

| 契約 | 自動路由 |
| --- | --- |
| `text` | 自由文字；不注入閱讀短卡或原文抽取限制，可指定寫作／翻譯等完整技能 |
| `short_answer` | 中文詞彙檢索、短卡與原文答案約束 |
| `number` | 算術短卡、直接推理 |
| `python` | 程式短卡、隔離執行公開測試、失敗後修復 |
| `tool_call` | 原生工具呼叫、schema 檢查；不執行外部服務 |
| `choice` | 選擇題短卡、閉卷作答 |

## 批次問題與文件

```bash
barbet-infer run --family mini \
  --request-file examples/inference/requests.jsonl \
  --documents examples/inference/documents.jsonl \
  --batch-size 5 \
  --output outputs/batch.jsonl \
  --trace outputs/batch-calls.jsonl
```

範例包含閱讀、數學、程式、工具與選擇題。支援 `.json` 的單一物件或陣列，
以及 `.jsonl` 的逐行物件。輸出與 trace 都要求新路徑，不會覆寫既有檔案。

請求只接受 `id`、`prompt`、`output_contract`、`tools`、`document_ids`、
`public_tests`、`setup`。`gold`、`answer`、`hidden_tests` 等未知欄位會被拒絕。
公開測試及 setup 由呼叫端提供；隱藏答案與測試不應放進推論請求。
`public_tests` 與 `setup` 只接受於 `python` 契約，避免把未執行的檢查當成通過。

文件包含 `id`、`text`、`source` 及可選的 `title`。閱讀請求未指定
`document_ids` 時會搜尋本次提供的全部文件；若指定，則只使用指定候選包。
保留完整來源、雜湊與因預算省略的文件 ID，使用目標 tokenizer 計算長度。
目前不自動下載網頁，也不把文件中的指令當作使用者指令。

工具採用 `{"type":"function","function":{...}}` 或直接的 function 定義。
schema 支援 `type`、`properties`、`required`、布林 `additionalProperties`、
`items`、`enum`、`description`、`title`。未實作的限制（例如 `minimum`、
`anyOf`）會被拒絕，不會默默忽略後宣稱通過。

## 指定技能與策略

```bash
barbet-infer list-skills --scope math

barbet-infer prepare --family mini \
  --prompt '每箱 12 瓶，買 3 箱共有多少瓶？' --contract number \
  --skill arithmetic-modeling \
  --output outputs/prepared.jsonl
```

不指定 `--skill` 時採用短卡及契約路由。指定時使用技能庫完整程序，
保留適用／不適用條件、失敗處理、驗證與版本雜湊，取代自動短卡；可重複
`--skill` 組合技能。`--library PATH` 可選另一個技能 repo 或 `skills/` 目錄。
完整技能較長，超出必要輸入預算時會報錯，不會靜默截斷。

寫作、翻譯或一般對話可使用 `--contract text`，例如：

```bash
barbet-infer run --family mini --contract text \
  --prompt '請將這句話改寫成自然的台灣用語：點擊鏈接獲取更多信息。' \
  --skill taiwan-text-rewrite --output outputs/rewrite.jsonl
```

`text` 不會強制最短答案或照抄來源。需要從來源抽取事實時，使用 `short_answer`。

`prepare` 只載入 tokenizer，輸出實際 prompt、來源及 token 數，不載入生成模型。
自訂策略可參考 `examples/inference/policy.json`，透過 `--policy-file PATH` 載入。
`--policy bare` 可停用自動短卡、檢索排序及執行回饋，方便同一應用自行比較。

資源控制：`--input-tokens`、`--max-output-tokens`、`--continuation-input-tokens`、
`--max-model-len`、`--max-calls 1|2|3`。預設單輪輸出上限 4,096，
最多三次生成；只有需要修復時才追加呼叫。強制 CLI token 限制也適用於契約覆寫設定。
完整 prompt 加輸出預算不得超過 backend context ceiling。

## HTTP 推論服務

伺服器與客戶端必須使用相同模型／tokenizer 版本。以下以已存在的服務為例：

```bash
barbet-infer run --family mini --backend http \
  --base-url http://127.0.0.1:8000/v1 \
  --served-model mini \
  --request-file examples/inference/requests.jsonl \
  --documents examples/inference/documents.jsonl \
  --output outputs/http.jsonl --trace outputs/http-calls.jsonl
```

需要驗證時，從 `BARBET_API_KEY` 環境變數讀取；用 `--api-key-env NAME` 更換變數。
金鑰不寫進結果或 trace。傳送的是已渲染的 token IDs，使用 vLLM 的
`stop_token_ids`、`skip_special_tokens` 等 completion 擴充欄位。
客戶端不會再次套用聊天模板，也不自動重試 HTTP 請求。

HTTP timeout 的遠端完成狀態與成本可能未知，會明確報錯。服務必須回傳實際
token usage；不能缺少 usage 卻假裝成本為零。客戶端 metadata 會標記
`server_revision_verified: false`，因為 API 模型名稱無法證明遠端權重版本。

## Python API

```python
from barbet_inference import Document, Harness, make_request
from barbet_inference.backends import VLLMBackend

backend = VLLMBackend('/path/to/MiniCPM5-1B-snapshot', family='mini')
harness = Harness(backend, family='mini', documents=[
    Document('tower', '青塔不是三十層，而是四十層。', 'local:tower')
])
result = harness.run(make_request('青塔共有幾層？', 'short_answer'))
print(result['prediction'])
print(result['validation'])
```

`Harness.run_many(requests)` 依輸入順序產生結果；模型 backend 在同一次使用期間重用。
API 結果保留完整 trace；CLI 預設輸出較精簡的結果，使用 `--include-trace` 可保留它。
`on_call` callback 在 host 執行前接收每個已返回的 completion，適合自行持久化。

自己的 backend 需提供 `tokenizer`、`terminal_strings`、`max_model_len`、
`max_batch_size`、`metadata`，以及 `generate(list[Generation]) -> list[Completion]`。
每個 completion 必須與輸入順序對應，包含真正的 token 數與時間。
核心不依賴 vLLM；測試用 backend 可在無 GPU 環境執行。

## 資料集與關係記憶介面

Barbet 訓練請使用 HF 預設的 `tokenized` config，直接保留 `input_ids` 與 assistant-only `labels`；MiniCPM tokenized config 已移除。載入範例與 masking 說明見 [Barbet 訓練格式](reports/barbet-only-v1/index.html)。以下 `records` adapter 用於建立推論請求。

HF 的 `records` config 保留 messages、工具 schema、完整 native 文字、
target spans 與來源 metadata。用 `prepare_dataset_request` 建立不包含目標答案的
請求；不要把整列 `source_refs_json` 放進 prompt，因為其中包含資料建置的計算軌跡。

```python
import json
from barbet_inference.dataset_adapter import prepare_dataset_request
from barbet_inference.relational_dataset import (
    restore_relational_memory, relation_tool_callbacks,
)

# row 是一筆已下載的 records 資料；從第一個 supervised assistant 回合開始。
messages = json.loads(row['messages_json'])
first_target = next(i for i, m in enumerate(messages)
                    if m['role'] == 'assistant' and m.get('loss'))
request = prepare_dataset_request(
    row, model='your-served-model', target_turn=first_target,
)

# 僅用於具有 registered_relation_snapshot 的關係記憶資料。
memory = restore_relational_memory(row)
# 工具呼叫示例需選取有宣告 relation_lookup schema 的 tool_use 資料列。
handlers = relation_tool_callbacks(row)
```

預設 `target_turn` 是最後一個 supervised assistant 回合，適合評估給定先前歷史後的
答案生成；要從頭量測工具選擇與檢索，應像上例指定第一回合。不能把已記錄的未來工具
觀察先交給模型，再稱為模型自行完成檢索。

`restore_relational_memory` 核對 snapshot 雜湊及有界 facts。
`relation_tool_callbacks` 只提供 `relation_lookup`，由應用程式明確註冊到
`ContextAgent` 的 `handlers`，並提供該筆資料的對應 `tools` schema。
reading 列已在上下文提供 facts，可單獨使用 `memory.verify`；它沒有宣告工具 schema 時，不應把 callback 加入工具呼叫介面。
文件內的工具名稱不會自動變成可執行函式。查詢保留方向、完整 URI 與不同終點語意；
超過預算會報錯，不會回傳裁切後的部分結果。查不到只表示此有限資料沒有匹配紀錄。

應用程式也可用 `memory.verify(public_query, candidate)` 作為 `ContextAgent`
的 verifier，或搭配 `RefinementLoop` 進行最多三次候選生成。這個 callback 根據
題目公開的路徑與已註冊 facts 重算，回饋綁定實際候選的 SHA-256，不讀 benchmark
隱藏答案。它只驗證有限紀錄查詢，不證明來源敘述是現實世界的事實。重複候選、
耗盡步數、未完成生成與驗證失敗都會保留明確狀態。

目前新增的圖譜與詞典資料是正確查詢和真實 observation 軌跡；沒有錯誤及修復回合的
列不算 iterative-refinement 樣本。模型是否能學會這些程序，仍須以訓練後的獨立任務測量。

## 如何解讀輸出

`prediction` 是模型結果；`validation` 可能標記 `format_valid`、`schema_valid`、
`public_tests_passed`、`execution_failed`、`unfinished_thinking` 或 `not_checked`。
`passed` 只對該檢查範圍有效，`semantic_correctness` 保持 `unknown`。
程式沒有公開測試時，即使執行成功也只標記 `execution_succeeded`。

`calls`、`prompt_tokens`、`output_tokens` 包含修復；trace 記錄每次真正回傳的生成。
同一 batch 的時間為攤提吞吐成本，HTTP 請求時間含網路與服務等待，不能混稱單一模型延遲。
沒有因為 JSON 合法或公開測試通過就自動宣稱完整任務正確。

本套件重用 `testtime/` 中已封存的核心，但新增 CLI、自訂文件與完整技能功能本身
不是先前 benchmark 的新分數。原實驗 runner、主評分器及封存檔保持原樣；
過去的能力結論見[實驗報告](test-time-study.html)。


## 台灣公民價值與 Barbet 續訓格式

應用程式明示採用這個資料集的公民價值設定時，可以傳入
`skills=["taiwan-civic-values", "grounded-evidence-answer"]`。範圍是台灣主權、
人民自決、民主、自由、人權、法治、平等及多元，不預設支持特定政黨。
此技能須與有版本的原始文件一起使用；它不是法律適用、外交承認或歷史事實的證據。

對使用本資料集 portable JSON 格式續訓的 Barbet，設定
`ContextAgent(..., prompt_format="barbet_portable_v1")`，或從資料列建立
`prepare_dataset_request(row, model="your-barbet", mode="barbet_completion")`。
保留完整問題、可用記憶與 assistant-only labels，不再套第二個 chat template
或多加 BOS。原生 checkpoint 的 Jinja 與這個訓練格式不同。

訓練端使用 `barbet_inference.training.BarbetSFTCollator`；labels 不預先位移，
右側 padding 的 label 為 -100，模型內只位移一次。不要強制在 padding mask 上
只用 Flash kernel；實際短 batch 測試以預設可處理 mask 的 SDPA 通過。
跨樣本 packing 的 attention 與 Mamba state reset 尚未驗證，預設關閉。

新增公民資料有 933 筆，以來源條文摘錄與明示規範案例為主；它們不代表已驗證
模型的政治價值泛化。HF 的 `curriculum/taiwan-civic-v1.parquet` 可依 record_id
辨識分層與 split。這個索引、source_refs_json、稽核報告與答案標記都屬於
訓練／抽樣 metadata，不能當成模型的 RAG 證據。
