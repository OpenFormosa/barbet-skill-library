# 完整資料重處理 v4

本版處理既有 active release 的全部 10,018,967 筆資料，涵蓋 1,504 個來源分片。
每筆都產生新的 records／Pangolin tokenized 表示或明確隔離處置；不以來源索引、抽樣或第一批資料代替全量交付。
原 main、v3 release、checkpoint、來源快照與既有 holds 保留。

## 實際檢查

1. 核對來源 records 和原 tokenized 檔案 SHA-256；重新構建原始文字、spans、IDs 和 labels。
2. 套用來源工具合約、內容與價值政策。有限圖答案另用獨立演算法驗證並真實回放工具；QA 與 NI 對照固定原始來源；其他來源明列尚未獨立驗證語義。
3. 更新單一合格答案的 skill 前綴。SQuAD 單一答案可採用 v3 已核對的自然答案；所有多候選 frozen-v26 輸入與勝出答案保留原樣。
4. 逐一構建每個 accepted assistant decision 的無答案 Chat／Completion 前綴，測量 Pangolin 與實際 Qwen tokenizer，預留 4,096 output tokens，完整上限 131,072，不截斷。
5. 寫出固定 schema 的 Parquet，再從落盤資料逐筆重算 token arrays。每筆 parent 都有處置紀錄。
6. 上傳後匿名下載完整 bytes，比對 SHA-256，才釋放本次新產生的本地 Parquet。原始來源與 immutable receipts 保留。
7. 獨立掃描全部新 records，統計各來源、領域、品質範圍、split、長度與 supervised-token exposure；完整總帳和所有來源／發布 receipts 必須一致。

## 三種品質範圍

`verified` 的精確驗證能力記於每筆 `verification_scope`。圖答案的獨立計算檢查、SQuAD gold/span、Hotpot gold/support IDs、官方逐字引述，不能被視為相同強度的證據。`semantic_correctness_independently_certified` 只對獨立有限圖驗證設為 true。

`source_reference` 表示核對過 pinned reference 與題目輸入，或保留既有作者規範立場的接受紀錄；它不是獨立語義審查。

`reference_only` 只通過完整工程與來源追溯檢查，答案語義未獨立驗證。它另版提供，不自動進入訓練。完整處理不等於把全部舊答案再次核准。

所有品質範圍分別提供 records／tokenized configs。預設 config 是完整 `corpus` 處置表，避免不小心把整個未平衡來源池當成 TRAIN。records 和 tokenized 是同一批題的兩種表示，不是兩倍資料量。

## 推論介面

```python
from barbet_pipeline.full_requests import prepare_full_request

body = prepare_full_request(row, model="configured-server-model", mode="chat")
body = prepare_full_request(row, model="configured-server-model", mode="completion")
```

可指定 `target_turn` 選擇已量測的決策點。介面只序列化該 assistant target 之前的 messages／工具定義，不包含答案、private metadata、execution receipts 或未來結果。工具觀察採明確 `user` continuation 和 `TOOL_OBSERVATION:` 前綴，與資料 renderer 相符。Completion 不加第二層 template 或額外 special tokens。

Qwen tokenizer 只作 portable transport 檢查；Barbet 訓練仍使用 Pangolin，沒有 MiniCPM tokenizer。這些格式及預算檢查不是 Barbet vLLM 或 1M serving 證明。

## 不覆寫失敗與修正紀錄

首輪程式對少數 ranked SQuAD 題目嘗試進入 singleton 改寫路徑，原有保護拋出 `Cannot change a ranked multi-candidate input`。98 個完整來源分片已通過；30 個未完整來源分片的輸出歸檔，沒有算成接受或來源拒絕。

`full_corpus.py` 保持凍結。`full_corpus_ranked.py` 只在 SQuAD 分支條件增加 singleton 限制，讓 ranked 輸入走原樣保留路徑。三個真實來源控制證明：原錯誤可重現、修正後原輸入／IDs／labels 不變、原 singleton 和其他來源輸出不變。修正與保留的首輪 receipt 清單另有 hash-bound amendment；未覆寫或放寬原 gate。

執行器最多 128 CPU workers、條件式預取下一組最多 64 分片、單一 remote publisher，保留至少 50 GiB 磁碟空間。沒有租用運算、呼叫付費 API 或啟動訓練。

## 驗證與發布

- 全部單元測試：332 passed，另 112 subtests passed。既有 fork 測試產生 34 個 Python deprecation warnings。
- 真實 served `/tokenize`：四個 Chat／Completion ID 控制通過；沒有生成答案或訓練。服務已停止。
- `collect_complete_release` 必須核對所有來源、資料／tokenized 配對、完整處置數、獨立統計與逐檔匿名回讀。
- 發布最後仍檢查固定 revision 的完整遠端樹身分，以及每個 config／split 的匿名實際載入。這是介面控制，不用來替代已執行的全量 row 檢查。
- `FULL_V4_STAGING_INCOMPLETE.txt` 只在完整檢查後與 completion receipt 同一 commit 移除。

發布後可由 `full-v4/release-manifest.json`、`full-v4/publication-complete.json` 與 `full-v4/report.html` 核對實際總量及不可變 commit。

## 能力與資料使用界線

圖操作仍占多數；完整重新輸出沒有解決廣泛能力課程的所有缺口。不可均勻混合所有資料，再用 source loss 下降推論通用能力進步。exposure policy 提供實測 supervised-token 分布，50/50 一般／語義混合仍只是一個待驗證建議。

現有七類語義對照、invariance 與撤提示資料保留為獨立 config；不把模板複製成百萬新任務。v4 已包含 SQuAD parents，不能再拼接舊自然閱讀 config 重複計數。私有 calibration、TEST、oracle 沒有公開。

沒有新的訓練、可學習性、未見組合遷移或同 checkpoint 保留成績；沒有新 1M 課程或 MiniCPM／Qwen 能力等價宣稱。這不影響全量資料處理必須完成，但必須保留資料工程完成與模型能力成功之間的證據界線。
