# Barbet 語義課程 v3：資料重整候選版

日期：2026-11-04。此版本實作資料與工具基礎，沒有啟動 SFT、RL 或模型升級。

HF 候選分支為 [`semantic-curriculum-v3-20261104`](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/semantic-curriculum-v3-20261104)，原發布版固定在 `b88f4ff7171bb6fb8c1b775ae21de814ea2123fb`。舊資料與 checkpoint 保留。新資料必須依 config 分別載入，不可把整個 source bank 當成已重新接受的 TRAIN。

## 實際交付

| 產物 | TRAIN | DEV / validation | 私有資料 |
|---|---:|---:|---:|
| 自然閱讀 learner views | 111,259 | 2,866 | 無新增 |
| 語義 learner views | 6,286，2,000 roots | 629，200 roots | calibration 628、TEST 629，各 200 roots |
| 來源課程索引 | 原始 TRAIN 9,817,673 | 原始 validation 201,294 | 不新增接受資格 |

公開 learner views 合計 **121,040**，106,824,940 input tokens、2,042,774 supervised tokens。語義 TRAIN 單次曝光為 **462,008 supervised tokens**；它尚不足以無重播完成原提案的 2M / 8M 觀察點。自然閱讀是既有來源重整，不能算新增 114,125 個獨立來源問題。

來源索引完整覆蓋 10,018,967 筆目前有效資料，另排除 310,662 筆既有 hold。5,414,792 筆能由明確 query 判定有限圖查詢；其餘 4,604,175 筆不能只靠 metadata 推定語義能力。每筆索引 `eligible_for_automatic_training=False`，不代表重新核准原 targets。

## 資料如何改變

- 自然閱讀移除強制 JSON 與無必要工具包裝，保留原始問題、文章、來源 gold 和原 article split。75,609 筆可回答、38,516 筆來源不支持答案；一筆超預算 view 隔離，沒有截斷補齊。
- 文件在問題前後皆有覆蓋；自然閱讀 front / back 為 56,950 / 57,175。57,027 筆有 evidence skill，57,098 筆不帶 skill。這些是不同題的路由比例，**不是同題 skill 效果實驗**。
- 語義核心為比較、排序、binding、單位、有效規則版本、證據狀態、執行狀態，共七個原子家族。每個 root 有 original、會改答案的 counterfactual、保持答案的 invariance；invariance 也改變記憶位置與列順序。
- 比較題含等於門檻的值；排序題在選定順位有平手；虛構規則的 v1/v2 定義與目前有效版本均可變，不把版本名稱當固定答案。
- binding roots 另外加入撤提示視圖：只給顯示名稱與欄位意思，不在問題中給正確 row ID 或 field code。公開新增 315 views，沿用既有 roots，沒有灌大獨立任務數。
- 工具觀察來自真實確定性 executor 或明示的版本化本機 state simulator。Host 不替模型改 operator 或選參數；沒有執行來源提供的任意程式碼。

Qwen 問法生成與檢查都留存首次 receipts，不能單獨決定 target 正確性。核心 140 個候選 frame 接受 133、隔離 7；撤提示 frame 共 12 個，接受 6、隔離 6。額外逐句覆核（由本次 Codex 執行，並非人工審標）曾抓出 teacher 漏判的「另報 JSON 型別」需求漂移，相關候選未再送審。後續新的審核 protocol 先用正負控制確認，再處理全新候選，舊 hold 未解除。

## 品質證據與限制

1. 全部 121,040 筆公開資料逐筆重建 text、Pangolin input IDs、unshifted labels、assistant EOS 與 answer-free decision prefix。
2. 全部公開 decision prefixes 量測 Qwen portable Chat / Completion token 數；各路徑使用真實 `/tokenize` 控制核對 token IDs，皆保留 4,096 output tokens。這是輸入傳輸檢查，不是用 Qwen tokenizer 訓練 Barbet。
3. 全部 8,172 筆語義資料（含私有 calibration / TEST）核對 root/source/frame 切分與完整變體組；重播 7,355 次工具呼叫。7,059 筆使用獨立數值、型別或狀態 oracle；1,113 筆為來源與證據狀態一致性檢查。
4. 程式完整測試 316 項通過。私有 oracle、calibration、TEST、隱藏答案未放入 HF 候選分支。
5. v26 PPL 規則保留，但新 targets 為獨立合格的 deterministic singleton，`v26_ppl=null`。沒有捏造 PPL，也沒有把低 PPL 當正確性或課程價值。

目前仍是短任務試作，最大 total budget 8,192，沒有 128K–1M 新 SFT 課程。自然閱讀僅驗證來源 gold 與 span，沒有證明世界事實正確或教學增益。七個 atomic program graphs 跨 split 共用，因此不能宣稱未見組合遷移；語義 frame 數遠少於 roots，這個限制有明列。516 題遠端 DEV 原始輸出仍不可得，未聲稱完成其全部近鄰除汙。

## Skill、runtime 與 lineage

補強既有七項 skill 的比較邊界、平手、型別、缺值、版本、證據不足與狀態契約，總數維持 65。`barbet_inference/barbet_backend.py` 以固定權重／程式雜湊載入真實 Barbet，保留必要 FP32 參數。

`OpenFormosa/barbet-1b-base@004ef70881dcb87f1b423951bd418f1748db8b13` 的 release manifest 宣告 `supervised_fine_tuning`。權重核對為 29 blocks、22 attention / 7 Mamba、原生 1,048,576 context；BF16 1,118,798,760 elements，FP32 336 elements。先前 `4dbb352...` 亦已做 attention / surface-binding 適配，原始預訓練 lineage 未確認，故 training parent 保持未選。

真實 backend smoke 只測 14 / 290-token prompts，各生成 12 tokens；兩次都碰到輸出上限。3 個 cache 步驟 top-1 相同，但最大 logits 差異為 0 / 0.15625 / 0.21875。這不能當成完整數值 parity、自然停止、任務能力或 1M 生成驗收。

## 可執行介面

`python -m barbet_pipeline --help` 目前提供 `audit`、`build-data`、`validate-data`、`run-barbet`。訓練、promote 與 Megatron parity 尚未實作為通過驗證的 CLI，不使用空殼名稱宣稱存在。

以下路徑由操作者指定，`TOKENIZER_JSON` 必須是固定 SHA 的 Pangolin tokenizer 檔案：

```bash
python -m barbet_pipeline validate-data \
  --manifest PATH_TO_LOCAL_SUMMARY_JSON \
  --tokenizer TOKENIZER_JSON \
  --output NEW_VALIDATION_RECEIPT_JSON

python -m barbet_pipeline run-barbet \
  --model-folder HASH_VERIFIED_RELEASE_DIRECTORY \
  --model-manifest configs/semantic_v3_model_manifest.json \
  --episode PUBLIC_EPISODE_JSON \
  --environment PUBLIC_ENVIRONMENT_JSON \
  --output NEW_TRAJECTORY_JSON
```

`run-barbet` 必須在具備發布模型依賴的環境執行。Native plain chat 與 tool harness 的 `barbet_portable_v1` 明確區分；回饋使用 `TOOL_OBSERVATION` user continuation，與資料 renderer 相同。部署 harness 不匯入 private oracle；`answer_proposed` 不是答案正確的宣稱。任意 code sandbox 不屬於本輪已完成 executor。

## 下一階段門檻

50/50 supervised-token 配比、2M / 8M 預算與非劣性界線維持 proposed。A ordinary 有 2,000 views、148,328 supervised tokens；B contrasts + fading 有 6,286 views、462,008 supervised tokens。它們是資料選單，沒有假裝已按同預算跑完 A/B。

正式訓練前仍需固定合適 parent、完成 Megatron converter / loss / CP / recompute / export parity、補齊一般指令／數學／程式保留課程及未見組合／自然來源 transfer。之後才做 parent/new × bare/same-harness 四格及逐項保留驗收。資料量、loss、格式、工具成功都不能代替這些結果。

原 Taiwan civic / value alignment 政策和 933 筆來源保留在舊來源池，支持台灣主權、人民自決、普世人權並保留事實及引述界線。本輪沒有新增政治樣本或聲稱重新訓練完成價值對齊。SQuAD 改作沿用 CC BY-SA 4.0；新 authored semantic cases 使用 Apache-2.0；其他來源沿用其各自授權。

已驗證 teacher API 呼叫 32 次，48,384 prompt / 26,653 completion tokens。8-GPU teacher server 生命週期約 0.903 小時，配置時間約 7.223 GPU-hours（含載入及閒置，不是 GPU 利用率積分）。訓練 GPU-hours 為 0；teacher 已停止，全部本次持有程序退出。
