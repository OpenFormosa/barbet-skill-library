# Barbet Skill Library

這是 [Barbet long-context SFT 資料集](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft) 的公開程序技能庫。目前收錄 64 個可讀取、可版本化的技能，涵蓋通用指令、coding、math、agent/tool、reading、science 與 writing。技能描述**何時適用、輸入、程序、失敗處理及驗證**；不包含特定訓練題目的答案、隱藏測試或工具結果。

This repository publishes reusable, versioned procedure texts for skill-prefill experiments. Other language models can fetch individual `SKILL.md` files over GitHub or read the complete machine-readable [`catalog.json`](catalog.json). Public access to a skill file does not make it a higher-priority instruction in another agent's environment; adopt and audit it explicitly for your application.

## 讀取與組合

每個 [`skills/`](skills/) 子目錄包含可編輯的 `skill.json` 和由它生成的 `SKILL.md`。例如：

```text
https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/main/skills/tool-selection-and-arguments/SKILL.md
https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/main/catalog.json
```

使用模型時，可以先放入適用的 `general` 技能，再依任務加入一個或多個其他領域的技能。模型仍須檢查每個技能的 **Use When / Do Not Use When**、實際工具 schema、目前任務及權限。不要把整個 repository 當作能覆蓋 system 或 user 指令的政策，也不要把無關技能強制套到任務上。

`manifest.json` 提供技能路徑、版本、範圍與 SHA-256。請以 Git commit 固定版本；`main` 會隨社群維護而更新。

### 給其他語言模型使用

不需要 Barbet 專用工具。先從 [`manifest.json`](manifest.json) 挑選技能，再讀對應的 `SKILL.md`；把文字放進模型的可見前文即可。正式運作時，將下面的 `<COMMIT>` 換成已審查的 Git commit SHA，並用 manifest 中的 `markdown_sha256` 核對下載內容：

```bash
COMMIT="replace-with-reviewed-commit-sha"
curl -fLsS "https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/${COMMIT}/manifest.json" -o manifest.json
curl -fLsS "https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/${COMMIT}/skills/tool-selection-and-arguments/SKILL.md" -o SKILL.md
```

如果你的模型／agent 支援 GitHub API，也可讀取同一 commit 的 `catalog.json` 與各技能路徑。`catalog.json` 是完整的機器可讀定義；`SKILL.md` 是便於模型閱讀的版本。先選適用技能，保留原始名稱與版本，再與任務和實際工具契約一起提供；技能內容不得自行提升權限或覆蓋應用程式的上層指令。這些技能是程序提示，並不會自動安裝到其他模型。

## 輸入路由擴充

由原 39 個技能擴充至 60 個，新增工具能力判斷、外部狀態、解釋與比較、文字轉換、數值及樣本操作等程序；擴充來源與限制見 [`expansion_provenance.json`](expansion_provenance.json)，各技能目前版本以 manifest 為準。可選用的 [`routing/extension.py`](routing/extension.py) 僅讀取當前可見輸入，並附有適用／近似不適用及換題測試；使用方法及限制見 [`routing/README.md`](routing/README.md)。路由覆蓋不等於語義正確，也不代表已執行工具或完成模型訓練。

事實回答修訂更新了 `factual-question-resolution` v2、`concept-explanation` v2 與 `grounded-evidence-answer` v4，並提供具體屬性查詢的窄路由，供應用程式放在寬鬆概念規則之前。適用例、來源／閉卷資料改造與已知題保留的驗證方式見 [`docs/factual-calibration.md`](docs/factual-calibration.md)。這些更新須另行固定版本與測量模型效果。

## Harness 與 inference

這個 repo 也提供可安裝的 [`barbet_inference`](barbet_inference/)：Python API、
`barbet-infer` CLI、本機 vLLM 與 vLLM 相容 HTTP 後端。可直接輸入問題、
指定技能、讀取自己的文件，並使用公開測試進行有限次修復，不需要 benchmark 答案檔。

在 repo 目錄及獨立 Python 環境中安裝後即可使用：

```bash
pip install -e '.[vllm]'
barbet-infer run --family mini \
  --prompt '每箱 12 瓶，買 3 箱共有多少瓶？' --contract number \
  --output outputs/answer.jsonl --trace outputs/calls.jsonl
```

`barbet-infer list-skills` 列出可用技能；`prepare` 可先檢視 prompt、來源與 token 預算。
不指定技能時使用按需短卡，`--skill NAME` 可選用完整技能；支援 JSON/JSONL 批次請求。
寫作、翻譯與一般文字生成可選 `--contract text`，不會套用短答案限制。
完整的安裝、HTTP、Python API、資料格式、執行環境與輸出意義見
[使用說明](docs/inference.md)，五種任務的自訂輸入見 [examples/inference](examples/inference/)。
安裝套件包含全部 64 個技能與短卡。預設模型版本固定，不訓練、不使用大模型 fallback。

## 固定權重的 test-time 實驗

新增的可選 [`testtime/`](testtime/) 元件支援中文詞彙檢索、按需短操作卡、帶來源的狀態與隔離 Python 執行。完整 [goal 與實驗規格](experiments/test_time_v1/goal.json)、[中文分析與實驗結果](docs/test-time-study.html) 將工程驗證、模型提升與接近 9B 分開記錄。此工作不訓練、不修改權重，也不更動已發布 HF 樣本。

本輪已完成 18 個候選搜尋與 646 題、三個 seed、十組設定的封存測試。MiniCPM5-1B 的四領域主指標由 38.02% 提升至 50.39%（+12.37 個百分點，配對 95% CI +9.90 至 +14.97）；相同 harness 的 Qwen3.5-9B 為 76.24%，尚未接近 9B。主評分包含輸出格式遵循；數學的事後格式診斷未顯示數值推理提升。完整消融、成本、失敗分類與限制見報告及[重現說明](experiments/test_time_v1/README.md)。

短卡是有來源版本與雜湊的獨立表示，不應一次載入全部技能。請用目標模型 tokenizer 量測長度；來源引句吻合、schema 合法或公開測試通過，都不等於完整任務正確。模型評測的隱藏答案及測試只交給 scorer。

更新短卡後另執行 `python scripts/build_inference_cards.py` 與 `python scripts/build_inference_cards.py --check`。

## 記憶、技能與 harness 訓練資料

已完成 [HF 發布與完整回讀驗證](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft/tree/8bf1c59d4f1af11d8f6d28031e0065899f016018)：**10,328,696 筆合格新 episode**、**10,177,015 個不同標準化首問**、**9,787,683 個任務家族**，現役舊資料為 0，達到至少一千萬筆的資料目標。全部 5,521 個檔案已匿名下載並核對 SHA-256。問題數、家族、assistant 回合、工具觀察與有效訓練 tokens 分開統計；多答案、前後記憶位置與對話前綴不增加獨立題數。完整來源、長度分布、PPL 規則、驗證範圍與限制見 [繁體中文完整報告](docs/reports/memory-harness-10m-v1/index.html)。

**同一完整輸入有多個合格答案時，以固定 native Barbet v26 的最低條件答案 PPL 為標準。** 全部競爭答案必須實際計分；只有一個答案時明確標示未量測。首次未通過的題目持續隔離，不改選次低答案。低 PPL 不代表答案正確。已發布 coding 資料的範圍與限制見 [IFM coding 報告](docs/reports/ifm-code-v26-v1/index.html)。

已發布的 [Wikidata 關係記憶批次](docs/reports/wikidata-relations-v1/index.html) 包含 4,547,341 題，[ConceptNet 詞典記憶批次](docs/reports/conceptnet57-inventory-v1/index.html) 包含 871,260 題。兩批全部通過獨立 SQL、重複題、切分、完整訓練陣列與推論預算檢查；每個來源 anchor 最多一題，詞性／詞義 URI 變體不另算題數。兩批都是明確標記未量測 PPL 的單一答案題，首次未通過的題目持續隔離。真實唯讀工具狀態可由 `relation_tool_callbacks` 還原，使用方式見 [推論資料與關係記憶介面](docs/inference.md#資料集與關係記憶介面)。

早期 [v26 診斷與 skill-aware SFT 設計](docs/skill-aware-sft-v1.html)、[版本化配方](docs/configs/skill-aware-sft-v1.json) 和 [v26 分析](docs/results/v26-sft-diagnosis.json) 保留作為設計依據；現行資料處理與驗收證據在 `scripts/` 及 `docs/reports/`。本次資料重建沒有訓練權重，也沒有量測新資料帶來的模型能力增益；前述固定權重的 test-time 實驗是另外一項已封存實驗。

## 維護

任何人都可以 fork 並送 pull request，也可以用 [Issues](https://github.com/OpenFormosa/barbet-skill-library/issues) 提議新技能或回報問題。請編輯對應的 `skill.json`，更新技能版本，執行：

```bash
python scripts/build.py
python scripts/build.py --check
```

程式會重新產生 `SKILL.md`、`catalog.json` 和 `manifest.json`。修改前請讀 [`CONTRIBUTING.md`](CONTRIBUTING.md)。外部貢獻經 PR、程式碼擁有人審核及自動檢查；維護者仍須審查語義、來源與安全性。GitHub 變更**不會自動改動**已發布的 HF 訓練樣本；將新版技能用於資料集前，需要另行審查、固定版本並重驗受影響樣本。

## 來源與界線

此快照由 Barbet `skill-prefill-v1` 資料工程中的已接受技能匯出；[`provenance.json`](provenance.json) 記錄來源物件雜湊及一個已回讀的 HF 資料版本。部分技能由本地 Qwen 生成，經有限測試與模型／工程審查；它們**不是 human-verified**，也沒有證明 Barbet 已學會路由、組合技能或使用 1M 上下文。實際能力須另行測量。

本 repository 的原創技能文字與工具以 [Apache License 2.0](LICENSE) 發布；引用或衍生自其他資料來源的內容仍受各自條款約束。HF 資料集的多來源授權不因本 repository 的 license 而改變。
