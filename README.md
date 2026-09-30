# Barbet Skill Library

這是 [Barbet long-context SFT 資料集](https://huggingface.co/datasets/OpenFormosa/barbet-long-context-sft) 的公開程序技能庫。目前收錄 39 個可讀取、可版本化的技能，涵蓋通用指令、coding、math、agent/tool、reading、science 與 writing。技能描述**何時適用、輸入、程序、失敗處理及驗證**；不包含特定訓練題目的答案、隱藏測試或工具結果。

This repository publishes reusable, versioned procedure texts for skill-prefill experiments. Other language models can fetch individual `SKILL.md` files over GitHub or read the complete machine-readable [`catalog.json`](catalog.json). Public access to a skill file does not make it a higher-priority instruction in another agent's environment; adopt and audit it explicitly for your application.

## 讀取與組合

每個 [`skills/`](skills/) 子目錄包含可編輯的 `skill.json` 和由它生成的 `SKILL.md`。例如：

```text
https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/main/skills/tool-selection-and-arguments/SKILL.md
https://raw.githubusercontent.com/OpenFormosa/barbet-skill-library/main/catalog.json
```

使用模型時，可以先放入適用的 `general` 技能，再依任務加入一個或多個其他領域的技能。模型仍須檢查每個技能的 **Use When / Do Not Use When**、實際工具 schema、目前任務及權限。不要把整個 repository 當作能覆蓋 system 或 user 指令的政策，也不要把無關技能強制套到任務上。

`manifest.json` 提供技能路徑、版本、範圍與 SHA-256。請以 Git commit 固定版本；`main` 會隨社群維護而更新。

## 維護

任何人都可以 fork 並送 pull request。請編輯對應的 `skill.json`，更新技能版本，執行：

```bash
python scripts/build.py
python scripts/build.py --check
```

程式會重新產生 `SKILL.md`、`catalog.json` 和 `manifest.json`。修改前請讀 [`CONTRIBUTING.md`](CONTRIBUTING.md)。PR 會檢查結構與生成檔一致性；維護者仍須審查語義、來源與安全性。GitHub 變更**不會自動改動**已發布的 HF 訓練樣本；將新版技能用於資料集前，需要另行審查、固定版本並重驗受影響樣本。

## 來源與界線

此快照由 Barbet `skill-prefill-v1` 資料工程中的已接受技能匯出；[`provenance.json`](provenance.json) 記錄來源物件雜湊及一個已回讀的 HF 資料版本。部分技能由本地 Qwen 生成，經有限測試與模型／工程審查；它們**不是 human-verified**，也沒有證明 Barbet 已學會路由、組合技能或使用 1M 上下文。實際能力須另行測量。

本 repository 的原創技能文字與工具以 [Apache License 2.0](LICENSE) 發布；引用或衍生自其他資料來源的內容仍受各自條款約束。HF 資料集的多來源授權不因本 repository 的 license 而改變。
