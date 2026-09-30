# Contributing

歡迎透過 fork 和 pull request 新增或改善技能。請保持技能能跨多個任務重用；不要把單題答案、完整 patch、隱藏測試、私人資料或憑空捏造的 API／公司政策寫進技能。

1. 編輯 `skills/<name>/skill.json`，或以小寫英數與連字號建立新目錄。既有技能有語義變更時，更新 `version`；保留變更理由與受影響情境於 PR。
2. 寫清楚 `use_when`、`do_not_use_when`、`inputs`、`procedure`、`output_contract`、`failure`、`validation`。程序不得越過使用者授權或宣稱未執行的工具結果。
3. 若內容依賴外部文件、API 或資料集，在 PR 描述附上來源、版本及可用權利。避免貼入長篇來源原文。
4. 執行 `python scripts/build.py` 與 `python scripts/build.py --check`，提交生成的 Markdown、catalog 與 manifest。
5. 提供至少一個適用例、一個相似但不適用的例子，以及可執行或可審核的驗證方法。描述修改後對既有技能使用者的影響。

`skill.json` 是編輯來源；`SKILL.md` 是給模型與讀者使用的生成檔。PR 不應只修改生成檔。檢查通過不代表內容正確；維護者需審查技能邊界與來源。此 repo 的變更不會自動更新 HF 訓練資料。

Contributions to the authored skill library are offered under the repository's Apache-2.0 license. Do not submit material that you cannot authorize for that use.
