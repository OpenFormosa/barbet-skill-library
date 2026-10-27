# 固定權重的推論元件

這個套件讓應用程式明確管理來源、步驟、狀態和執行結果。它不訓練模型，不把模型自行宣稱的完成或通過當作驗證，也不會呼叫更大的模型補答案。

`harness.Episode` 只接受 `contracts.PublicRequest`：使用者題目、要求的輸出形式、可用工具 schema、來源 ID、公開測試及公開 setup。未知欄位會被拒絕。資料集答案和隱藏測試只交給獨立 scorer；生成程式沒有答案檔案的讀取路徑。模型也沒有檔案或網路工具可繞過這個分界。

可以分別採用這些元件：

| 元件 | 用途 | 不能證明什麼 |
| --- | --- | --- |
| `retrieval.LexicalIndex` | Unicode 詞彙、中文字元及雙字詞 BM25；回傳原文與來源雜湊 | 字面相關不等於語義足夠；不是語意 embedding |
| `retrieval.pack_documents` | 以目標 tokenizer 計算完整來源是否放得下，記錄省略項目 | 更長輸入不保證更好的閱讀 |
| `cards.json` | 有來源版本及雜湊的按需短操作卡 | 適用規則不是能力保證 |
| `state.State` | 保存來源片段、假設、失敗及 host 執行收據 | 引句存在不等於支持結論；公開測試通過不等於全題通過 |
| `contracts.validate_call` | 驗證工具名稱、欄位、型別、必填及 enum | 合法參數仍可能違反使用者語意 |
| `execution.run_python` | 在受限子程序中實際運行 Python 及測試 | 算術正確不代表題目建模正確 |
| `harness.Episode` | 最多三次模型呼叫，按真實失敗回饋修復，保留完整成本 | 重試和自行反省未必有增益 |

呼叫者提供模型 tokenizer，再重複呼叫 `generate_request()`、自己的生成後端、`accept()`，直到 `done`。`experiments/test_time_v1/run.py` 是可重現的 vLLM 後端範例。主套件除 Python 標準函式庫外不依賴生成框架；模型後端與統計程式另有依賴。

工具輸出可以使用模型原生表示或顯式 JSON。解析只處理語法，不翻譯識別碼、不改工具名稱、不猜布林值，也不依隱藏答案修復。程式型算術可以使用指定變數的 JSON 介面或只印出一個數字的 Python 介面；兩種都必須單獨量測，不能假設較複雜的介面較好。

Python 執行目前限 Linux 與 libseccomp：先載入允許的標準函式庫，設定 CPU、記憶體及輸出限制，降低 UID，之後拒絕檔案開啟、網路、process creation 等系統呼叫。缺乏隔離能力時直接失敗。另有語法／匯入限制，因此並非完整 Python 相容環境。這是研究用受限執行器，未宣稱通過通用敵意程式安全認證；正式產品仍應依部署環境加入容器或 VM 隔離。

輸入預算限制第一輪「題目＋schema＋短卡＋來源」。延續回合另有明確的上限，所有回合的輸入與輸出 token 都列入成本。輸出的 batch 時間是攤提吞吐成本，不是單一使用者延遲；固定 seed 也不保證跨 batch 與 GPU kernel 位元一致。

完整實驗規格、停止條件及目前結果見 [goal](../experiments/test_time_v1/goal.json) 與 [HTML 報告](../docs/test-time-study.html)。
