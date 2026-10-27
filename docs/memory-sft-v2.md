# 外部記憶、技能與具型別工具

0.2.0 新增 `ContextAgent`、`MemoryStore`、技能檢索及新資料格式的推論轉接器。
這些介面與既有 `Harness` 共存；既有固定 benchmark 的規則與結果維持原版。
新介面有工程測試，目前沒有新的模型準確率實驗或訓練結果。

## 記憶與技能

```python
from barbet_inference import ContextAgent, MemoryItem, SkillLibrary

library = SkillLibrary()
question = "依照手冊，設備 X 的允許溫度是多少？"
skills = [s["name"] for s in library.search(question, top_k=2)]
agent = ContextAgent(
    backend,  # 與既有 Harness 相同的 generate/tokenizer/max_model_len 介面
    library=library,
    skills=skills,
    memory=[MemoryItem("manual-x", "設備 X 允許操作溫度為 5–35°C。", "手冊第 3 頁")],
    memory_placement="back",  # front/back 都在答案之前，不是答案後偷看未來
)
result = agent.run(question)
```

技能搜尋只讀使用者問題與技能目錄，分數是詞彙相關性，不是正確率。
未指定技能時有保守預設；尚未宣稱這個預設對每個模型最佳。
記憶保留來源、雜湊、類型與最早可見回合。工具回傳發生後才寫入記憶，
同一個 agent 後續提問可以檢索；模型自己的猜測不能直接當成已驗證事實。

## 工具與驗證

`ContextAgent(..., tools=schemas, handlers={"lookup": lookup})` 接受明確註冊的
Python callback。完整 JSON Schema 驗證通過後才執行；同批先驗證全部參數。
沒有 handler 時回傳 `awaiting_tool_execution`，不製造觀察。
工具描述與檢索文件不會註冊新工具，也不會直接執行下載的程式碼。
應用端負責 callback 的權限、隔離與副作用界線。

`verifier(answer)` 應回傳 `Verification(passed, feedback, candidate_sha256, scope)`。
雜湊必須對應當次完整答案；只有真實回饋會觸發修正，最多兩次。
相同答案反覆出現會停止，通過狀態只代表指定 scope 通過。
未提供 verifier 時狀態是 `answered_without_external_verification`。

每一步用實際 backend tokenizer 計算完整歷史加輸出預算；超長不截斷。
原生聊天分隔符出現在輸入或工具字面值時目前會明確拒絕，避免把資料誤當聊天邊界；
呼叫端必須提供安全的引用表示，不能把被改動的參數當成原始值執行。
callback 例外只公開錯誤類別，執行是否已完成標為 unknown。

## 訓練資料轉成推論請求

```python
from barbet_inference import prepare_dataset_request

request = prepare_dataset_request(row, model="served-model", mode="chat")
# client.chat.completions.create(**request)

request = prepare_dataset_request(
    row, model="served-model", mode="pangolin_completion"
)
# 直接送 /v1/completions；不可再套 chat template 或重加 BOS。
```

`target_turn` 可指定任一有監督的 assistant 回合；預設最後一回合。
目標答案與未來回合都移除，較早真實 assistant 行動與工具回傳保留。
Chat 使用完整工具 schema 與 JSON 呼叫／觀察封裝，不另外傳第二套 `tools`。
Completion 可選 `pangolin_completion` 或 `minicpm_completion`，
直接取已驗證 renderer 的字元前綴，不把儲存的訓練 token IDs 移植到其他 tokenizer。

`tokenized` 與 `minicpm5_tokenized` 的 labels 未做 shift，非 assistant 位置為 -100。
訓練框架需只 shift 一次。兩套 tokenizer 視圖、不同前後記憶位置都不算獨立新題。
資料建置預留 4,096 個輸出 token；推論換模型或改 max_tokens 後仍需重新量測。

## 資料規模與限制

`configs/memory_sft_v2_scale_10m.json` 將目標定為一千萬筆通過檢查的新軌跡，
並另報獨立首題、保守家族、有效監督回合與 token。第一批不代表 goal 完成。
原始來源數、重複教師回答與 tokenizer 副本不直接當成獨立難題。

發布流程是 `build_memory_sft_v2.py` → `package_memory_sft_v2.py` →
`release_memory_sft_v2.py prepare/publish/verify`。每步保留來源修訂與雜湊，
確認 family/query 不跨 split，逐筆測量兩套訓練 tokenizer 與 Qwen Chat/Completion 預算，
讀回所有 Parquet，發布後再匿名讀取全部新檔案核對 SHA256。

舊版資料與首次失敗仍保留為歷史證據，不加入現役資料灌量。
靜態工具軌跡保留上游結果，不宣稱本機重新執行；工程檢查不等於每筆答案都正確。
`recorded_error_followup` 目前只是工具觀察中的錯誤詞偵測，不能作為成功修復統計。
實際模型增益需要後續獨立的訓練與 held-out 測試；本階段不訓練權重。
