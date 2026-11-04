---
name: "external-state-grounding"
version: "2"
scope: "agent"
description: "區分需要外部狀態的查詢或操作與可直接提供的說明，避免虛構即時資訊及執行結果。"
authority_basis: "authored_task_contract"
---

## Use When

請求涉及即時數值、外部帳號或資產、遠端查找、實際改變外部狀態，或查核明確標示的版本化模擬器狀態。

## Do Not Use When

不需查核狀態的一般概念解釋、已提供完整資料的離線計算；模擬器回執不能冒充真實外部執行。

## Inputs

當前請求、可見來源和日期、現有工具與權限、輸出契約。

## Procedure

1. 確認任務、operation ID、目前可見版本與時間。2. 區分 submitted、running、partial、failed、cancelled、completed；只有已完成且符合任務成功條件才可宣稱完成。3. 讀取工具回執，將查詢成功與工作完成分開。4. 相同 operation ID 不重複提交；重試前確認故障類型與當前狀態。5. 取消後晚到的結果不自動恢復或完成工作。6. 來源不可取得就說明已知狀態與未知部分，不編造結果。

## Output Contract

遵從原格式與語言；不因本技能額外要求工具、瀏覽器或新權限。不得把建議步驟寫成已完成的操作。

## Failure

來源過期、不可存取或無執行能力時保留未知，避免編造價格、聯絡資料、識別碼、連結或成功狀態。

## Validation

逐一檢查時效詞、外部數值、個人資料及完成聲明的可見依據；不輸出未經驗證的推測作事實。
