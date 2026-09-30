---
name: "sequence-inspection-and-reduction"
version: "2"
scope: "coding"
description: "提供通用序列掃描與歸約程序，處理空值判斷、極值搜尋及布爾輸出，嚴格遵守不可變性與操作限制。"
authority_basis: "authored_task_contract"
---

## Use When

需檢查序列是否為空、計算元素極值、執行線性掃描歸約，且任務不涉及排序、動態規劃或外部狀態時。

## Do Not Use When

任務需要配對、區間DP、修改原始輸入、使用被禁用的內建函式，或處理非序列結構時。

## Inputs

待處理序列、歸約目標（空檢查/極值/布爾）、操作限制清單（禁用builtin/method）、預期輸出類型。

## Procedure

先依契約處理空輸入，再掃描或歸約；空檢查區分無元素與0/False元素，極值初始化不得忽略全負數。保留Boolean型別、非修改性、原始順序與禁用builtin/method限制，不強制使用len或iterator等被禁操作。

## Output Contract

只產生目前題目要求的函式、Boolean或數值等輸出；不自行加入None、錯誤代碼或額外訊息。

## Failure

程序不適用時不啟動；發現候選違反公開契約時修正並重驗，不為合法輸入添加題目未要求的錯誤輸出。

## Validation

檢查輸出類型是否嚴格符合契約（如Boolean非int），確認輸入序列未被修改，並驗證極值計算涵蓋邊界情況（如單元素、全負數）。
