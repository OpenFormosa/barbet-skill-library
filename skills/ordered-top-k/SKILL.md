---
name: "ordered-top-k"
version: "1"
scope: "math"
description: "依明示排序鍵、方向及重複值規則選出前 k 項。"
authority_basis: "authored_task_contract"
---

## Use When

給定集合或列表，要求排序、名次、最大或最小的 k 個值。

## Do Not Use When

要查外部排行榜、只有線性總和，或需要未給出的排序資料。

## Inputs

完整項目、k、排序方向、排序鍵與同分規則。

## Procedure

1. 確認選項是值還是記錄、k 的範圍及升降序。2. 除非要求唯一值，重複值仍視為不同項目。3. 多欄位排序依明示優先序；同分未指定時保留原相對順序或說明採用方式。4. 排序後取指定區段，不把取 k 個變成取 k 種不同值。

## Output Contract

保留原值與要求的排序形式；不補入不存在的項目。

## Failure

k 超過可用項目、鍵缺失或型別不可比較時依契約澄清或說明邊界。

## Validation

結果元素均來自原集合，數量、方向、同分及重複規則一致。
