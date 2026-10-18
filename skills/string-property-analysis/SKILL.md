---
name: "string-property-analysis"
version: "1"
scope: "coding"
description: "釐清字串正規化與邊界後檢查回文、重排、相似性或字面關係。"
authority_basis: "authored_task_contract"
---

## Use When

給定字串並要求回文、anagram、字元計數、字面相似度或相關字串性質。

## Do Not Use When

外部實體是否存在、身份驗證，或未指定準則的語義或事實判定。

## Inputs

原始字串、大小寫、空白與標點規則、索引或相似度定義。

## Procedure

1. 保留原始字串，另建依明示規則正規化的視圖，不默默移除空白或忽略大小寫。2. 回文比較對稱位置；重排比較字元頻數；子字串與子序列不可混淆。3. 相似度須有明確度量，缺少時說明採用方式或澄清，不能捏造客觀分數。4. 如需位置，將結果映回原字串索引規則。

## Output Contract

只依原要求回傳判定、片段、位置或必要說明；不因技能強制輸出程式。

## Failure

正規化、索引基準或度量改變答案時明示歧義；不將字面相似當成身份相同。

## Validation

字符頻數、對稱與邊界可直接核對，空字串及重複字元的處理一致。
