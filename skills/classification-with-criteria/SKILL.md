---
name: "classification-with-criteria"
version: "1"
scope: "general"
description: "先定義分類標準，再將可見特徵對到標籤，保留歧義。"
authority_basis: "authored_task_contract"
---

## Use When

要求對項目分類、標記、辨識情緒、類型或是否符合條件。

## Do Not Use When

要求自由創作新類別、未知外部實體的權威分類，或缺項目且無法確定範圍。

## Inputs

待分類項目、允許標籤、定義與可見特徵。

## Procedure

1. 優先使用使用者提供的標籤及標準；沒有指定時選與任務一致的常見分類方式並避免假稱權威。2. 根據項目實際特徵判斷，不用名稱聯想補出未見內容。3. 分清互斥分類、多標籤及程度值。4. 邊界情況只在原格式允許時交代不確定，不以多餘標籤逃避選擇。

## Output Contract

嚴守標籤集合與輸出格式，禁止增加未要求的評分欄位。

## Failure

分類準則矛盾或證據不足時依原契約澄清或標未知。

## Validation

標籤合法，判準一致，證據支援分類，沒有把偏好當成事實。
