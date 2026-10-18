---
name: "conversation-reference-resolution"
version: "1"
scope: "general"
description: "將追問、代詞或補充資料綁定到可見前文，避免沿用已換題的任務。"
authority_basis: "authored_task_contract"
---

## Use When

當前訊息有明確前文指涉、繼續要求，或回答前一輪的澄清問題。

## Do Not Use When

沒有可見前文，或新訊息已提出完整且獨立的新任務。

## Inputs

目前使用者訊息及其之前的可見對話；不含待預測答案或未來訊息。

## Procedure

1. 找出需要解析的代詞、序號、補充值或「繼續」所指的內容。2. 只在可見前文中找符合語義與時間順序的先行詞；以前的 assistant 敘述不是外部事實的證明。3. 檢查是否換題、否定或撤回，新的明示條件優先。4. 延續時補足當前問題，不無故重做整段；若多個先行詞都可能成立，依原格式簡短澄清。

## Output Contract

保留目前要求的語言、形式與範圍，不強制重述整段對話。

## Failure

沒有可見先行詞或指涉矛盾時不要假裝記得未提供內容。

## Validation

解析依據在當前輸入內，未把舊題套到新題，未將舊敘述升格為已驗證結果。
