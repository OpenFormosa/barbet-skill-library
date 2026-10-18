---
name: "descriptive-statistics"
version: "1"
scope: "math"
description: "釐清樣本範圍、分母與統計量定義後計算描述統計，避免自動刪除離群值。"
authority_basis: "authored_task_contract"
---

## Use When

給定數據並要求平均數、中位數、眾數、變異數、標準差或頻率。

## Do Not Use When

未給數據的概念說明、推斷因果、或僅做排名。

## Inputs

原始數值、缺失值規則、母體或樣本定義、精度。

## Procedure

1. 確認有效觀測值及數量，保留重複值；除非明示不得刪除離群值。2. 對需要排序的統計量排序，分清空集、單一值與奇偶樣本。3. 變異數及標準差先確認母體分母 n 或樣本分母 n-1，未明示且影響答案時標示採用定義或給必要分支。4. 最後統一四捨五入。

## Output Contract

依原要求呈現數值及必要定義；不強制額外圖表或推論。

## Failure

空集、非法值或不足以計算樣本變異數時說明未定義；未知值不能當成零。

## Validation

觀測數、分母、平方根及精度可核對，數據沒有被擅自改動。
