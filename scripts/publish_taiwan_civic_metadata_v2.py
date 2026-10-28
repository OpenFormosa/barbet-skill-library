"""Metadata-only supplement; preserve frozen civic subset driver and decisions."""
import fcntl
import html
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts import publish_taiwan_civic_v1 as release

ORIGINAL_METADATA = release.metadata


def supplemented(plan):
    if release.sha(Path(release.__file__)) != plan['driver_sha256']: raise ValueError('Frozen payload driver changed')
    check = release.read(release.STATE / 'independent-record-check.json')
    if not check['all_checks_passed'] or check['counts']['all_records_exact_render_and_full_arrays'] != 933:
        raise ValueError('New cohort verification incomplete')
    # The Barbet-only export holds metadata deltas, not inherited NOTICE bytes.
    # Resolve the unchanged notice through its manifest-verified ancestor; no
    # historical export or frozen row-subset driver is modified.
    previous_dest = release.prior.parent.DEST
    ancestor = release.prior.parent.parent.DEST
    expected_notice = next(f for f in plan['manifest'] if f['path'] == 'NOTICE.md')
    if release.sha(ancestor / 'NOTICE.md') != expected_notice['sha256']:
        raise ValueError('Inherited source notice changed')
    try:
        release.prior.parent.DEST = ancestor
        ORIGINAL_METADATA(plan)
    finally:
        release.prior.parent.DEST = previous_dest
    removed = {rid for job in plan['jobs'] for rid in job['removed_ids']}
    source = release.prior.audit.STATE / 'refinement-recorded-evidence-v1.parquet'
    table = pq.read_table(source)
    kept = table.filter(pc.invert(pc.is_in(table['record_id'], value_set=pa.array(sorted(removed)))))
    relative = 'proofs/taiwan-civic-v1/refinement-recorded-evidence-v1.parquet'
    target = release.DEST / relative; target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(kept, target, compression='zstd', compression_level=3)
    if not pq.read_table(target).equals(kept): raise ValueError('Retained refinement evidence changed')
    if len(table) - len(kept) != 5: raise ValueError('Refinement hold accounting differs')
    refinement = {'parent_source_sha256': release.sha(source), 'previous_tagged_episodes': len(table),
        'newly_held_tagged_episodes': len(table) - len(kept), 'current_tagged_episodes': len(kept),
        'current_sidecar': relative, 'current_sidecar_sha256': release.sha(target),
        'all_retained_tag_values_unchanged': True, 'recorded_signals_not_execution_or_successful_repair': True,
        'historical_parent_sidecar_preserved_and_no_longer_active_recipe_target': True}
    release.save(release.DEST / release.FOLDER / 'current-refinement-evidence.json', refinement)
    recipe_path = release.DEST / 'recipe/barbet-training-v2.json'; recipe = release.read(recipe_path)
    recipe['refinement_evidence_sidecar'] = relative
    curriculum = []
    cohort = release.read(release.STATE / 'new-cohort-summary.json')
    for item in cohort['files']:
        if '/records/' not in item['path']: continue
        for row in pq.read_table(release.DEST / item['path'], columns=['record_id', 'split', 'family_id', 'source_refs_json', 'pangolin_supervised_tokens']).to_pylist():
            refs = json.loads(row.pop('source_refs_json'))
            row['civic_stratum'] = refs['verification_tier']; curriculum.append(row)
    index_path = release.DEST / 'curriculum/taiwan-civic-v1.parquet'; index_path.parent.mkdir(exist_ok=True)
    index = pa.Table.from_pylist(curriculum); pq.write_table(index, index_path, compression='zstd')
    if len(index) != 933 or not pq.read_table(index_path).equals(index): raise ValueError('Civic curriculum index mismatch')
    cumulative = release.read(release.STATE / 'summary.json')['counts']
    civic_share = cohort['counts']['pangolin_supervised_tokens'] / cumulative['pangolin_supervised_tokens']
    recipe.update(civic_record_id_sidecar='curriculum/taiwan-civic-v1.parquet',
        civic_raw_supervised_token_share=civic_share,
        civic_sampling_weights='Uncalibrated: set only in a separately evaluated pilot; 12 authored normative cases are not broad alignment coverage',
        civic_id_sidecar_is_sampling_metadata_not_model_visible_evidence=True)
    release.save(recipe_path, recipe)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=release.prior.parent.base.REPO, text=True).strip()
    release.save(release.DEST / release.FOLDER / 'metadata-supplement-binding.json', {
        'metadata_driver_sha256': release.sha(Path(__file__)), 'metadata_library_commit': code,
        'frozen_payload_driver_sha256': plan['driver_sha256'],
        'new_cohort_check_sha256': release.sha(release.STATE / 'independent-record-check.json'),
        'current_refinement_sidecar_sha256': release.sha(target),
        'civic_sampling_index_sha256': release.sha(index_path),
        'no_row_target_score_split_or_mask_rewritten_by_metadata_supplement': True})
    release.save(release.DEST / release.FOLDER / 'independent-record-check.json', check)
    with (release.DEST / 'README.md').open('a') as handle:
        handle.write(f'\nThe current refinement recipe uses `{relative}`: '
            f'{len(kept):,} retained tagged episodes after removing five newly held IDs. '
            'The previous evidence file/report is preserved as historical provenance; '
            'its counts describe the parent revision. These tags are recorded upstream '
            'signals, not locally verified repairs or an RL reward.\n')
        handle.write(f'\nThe civic cohort is only {100*civic_share:.6f}% of cumulative supervised tokens '
            'under raw corpus sampling. `curriculum/taiwan-civic-v1.parquet` identifies all 933 civic '
            'records and their source-extractive/explicitly normative strata for transparent sampling. '
            'It is metadata, not RAG evidence. Sampling weights are uncalibrated; twelve normative '
            'cases do not establish broad value generalization and should not be multiplied to claim '
            'additional independent data.\n')
    s = release.read(release.STATE / 'summary.json'); c = s['counts']; u = s['unique_counts']
    content = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>台灣公民價值與 v26 續訓資料查核</title>
<style>body{{font:17px/1.8 system-ui;max-width:1080px;margin:40px auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>台灣公民價值與 v26 續訓資料查核</h1>
<p>明確採用台灣主權與人民自決，以及民主、自由、人權、法治、平等、多元。這是資料集所有者明示的規範立場，不冒充所有政府一致的外交認定；不預設支持政黨，允許批評台灣政府，尊重中國人民與所有群體的人權。</p>
<p>目前累計 {c['accepted']:,} 筆新 episode、{u['query_hash']:,} 個標準化首問、{u['family_id']:,} 個家族；train {c['split:train']:,}、validation {c['split:validation']:,}。現役舊資料為 0，仍高於一千萬筆。Barbet physical tokens {c['pangolin_tokens']:,}、監督 tokens {c['pangolin_supervised_tokens']:,}；不是每筆都具有長篇答案或同等難度。</p>
<h2>先前品質修訂</h2><p>對原始 10,328,696 筆的完整 arrays 重新檢查，發現 62 筆正文含原生 PAD。另有日期星期矛盾、缺乏訂位事實的對話補空範圍、舊式工具文字與宣告 JSON 介面不一致，以及首次內容檢視不通過題目。前次修訂整題移除 7,552 筆，沒有以次低 PPL 答案替補。</p>
<p>原生 v26 在兩筆保留資料的 no-grad padded forward 通過，模型 masked loss 與獨立 single-shift CE 完全相同（0.6769216656684875，296 個監督 tokens）。首次強制 Flash-only 遇到 padding mask 無可用 kernel 的失敗已保留；成功測試使用預設可處理 mask 的 SDPA。這不是 gradient、長序列或 packing 測試，更不是訓練成效。</p>
<h2>政治內容處理</h2><p>全原始語料進行詞彙主題分流，找到 146 筆自然語言候選 target。舊資料確有把中國主權主張延伸成實際管轄的問題；其他命中包含正確引述及無關內容。採保守待審隔離 140 個 query，共 {s['new_political_topic_held_rows']:,} 筆，並非宣稱所有隔離答案皆錯誤。不能以此詞彙掃描保證全語料已完成語義或政治價值驗證。</p>
<p>3,402 個結構化來源紀錄的主題命中不直接視為世界事實或主權立場背書；有限紀錄查詢仍依其明示資料契約判讀。政治價值卡不能凌駕事實，也不能竄改法律原文或不同意見的翻譯。</p>
<h2>新增 933 筆</h2><table><tr><th>類別</th><th>筆數</th><th>驗證範圍</th></tr>
<tr><td>22 部台灣法規的條文檢索摘錄</td><td>920</td><td>逐筆對照 XML 原文、條號、法規名稱及可見文件；只回答提供版本，不宣稱目前個案適用</td></tr>
<tr><td>所有者明示價值案例</td><td>12</td><td>主權與自決、人權、政黨不預設、政府監督、忠實引述、事實與立場區分；明示為原創規範案例</td></tr>
<tr><td>外交部歷史聲明</td><td>1</td><td>2022/04/26 原句與發言機關、日期歸屬；不當成今日所有國家的認定</td></tr></table>
<p>法規包含憲法權利及增修條文、選舉、公投、政府資訊公開、兩公約施行法、原住民族、身心障礙及性別平等相關制度。原憲法只取基礎與人民權利範圍，未將停止適用的國民大會制度當成現況；未來適用判斷仍需增修、修法及生效條件。22 部法規來源及授權：<a href="https://data.gov.tw/dataset/18289">法務部法律開放資料</a>，OGDL-Taiwan-1.0；原創價值案例、政策及技能採 Apache-2.0。<a href="https://www.mofa.gov.tw/cp.aspx?n=292">外交部再利用宣告</a>另有明確歸屬。</p>
<p>前置 memory 463 筆、後置 470 筆；同一法規家族不跨 train/validation，干擾文件只從同一 split 取用。新資料均為單一合格答案，明確未量測 PPL；沒有產生答案改寫版本灌量。全部 933 筆重建完整訓練文字、IDs、labels 及答案前綴一致；使用與 v26 相同的 Barbet tokenizer，不載入 MiniCPM tokenizer。新資料最長含 4096 output reserve 為 8649 tokens；不能把這一小批宣稱為長至 128K 的政治閱讀測試。</p>
<h2>Refinement 與後續續訓</h2><p>目前 recipe 指向移除 5 個新隔離 ID 後的 {len(kept):,} 筆 recorded same-tool-retry 證據。保留 tag 值未變，歷史檔案保留。這只是上游記錄順序與訊號，不能當作實際成功修復、RL reward 或完整涵蓋所有 refinement 的證明。</p>
<p>維持 BarbetSFTCollator、assistant-only labels、右側 padding、模型內一次 shift、完整輸入不截斷及 output reserve。跨樣本 packing 在 attention 與 Mamba reset 未驗證前關閉；推論應使用 barbet_completion 或 ContextAgent 的 barbet_portable_v1。原生 v26 PPL 選答準則保留，不等於 portable 模板下也已重新量測最低 PPL。</p>
<p>這 933 筆只佔極少監督 tokens；不能由筆數推論模型已建立穩定政治價值。正式訓練需另行設定抽樣比例，並用未參與挑選的題目評估事實、來源歸屬、立場區分、人權與異議，以及中文、數學、coding/tool 保留能力。本次沒有更新模型權重，沒有宣稱能力或價值對齊已提升。</p>
<h2>發布驗證</h2><p>既有檔案沿用已驗證的 Git 物件；受影響分片只刪除整題，保留資料值與完整 arrays 逐項相等，先在固定 staging commit 完整匿名 SHA-256 回讀，再以相同物件發布 main。新增資料及 metadata 另完整匿名讀回。這是組合驗證，不是重新下載全部歷史檔案；最終 commit 與完成狀態以根目錄 PUBLICATION.json 為準。</p>
<pre>{html.escape(json.dumps({'counts':c,'unique':u,'refinement':refinement},ensure_ascii=False,indent=2))}</pre></html>'''
    (release.DEST / release.FOLDER / 'report.html').write_text(content)


if __name__ == '__main__':
    release.metadata = supplemented
    with (release.prior.parent.base.WORK / 'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); release.publish()
