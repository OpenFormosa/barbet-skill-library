"""Render measured evidence without promoting proxy scores to capability claims."""
import html


def gpu_report_html(summary, probes, controls):
    t = summary['totals']
    table = ''.join('<tr><td>' + html.escape(r['family']) + '</td><td>' + str(r['roots']) +
        '</td><td>' + (f"{r['sum_logprob_pairs_correct']} / {r['discriminating_first_action_pairs']}" if r['discriminating_first_action_pairs'] else '首動作相同，未計分') +
        '</td><td>' + (f"{r['mean_logprob_pairs_correct']} / {r['discriminating_first_action_pairs']}" if r['discriminating_first_action_pairs'] else '未計分') +
        f"</td><td>{r['full_nll']:.4f}</td><td>{r['skill_gain_nats']:+.4f}</td><td>{r['evidence_gain_nats']:+.4f}</td></tr>"
        for r in probes['families'])
    source_rows = ''.join('<tr><td>' + html.escape(name) + f"</td><td>{r['rows']:,}</td><td>{r['token_weighted_nll']:.5f}</td><td>{r['example_mean_nll']:.5f}</td></tr>"
        for name, r in summary['source_groups'].items())
    return f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Barbet v4 全量 GPU 資料評量</title>
<style>body{{max-width:1150px;margin:36px auto;padding:0 22px;font:17px/1.7 system-ui;color:#25372f}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #bbb;padding:7px}}code{{overflow-wrap:anywhere}}a{{color:#176e58}}</style>
<h1>Barbet v4：全量 GPU 評量與可學習性界線</h1>
<p>全部 {t['rows']:,} 筆新版資料、{summary['files']:,} 個 tokenized 分片已用凍結 v26 完成計算，逐筆對回公開來源，沒有遺漏、重複或截斷。共有 {t['input_tokens']:,} input tokens、{t['supervised_tokens']:,} supervised tokens。使用同一台機器最多 8 張 H200；沒有模型更新。</p>
<p>模型固定為 <code>voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66</code>，權重 SHA-256：<code>{summary['weight_sha256']}</code>。這不是目前最佳 recovery checkpoint 的替代品，也不是 Barbet base 的能力成績。</p>
<h2>全資料實測</h2><ul><li>按 supervised tokens 加權的 NLL：{t['token_weighted_nll']:.6f}；逐筆平均 NLL：{t['example_mean_nll']:.6f}。</li><li>Teacher-forced 下一 token 命中率：{t['teacher_forced_top1_fraction']:.2%}。它不是自由生成任務成功率。</li><li>每筆另外保存 readout gradient sketch norm 與對七種 TRAIN 語義原型的 cosine；這些是資料相關性代理，不是正確性判定。</li><li>模型偏好沒有改變任何來源品質級別、目標答案或先前隔離決定。</li></ul>
<h2>低 NLL 掩蓋的語義錯誤</h2>
<p>對所有 2,000 個既有 TRAIN 根任務，建立原題、語義反事實、不變改寫、移除 skill、移除證據與 answer-only 等 15,430 個候選評分輸入。1,715 組首動作確實不同，285 組首動作相同而不適合此成對指標。原題與反事實都必須偏好各自正確答案才算成對通過。另同時報告總 log probability 與長度正規化結果，避免隱藏長度偏差。</p>
<table><tr><th>能力</th><th>根任務</th><th>總 log probability 成對通過</th><th>長度正規化成對通過</th><th>原題 NLL</th><th>Skill 增益</th><th>證據增益</th></tr>{table}</table>
<p>增益定義為「移除後 NLL − 完整輸入 NLL」，正值代表加入內容提高參考答案機率。移除證據後仍評同一個 gold，只測機率敏感度；缺證據時的正確行動可能改為補查或澄清，因此不能把此數字當作缺證據作答準確率。這些是 TRAIN 上的診斷，不是封存 TEST 或泛化成績。</p>
<p>獨立覆核已重建所有 9,430 個原題／反事實／不變候選的實際 prefix 與 target，並核對全部 15,430 個評分身分。比較 287 組、排序 286 組、版本規則 284 組的兩個候選只差 operation 欄位。結果支持優先保留語義對照與操作綁定；不能把其中 NLL 較高、但條件下正確的一側刪除。</p>
<h2>對資料課程的實際處理</h2><ul><li>每筆 GPU 測量以 record_id 與原資料對齊，保留來源、品質範圍、split、root 與 parent 身分。</li><li>同一完整輸入的既有 frozen-v26 勝出答案保持不變；不同條件的反事實不是同題答案競爭，不適用擇低 PPL 刪除。</li><li>不按全資料 NLL 或 cosine 全域排序決定去留；這會偏向熟悉格式，也可能削弱數學、繁中、自然指令與價值對齊內容。</li><li>reference_only 不因低 loss 轉成已驗證資料。台灣主權、人民自決、普世人權與原始事實／引述忠實性，不因學生模型意見不同而移除。</li><li>50% 一般能力、50% 語義課程仍只是按 supervised-token 曝光的候選配比。有限圖資料占比很大，不能直接均勻取樣全部一千萬筆。</li></ul>
<h2>數值驗證與文獻依據</h2>
<p>實際 Barbet 控制中，短序列 dense NLL 與只投影監督位置的 NLL 都是 {controls['dense_nll']:.9f}；單筆與 batch 的 NLL 比對、sketch cosine 與小矩陣 analytic/autograd 對照均通過。權重與 native loader 固定，保留必要 FP32 參數，沒有改 tokenizer 或使用 MiniCPM tokenizer。</p>
<ul><li><a href="https://arxiv.org/abs/2610.03702">LESSER（2026 預印本）</a>：採用 forward-only、雙側 Rademacher 的 readout gradient sketch。此處只測 detached readout copy，並非 tied embeddings 的完整梯度，也沒有聲稱其 Barbet 選料效果已獲訓練驗證。</li><li><a href="https://arxiv.org/abs/2402.04333">LESS</a>：借鑑目標能力相關性；沒有重現其 warmup 與 optimizer-aware influence。</li><li><a href="https://arxiv.org/abs/2308.12032">Instruction-Following Difficulty</a>：另報 answer-conditioned 與 answer-only 的 NLL 比值。此處是凍結 v26 的 IFD-like 診斷，不是原論文完整流程，沒有套用其門檻篩掉資料。</li><li><a href="https://arxiv.org/abs/2206.07137">RHO-LOSS</a>：區分已會、值得學與可學；支持不用 loss 單獨判斷資料真實性。</li><li><a href="https://arxiv.org/abs/2411.06037">Sufficient Context</a>：資訊充分和模型能利用資訊是不同問題。</li><li><a href="https://huggingface.co/datasets/openbmb/UltraData-SFT-2605">UltraData 官方資料卡</a>：資料品質與小型訓練後的用途驗證分開。本次沒有重現其訓練驗證。</li></ul>
<h2>逐來源完整統計</h2><table><tr><th>品質範圍 | 來源 | split</th><th>筆數</th><th>Token 加權 NLL</th><th>逐筆平均 NLL</th></tr>{source_rows}</table>
<p>本次證明的是全量重處理、來源與格式核對，以及凍結學生模型的測量覆蓋。尚未證明訓練後增益、未見組合遷移、通用能力保留、1M 自由生成或追平 Qwen／MiniCPM。</p>
<p><a href="metadata/gpu-complete-summary.json">全量 GPU 總帳</a> · <a href="metadata/gpu-curriculum-policy.json">資料使用規則</a> · <a href="metadata/gpu-context-probe-summary.json">成對語義結果</a> · <a href="report.html">完整資料重處理報告</a></p></html>'''
