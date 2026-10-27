"""Offline HTML reporting; no external assets and no inferred missing results."""
from pathlib import Path
import datetime as dt
import html
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOMAINS = [('drcd','閱讀 EM'),('gsm8k','數學'),('mbpp','程式'),('tools','工具'),('tmmluplus','知識保留')]
E = lambda x: html.escape(str(x))
P = lambda x: '—' if x is None else f'{100*x:.2f}'


def detail(v):
    if isinstance(v, dict):
        return '<table class="detail">'+''.join('<tr><th>'+E(k)+'</th><td>'+detail(x)+'</td></tr>' for k,x in v.items())+'</table>'
    if isinstance(v, list):
        return '<ol>'+''.join('<li>'+detail(x)+'</li>' for x in v)+'</ol>'
    return E(v)


def table(rows, cols):
    return '<div class="scroll"><table class="sortable"><thead><tr>'+''.join('<th>'+E(k)+'</th>' for k,f in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+E(f(r))+'</td>' for k,f in cols)+'</tr>' for r in rows)+'</tbody></table></div>'


def runs(rows):
    if not rows:
        return '<p>尚無完整結果；不把部分題數當完整評分。</p>'
    cols=[('模型／設定',lambda r:r['model']+'/'+r['policy']),('題 × seed',lambda r:f'{r.get("unique_questions",r["items"])} × {len(r.get("seeds",[0]))}'),('主指標 %',lambda r:P(r.get('primary_macro')))]
    cols += [(label+' %',lambda r,k=k:P(r.get('domains',{}).get(k,{}).get('accuracy'))) for k,label in DOMAINS]
    cols += [('閱讀 F1',lambda r:P(r.get('secondary',{}).get('qa_character_f1'))),('呼叫／題',lambda r:f'{r["cost"]["calls"]:.2f}'),('輸入／題',lambda r:f'{r["cost"]["prompt_tokens"]:.0f}'),('輸出／題',lambda r:f'{r["cost"]["output_tokens"]:.0f}'),('截斷呼叫',lambda r:r['length_stopped_calls'])]
    return table(rows,cols)


def cost_plot(rows):
    rows=[r for r in rows if r.get('primary_macro') is not None]
    if not rows:return ''
    maximum=max(r['cost']['output_tokens'] for r in rows)*1.15 or 1
    svg='<svg viewBox="0 0 1000 310" role="img" aria-label="正確率與生成成本">'
    for value in [0,25,50,75,100]:
        y=265-value*2.35
        svg+=f'<path d="M60,{y} H970" stroke="#dce3e9"/><text x="50" y="{y+4}" text-anchor="end">{value}%</text>'
    for r in rows:
        x=65+r['cost']['output_tokens']/maximum*850;y=265-r['primary_macro']*235
        color='#296cbb' if r['model']=='qwen' else '#087d70'
        title=f'{r["model"]}/{r["policy"]}: {P(r["primary_macro"])}%, {r["cost"]["output_tokens"]:.0f} output tokens'
        svg+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{E(title)}</title></circle><text x="{x+7:.1f}" y="{y-5:.1f}" font-size="10">{E(r["policy"])}</text>'
    return '<figure>'+svg+'<text x="500" y="300" text-anchor="middle">每題總輸出 token，包含重試（線性軸）</text></svg><figcaption>同一階段與相同任務組成；越右代表生成更多文字，沒有自動更準確。不同 tokenizer 的 token 數不是等量 FLOPs。所有點的原始數字在表格中。</figcaption></figure>'


def context_panel(context):
    if not context:return '<p>上下文控制尚未完成。</p>'
    rows=context['table'];svg='<svg viewBox="0 0 1000 325" role="img" aria-label="上下文長度與閱讀 F1">'
    for v in [0,25,50,75,100]:
        y=280-v*2.4;svg+=f'<path d="M60,{y} H950" stroke="#dce3e9"/><text x="50" y="{y+4}" text-anchor="end">{v}</text>'
    for i,b in enumerate([2048,4096,8192,16384]):
        svg+=f'<text x="{100+i*270}" y="310" text-anchor="middle">{b//1024}K</text>'
    for family,color in [('mini','#087d70'),('qwen','#296cbb')]:
        for pos,dash in [('beginning',''),('middle','8 5'),('end','2 5')]:
            pts=[]
            for i,b in enumerate([2048,4096,8192,16384]):
                r=next((r for r in rows if r['model']==family and r['condition']==f'padding_{b}_{pos}'),None)
                if r:pts.append(f'{100+i*270},{280-r["character_f1"]*240:.1f}')
            svg+=f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>'
    out='<figure>'+svg+'</svg><figcaption>字元 F1；綠色 MiniCPM、藍色 Qwen。實線／虛線／點線：證據在開頭／中間／結尾。每個長度內使用相同來源與干擾集合，曲線不平滑。這是 32 題開發診斷，不是封存測試。</figcaption></figure>'
    out+=table(rows,[('模型',lambda r:r['model']),('條件',lambda r:r['condition']),('題數',lambda r:r['questions']),('EM %',lambda r:P(r['accuracy'])),('F1',lambda r:P(r['character_f1'])),('實際輸入 token',lambda r:f'{r["mean_prompt_tokens"]:.0f}'),('輸出 token',lambda r:f'{r["mean_output_tokens"]:.0f}'),('未結束思考',lambda r:r['unclosed_calls'])])
    return out+'<p>oracle_only 知道正確來源，不能當可部署檢索結果。不同長度的干擾集合可能不同，各條件也使用不同亂數串流；題目配對不等於逐 token 相同的隨機生成。</p>'


def gallery(rows):
    out='<p>模型生成前固定選定：開發集每領域依既定順序的前兩題。保留失敗，不依結果挑案例。完整來源與隱藏程式測試不在此重製。</p>'
    for r in rows:
        out+='<details><summary>'+E(r['domain']+' / '+r['id'])+'</summary><p>'+E(r['public_task_excerpt'])+'</p><p>參考：'+E(json.dumps(r['reference'],ensure_ascii=False))+'</p>'
        for t in r['trajectories']:
            out+='<h4>'+E(t['policy'])+'</h4><pre>'+E(json.dumps(t['prediction'],ensure_ascii=False,indent=2))+'</pre><details><summary>完整呼叫、執行收據與成本</summary><pre>'+E(json.dumps(t,ensure_ascii=False,indent=2))+'</pre></details>'
        out+='</details>'
    return out


def main():
    result=json.loads((HERE/'results.json').read_text());goal=json.loads((HERE/'goal.json').read_text())
    comparisons=result.get('comparisons') or {};selected=(result.get('stopping') or {}).get('selected_policy','待確認')
    delta=comparisons.get('selected_minus_bare',{}).get('primary_macro');gap=comparisons.get('qwen_same_minus_selected',{}).get('primary_macro')
    change='封存測試待完成' if not delta else f'{delta["delta_pp"]:+.2f} pp；95% CI [{delta["paired_95ci_pp"][0]:+.2f}, {delta["paired_95ci_pp"][1]:+.2f}]'
    gap_text='同工具比較待完成' if not gap else f'9B − 1B：{gap["delta_pp"]:+.2f} pp；CI 上界 {gap["paired_95ci_pp"][1]:.2f} pp'
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    css='''body{font:15px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#172534;margin:0}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.35}h2{margin-top:36px}a{color:#1767a4}p{max-width:1050px}.notice{background:#fff3d1;padding:15px;border-left:4px solid #b18115}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.tile,section,details{background:#fff;padding:18px;border-radius:8px;margin:12px 0}.tile b{display:block;font-size:20px}.tile small{color:#52616b}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{border:1px solid #dce3e9;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f6;cursor:pointer;white-space:nowrap}.scroll{overflow:auto}.detail{table-layout:fixed;overflow-wrap:anywhere}.detail th{width:25%;white-space:normal}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px;max-height:600px;overflow:auto;font-size:12px}svg{width:100%;height:auto}svg text{font-family:system-ui,sans-serif;font-size:12px}figcaption,footer,.muted{color:#52616b;font-size:13px}figure{margin:12px 0}li{margin:7px 0}nav{display:flex;gap:18px;flex-wrap:wrap}#stale{display:none;background:#fff3d1;padding:12px}footer{padding:20px 0}details details{border:1px solid #dce3e9}@media print{body{background:#fff}main{padding:0}nav,#stale{display:none}.scroll{overflow:visible}table{font-size:10px}}'''
    out='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>固定權重的小模型 test-time 實驗</title><style>'+css+'</style><main><h1>固定權重的小模型 test-time 實驗</h1><p>MiniCPM5-1B、skill、RAG、可回查狀態與真實執行回饋；以 Qwen3.5-9B 比較明確任務與成本範圍。</p><nav><a href="#outcome">結果</a><a href="#search">搜尋與驗證</a><a href="#context">上下文</a><a href="#failures">失敗分析</a><a href="#goal">完整 Goal</a><a href="#probes">固定案例</a></nav><p class="notice">完全不訓練、不改權重，沒有大模型替小模型生成或評審。主指標是四類任務的等權平均，知識保留另列。這不是所有任務的參數等效換算。</p><div id="stale"></div>'
    out+='<div class="tiles">'+''.join('<div class="tile">'+E(title)+'<b>'+E(value)+'</b><small>'+E(note)+'</small></div>' for title,value,note in [('搜尋上限','18 個候選','固定消融與上下文診斷不參與新策略搜尋。'),('選定設定',selected,'依驗證門檻選擇；封存測試後不換策略。'),('對小模型基準的改善',change,'達標：最終配對 CI 下界大於 0。'),('同工具 9B 差距',gap_text,'接近：差距 CI 上界 ≤3 pp，單項差距 ≤5 pp；知識另需通過。')])+'</div><p class="muted">沒有訓練 loss、step 或收斂 ETA。本研究只有固定權重推論；有限搜尋停止不能推論不存在更好的方法。</p>'
    out+='<h2 id="outcome">封存測試與成本</h2><section>'+runs(result.get('test',[]))+cost_plot(result.get('test',[]))+'</section><p>同一題三個 seed 先取平均，再做分層配對 bootstrap；另列文件／工具家族／科目的群聚敏感度。不能把重複 seed 當三倍獨立題數。攤提 batch 時間是吞吐成本，不是單一使用者延遲。</p>'
    if result.get('job_chain'):
        jobs=result['job_chain'];complete=sum(j['status']=='completed' for j in jobs)
        out+=f'<p>封存作業完成 {complete}/{len(jobs)}。這是作業數，不是等量計算進度；不據此虛構 ETA。</p><details><summary>GPU 作業與啟停時間</summary>'
        out+=table(jobs,[('GPU',lambda r:r['gpu']),('模型／設定',lambda r:r['model']+'/'+r['policy']),('seed',lambda r:r['seed']),('狀態',lambda r:r['status']),('啟動 UTC',lambda r:r.get('started_utc','—')),('結束 UTC',lambda r:r.get('finished_utc','—')),('log',lambda r:r.get('log','—'))])+'</details>'
    if comparisons:out+='<details><summary>所有配對 CI、每領域差距及驗收判定</summary>'+detail(comparisons)+'</details>'
    out+='<h2 id="search">開發搜尋與驗證</h2><section><h3>所有已完成開發候選</h3>'+runs(result.get('dev',[]))+cost_plot(result.get('dev',[]))+'</section><details><summary>單 seed 驗證，包含未通過的候選</summary>'+runs(result.get('validation',[]))+'</details><section><h3>三 seed 確認</h3>'+runs(result.get('validation_confirm',[]))+'</section><details open><summary>決策與停止原因</summary>'+detail(result.get('decisions',[]))+detail(result.get('stopping'))+'</details>'
    out+='<h2 id="context">上下文長度、來源與位置</h2><section>'+context_panel(result.get('context'))+'</section><h2 id="failures">哪些能力仍不足</h2><section>'+detail(result.get('diagnostics') or '最終失敗分類尚未完成；開發觀察不冒充封存測試結論。')+'</section>'
    if result.get('code_budget'):
        out+='<details><summary>程式：同一條軌跡在 1／2／3 次呼叫上限的結果與成本</summary>'+detail(result['code_budget'])+'</details>'
    if result.get('numeric_format_diagnostic'):
        out+='<details open><summary>事後格式診斷：不改動封存主分數</summary>'+detail(result['numeric_format_diagnostic'])+'</details>'
    out+='<h2 id="goal">完整 Goal 與可重現性</h2><details><summary>事先固定的目標、假設、比較、數量、預算與驗收門檻</summary>'+detail(goal)+'</details>'
    for name in ['amendments','policies','data_audit','model_integrity','data_environment']:
        p=HERE/(name+'.json')
        if p.exists():out+='<details><summary>'+name+'</summary>'+detail(json.loads(p.read_text()))+'</details>'
    out+='<details><summary>工程檢查與 final selection seal</summary>'+detail(result.get('engineering'))+detail(result.get('final_selection'))+'</details><h2 id="probes">固定開發案例與生成軌跡</h2>'+gallery(result.get('fixed_dev_gallery',[]))
    out+='<h2>借鑑來源與適用界線</h2><section><p><a href="https://github.com/itayinbarr/little-coder/tree/89d4fa0af864230527ab75d12604ed0eb320e6df">little-coder 固定版本</a>提供按需小卡、證據與錯誤回饋的設計參考；<a href="https://github.com/Doorman11991/smallcode/tree/624970cce81657fd2751ee7f11ba85f35f2bb5c3">SmallCode 固定版本</a>提供路由與檢索參考。此實驗另做中文檢索、真實執行與來源雜湊驗證，不把其他專案整體分數當成單一元件的因果效果。</p><p><a href="https://arxiv.org/abs/2607.08186">HiddenDecoding</a>與<a href="https://arxiv.org/abs/2607.27919">MemoryDecoder</a>涉及學習過的額外機制，不等同於把現有 MiniCPM 上下文拉長就取得相同能力。本研究沒有採用需要訓練的新權重，也沒有隱藏額外大模型成本。</p><p>資料來源：<a href="https://github.com/DRCKnowledgeTeam/DRCD">DRCD</a>、<a href="https://huggingface.co/datasets/openai/gsm8k">GSM8K</a>、<a href="https://huggingface.co/datasets/google-research-datasets/mbpp">MBPP</a>、<a href="https://huggingface.co/datasets/ikala/tmmluplus">TMMLU+</a>。公開 benchmark 的預訓練重疊未知；只排除已知舊實驗題與可偵測近似重複。程式題為參考程式可執行子集，工具題是合成單次呼叫，閱讀提供含支持來源的候選文件包，均不代表完整真實世界能力。</p></section>'
    out+='<footer>Generated UTC: <time id="generated">'+now+'</time><br>模型、資料、程式、短卡與原始輸出雜湊可回查。硬體：8 × NVIDIA H200 NVL；每個生成程序單 GPU、BF16。頁面離線可開啟，無 CDN 或外部追蹤。表頭可點選排序，數字不依賴 hover。</footer>'
    out+='''<script>document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>{let asc=false;th.onclick=()=>{asc=!asc;let rs=Array.from(t.tBodies[0].rows);rs.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent,n=Number(x),m=Number(y);let d=Number.isNaN(n)||Number.isNaN(m)?x.localeCompare(y):n-m;return asc?d:-d});rs.forEach(r=>t.tBodies[0].appendChild(r))}}));let age=(Date.now()-Date.parse(document.getElementById('generated').textContent))/3600000;if(age>24){let e=document.getElementById('stale');e.style.display='block';e.textContent='這是保存的研究快照，距產生時間 '+Math.floor(age)+' 小時；不是即時工作監控。'}</script></main></html>'''
    (ROOT/'docs/test-time-study.html').write_text(out)
