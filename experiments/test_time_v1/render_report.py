#!/usr/bin/env python3
"""Offline HTML goal/results report. Missing evidence stays visibly pending."""
from pathlib import Path
import datetime as dt
import html
import json

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def render(value):
    if isinstance(value, dict):
        return '<table>' + ''.join('<tr><th>' + html.escape(str(k)) + '</th><td>' + render(v) + '</td></tr>' for k, v in value.items()) + '</table>'
    if isinstance(value, list):
        return '<ol>' + ''.join('<li>' + render(x) + '</li>' for x in value) + '</ol>'
    return html.escape(str(value))


def main():
    goal = json.loads((EXP / 'goal.json').read_text())
    result_path = EXP / 'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'pending', 'measured_improvement': None, 'note': '尚未產生本輪模型實測；規格與工程測試不能當成模型能力提升。'}
    sections = ''.join(f'<details open><summary>{html.escape(k)}</summary>{render(v)}</details>' for k, v in goal.items())
    doc = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>固定權重：skill、RAG 與執行回饋實驗</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f5f6f8;color:#17202a;margin:auto;padding:32px;max-width:1200px}h1{line-height:1.3}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #d8dfe5;padding:10px;text-align:left;vertical-align:top}th{width:24%;background:#eaf0f6}details{background:white;padding:18px;margin:16px 0;border-radius:8px}summary{font-weight:bold;cursor:pointer}li{margin:8px 0}.notice{padding:16px;background:#fff4d6}footer{font-size:13px;color:#52616b}code{overflow-wrap:anywhere}</style><h1>固定權重：skill、RAG 與執行回饋實驗</h1><p>MiniCPM5-1B → 在明確任務與成本範圍內評估 Qwen3.5-9B 差距</p><p class="notice">不訓練、不修改權重、不使用更大生成模型代答。實作完成、測試通過、模型提升與接近 9B 是四個不同結論。</p>'''
    doc += '<h2>目前證據</h2>' + render(results) + '<h2>完整 Goal 與事先定義的實驗規格</h2>' + sections
    for name in ['amendments', 'policies', 'probe_selection']:
        path = EXP / (name + '.json')
        if path.exists():
            doc += '<details><summary>' + name + '</summary>' + render(json.loads(path.read_text())) + '</details>'
    doc += '<footer>Generated UTC: ' + dt.datetime.now(dt.timezone.utc).isoformat() + '<br>來源：experiments/test_time_v1/goal.json；所有預算和門檻是待驗證的工程選擇。新測試尚未開封時，不提供推估達標時間。</footer></html>'
    (ROOT / 'docs/test-time-study.html').write_text(doc)


if __name__ == '__main__':
    main()
