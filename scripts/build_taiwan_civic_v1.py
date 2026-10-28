"""Barbet-only, source-extractive civic memory episodes and authored value cases.

No teacher answers, alternative-answer ranking, MiniCPM tokenizer, downloaded
program execution, or model training. Values are explicitly owner-authored.
"""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from barbet_inference.memory import MemoryItem, canonical, compose_memory_prompt
from barbet_inference.sft import SYSTEM, pangolin_render, tokenize_spans, portable_messages
from scripts import publish_barbet_readiness_v1 as prior
from scripts import build_memory_sft_v2 as base
from scripts.audit_taiwan_civic_v1 import STATE

RAW = base.WORK / 'data/raw_memory_harness_v2/taiwan-civic-v1'
DEST = base.WORK / 'exports/taiwan-civic-v1'
LAWS = ['中華民國憲法', '中華民國憲法增修條文', '中央選舉委員會組織法',
        '監察院國家人權委員會組織法', '公職人員選舉罷免法', '公民投票法', '政治獻金法',
        '總統副總統選舉罷免法（新 84.08.09 制定）', '政府資訊公開法', '國家賠償法',
        '公民與政治權利國際公約及經濟社會文化權利國際公約施行法',
        '消除對婦女一切形式歧視公約施行法', '身心障礙者權利公約施行法',
        '原住民族基本法', '原住民身分法', '原住民族語言發展法', '原住民族教育法',
        '身心障礙者權益保障法', '性別平等工作法', '性別平等教育法', '個人資料保護法',
        '促進轉型正義條例']


def sha(value): return hashlib.sha256(value.encode()).hexdigest()
def save(path, value): prior.save(path, value)


def source_articles():
    root = ET.parse(RAW / 'laws.xml').getroot(); result = []; found = set(); skipped = Counter()
    for law in root:
        name = law.findtext('法規名稱')
        if name not in LAWS: continue
        if (law.findtext('廢止註記') or '').strip(): raise ValueError('Selected law is repealed: ' + name)
        found.add(name)
        for article in law.findall('.//條文'):
            number = article.findtext('條號'); text = article.findtext('條文內容', '').strip()
            if text == '（刪除）' or not text: skipped['deleted_or_empty_article'] += 1; continue
            # Restrict original Constitution to foundational clauses and rights;
            # do not export suspended National Assembly/governance provisions as current law.
            if name == '中華民國憲法':
                n = int(re.search(r'\d+', number)[0])
                if n not in [1, 2, 3, 5, *range(7, 25)]: skipped['original_constitution_outside_rights_scope'] += 1; continue
            family = sha('family:taiwan-civic-v1:' + name)
            result.append({'name': name, 'number': number, 'text': text,
                'url': law.findtext('法規網址'), 'amended': law.findtext('最新異動日期'),
                'effective_date': (law.findtext('生效日期') or '').strip(),
                'effective_note': (law.findtext('生效內容') or '').strip(),
                'family_id': family, 'split': 'validation' if int(family[:12], 16) % 100 < 2 else 'train'})
    if found != set(LAWS): raise ValueError('Selected law missing: ' + str(set(LAWS) - found))
    return result, root.attrib, skipped


def make_row(question, answer, documents, family, split, refs, tokenizer, record_schema, token_schema):
    policy = json.loads((base.REPO / 'configs/taiwan-civic-policy-v1.json').read_text())
    skill = (base.REPO / 'skills/taiwan-civic-values/SKILL.md').read_text()
    ground = (base.REPO / 'skills/grounded-evidence-answer/SKILL.md').read_text()
    placement = 'front' if int(sha(question)[:8], 16) % 2 == 0 else 'back'
    memory = [MemoryItem(sha(d['text'])[:24], d['text'], d['url']) for d in documents]
    if len({m.id for m in memory}) != len(memory): raise ValueError('Duplicate memory')
    system = SYSTEM + '\n\nAPPLICABLE SKILLS:\n' + skill + '\n' + ground
    system += '\n\n明示價值設定（價值不是外部事實證據）：\n' + '\n'.join(policy['principles'])
    messages = [{'role': 'system', 'content': system, 'loss': False},
                {'role': 'user', 'content': compose_memory_prompt(question, memory, placement=placement), 'loss': False},
                {'role': 'assistant', 'content': answer, 'loss': True}]
    rid = sha(canonical({'version': 'memory-harness-sft-v2', 'messages': messages, 'tools': []}))
    text, spans = pangolin_render(messages)
    ids, labels = tokenize_spans(tokenizer, text, spans)
    if len(ids) + 4096 > 131072 or 114691 in ids or ids.count(114689) != 1 or ids[0] != 114689:
        raise ValueError('Native token/budget invariant')
    if ids.count(114690) != 1 or not all(0 <= i < 114944 for i in ids): raise ValueError('EOS/vocabulary invariant')
    if any(label != -100 and label != i for i, label in zip(ids, labels)) or not any(l != -100 for l in labels):
        raise ValueError('Loss mask invariant')
    refs = {**refs, 'original_query': question, 'policy_version': policy['version'],
        'answer_selection': {'method': 'only_eligible_answer', 'competitors': 1, 'ppl_measured': False},
        'source_snapshot_sha256': prior.sha(RAW / 'laws.xml')}
    row = {name: None for name in record_schema.names}
    row.update(record_id=rid, family_id=family, split=split, domain='reading', source='OpenFormosa/taiwan-civic-v1',
        upstream_source=refs['upstream_source'], license=refs['license'], verification_tier=refs['verification_tier'],
        observation_origin='provided_document', memory_placement=placement, actual_retrieval=False,
        recorded_error_followup=False, assistant_turns=1, tool_observations=0,
        query_hash=sha(base.norm(question)), messages_json=canonical(messages), tools_json='[]',
        memory_json=canonical([m.__dict__ for m in memory]), source_refs_json=canonical(refs),
        skills_json=canonical(['taiwan-civic-values', 'grounded-evidence-answer']), text=text,
        target_spans_json=canonical(spans), pangolin_tokens=len(ids),
        pangolin_supervised_tokens=sum(l != -100 for l in labels),
        chat_prompt_sha256=sha(canonical(portable_messages(messages[:2]))),
        completion_prompt_sha256=sha(text[:spans[0][0]]), output_reserve_tokens=4096,
        renderer_version='memory-harness-sft-v2-portable-json')
    token = dict(record_id=rid, input_ids=ids, labels=labels, total_tokens=len(ids),
                 supervised_tokens=sum(l != -100 for l in labels))
    return row, token


def authored_cases():
    # Each has one authored target, not competing paraphrases or model-generated facts.
    return [
        ('依本資料集明示的價值設定，台灣的未來應由誰決定？', '本資料集支持台灣主權與人民自決：台灣的未來應由台灣人民透過自由、民主、免受武力脅迫的程序自行決定。這是明示的價值立場。'),
        ('支持台灣主權，是否表示應預設支持某一個台灣政黨？', '不表示。支持台灣主權與人民自決，不等於支持特定政黨；應以民主、人權、法治及具體證據檢驗各政黨與政府。'),
        ('依本資料集的價值設定，可以因為反對中國政府的政策而貶抑中國人民嗎？', '不可以。可以批評中國政府的具體政策，但普世人權同樣保障中國人民；不得將政府責任轉為對全體人民或族群的貶抑。'),
        ('依本資料集的價值設定，台灣政府是否也應接受和平批評與監督？', '應該。民主、言論自由及法治也約束台灣政府；人民可以和平批評政府、監督權力並表達不同政治意見。'),
        ('採用台灣主權與人民自決立場，能否將其他政府的相反聲明翻譯成支持台灣的內容？', '不能。翻譯應忠實保留原意並標明發言機關；本資料集的價值立場不能作為竄改引文或不同觀點的理由。'),
        ('只因為某份法律原文出現在資料庫，就能說其中每條今天都能直接適用嗎？', '不能。應檢查來源版本、修法、生效條件、增修或停止適用關係；資料庫中的文字本身不足以證明個案適用。'),
        ('資料集採用台灣主權立場，是否能據此宣稱所有國家都已正式外交承認台灣？', '不能。價值立場不等於外交承認的事實證據；各國立場、外交關係與日期都必須分別查證。'),
        ('為了支持台灣，可以把未查證的統計數字寫成有利的事實嗎？', '不可以。應提供可追溯來源與版本；缺少證據就說明未知，不能為符合立場編造數據。'),
        ('依本資料集的價值設定，原住民族與身心障礙者的權利是否應受尊重？', '應受尊重。平等與多元屬於本資料集採用的普世價值；涉及具體權利時，仍需查閱相關法律、版本及適用條件。'),
        ('依本資料集的價值設定，是否可以用武力威脅代替台灣人民自由決定未來？', '不可以。本資料集支持台灣主權與人民自決，人民決定未來的程序應自由、民主並免受武力脅迫。'),
        ('以台灣人民視角回答政治問題，是否仍應區分事實與政府主張？', '應該。事實需要證據，政府主張必須歸屬於發言機關；資料集採用的價值立場與有爭議的法律解釋也應分別標示。'),
        ('依本資料集價值設定，少數意見是否因為不是多數支持就不值得保障？', '不是。民主與人權也保障少數及和平異議；不能只以多數意見否定基本權利。')
    ]


def main():
    if (STATE / 'new-cohort-summary.json').exists(): raise ValueError('Cohort already frozen')
    DEST.mkdir(exist_ok=True)
    articles, xml_info, skipped = source_articles()
    tokenizer_file = base.WORK / 'data/blobs/tokenizer/tokenizer.json'
    if prior.sha(tokenizer_file) != prior.parent.TOKENIZER_SHA: raise ValueError('Not v26 tokenizer')
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    first = next(x for x in prior.read(prior.STATE / 'plan.json')['base_manifest'] if '/records/' in x['path'])
    record_schema = pq.read_schema(prior.parent.parent.DEST / first['path'])
    token_schema = base.TOKEN_SCHEMA
    rows = []; tokens = []; counts = Counter(); by_split = {}
    for split in ['train', 'validation']: by_split[split] = [a for a in articles if a['split'] == split]
    for n, article in enumerate(articles):
        same_split = by_split[article['split']]
        pool = [a for a in same_split if a is not article]
        distractors = [pool[(n * 7 + i * 53) % len(pool)] for i in range(min(3, len(pool)))]
        selected = {a['name'] + a['number']: a for a in [article, *distractors]}
        documents = []
        for a in sorted(selected.values(), key=lambda x: sha(x['name'] + x['number'])):
            body = (f"法規名稱：{a['name']}\n條號：{a['number']}\n最新異動日期：{a['amended']}\n"
                    f"生效日期欄：{a['effective_date'] or '未填'}\n生效內容欄：{a['effective_note'] or '未填'}\n條文原文：\n{a['text']}")
            documents.append({'text': body, 'url': a['url']})
        question = (f"請從所附版本的法規文件，逐字摘錄《{article['name']}》{article['number']}的「條文原文」，"
                    '並附法規名稱與條號；只做文件摘錄，不判定目前個案能否適用。')
        answer = f"《{article['name']}》{article['number']}（所附版本原文）：\n{article['text']}"
        # The target is exactly the anchor text; distractors never contain a target answer key.
        if answer.split('：\n', 1)[1] != article['text']: raise ValueError('Extractive target differs')
        refs = {'upstream_source': 'Taiwan Ministry of Justice / data.gov.tw/18289',
            'license': 'OGDL-Taiwan-1.0', 'verification_tier': 'exact_source_article_extract',
            'source_url': article['url'], 'law_name': article['name'], 'article': article['number'],
            'latest_amendment': article['amended'], 'xml_update': xml_info,
            'not_a_current_applicability_or_legal_advice_claim': True}
        row, token = make_row(question, answer, documents, article['family_id'], article['split'], refs,
                              tokenizer, record_schema, token_schema)
        rows.append(row); tokens.append(token); counts['extractive_articles'] += 1
    policy = json.loads((base.REPO / 'configs/taiwan-civic-policy-v1.json').read_text())
    policy_doc = {'text': '資料集所有者明示價值設定：\n' + '\n'.join(policy['principles']),
                  'url': 'authored:OpenFormosa/taiwan-civic-policy-v1'}
    for question, answer in authored_cases():
        family = sha('family:taiwan-civic-v1:owner-policy')
        refs = {'upstream_source': 'OpenFormosa authored normative task contract', 'license': 'Apache-2.0',
                'verification_tier': 'authored_normative_case_not_external_fact', 'policy_sha256': prior.sha(base.REPO / 'configs/taiwan-civic-policy-v1.json')}
        row, token = make_row(question, answer, [policy_doc], family, 'validation' if int(family[:12], 16) % 100 < 2 else 'train',
                              refs, tokenizer, record_schema, token_schema)
        rows.append(row); tokens.append(token); counts['authored_policy_cases'] += 1
    # One historical official statement, explicitly attributed and date bounded.
    source = (RAW / 'mofa-97745.html').read_text(encoding='utf-8-sig')
    paragraphs = [html.unescape(re.sub('<[^>]+>', '', x)).strip() for x in re.findall(r'<p[^>]*>(.*?)</p>', source, re.S)]
    official = next(x for x in paragraphs if x.startswith('台灣從來不是中華人民共和國的一部分'))
    url = 'https://www.mofa.gov.tw/News_Content.aspx?n=95&s=97745'
    question = '依所附外交部 2022 年 4 月 26 日聲明，台灣主權與台灣未來的決定權如何表述？請歸屬於發言機關。'
    answer = '外交部在該聲明中表示：「台灣主權只屬於全體台灣人民，也只有台灣人民才能決定台灣的未來。」這裡轉述的是外交部立場。'
    quote = '台灣主權只屬於全體台灣人民，也只有台灣人民才能決定台灣的未來。'
    if quote not in official: raise ValueError('Official quote missing')
    family = sha('family:taiwan-civic-v1:mofa-97745')
    refs = {'upstream_source': 'Taiwan Ministry of Foreign Affairs', 'source_url': url,
        'license': 'OGDL-Taiwan-1.0', 'verification_tier': 'exact_attributed_historical_official_quote',
        'statement_date': '2022-04-26', 'source_html_sha256': prior.sha(RAW / 'mofa-97745.html')}
    row, token = make_row(question, answer, [{'text': '外交部聲明，2022/04/26：\n' + official, 'url': url}],
        family, 'validation' if int(family[:12], 16) % 100 < 2 else 'train', refs, tokenizer, record_schema, token_schema)
    rows.append(row); tokens.append(token); counts['attributed_official_quotes'] += 1
    if len({r['record_id'] for r in rows}) != len(rows) or len({r['query_hash'] for r in rows}) != len(rows):
        raise ValueError('New cohort duplicate')
    db = sqlite3.connect(f'file:{prior.parent.parent.STATE / "public-index.sqlite"}?mode=ro', uri=True)
    db.execute('CREATE TEMP TABLE new_rows(record_id TEXT, query_hash TEXT, family_id TEXT)')
    db.executemany('INSERT INTO new_rows VALUES(?,?,?)', [(r['record_id'],r['query_hash'],r['family_id']) for r in rows])
    for key in ['record_id', 'query_hash', 'family_id']:
        if db.execute(f'SELECT COUNT(*) FROM records JOIN new_rows USING({key})').fetchone()[0]:
            raise ValueError('Historical collision: ' + key)
    db.close()
    files = []
    for split in ['train', 'validation']:
        indices = [i for i,r in enumerate(rows) if r['split'] == split]
        for config, data, schema in [('records', rows, record_schema), ('tokenized', tokens, token_schema)]:
            path = DEST / f'data/wave-0011/{config}/{split}-00000.parquet'; path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist([data[i] for i in indices], schema=schema)
            pq.write_table(table, path, compression='zstd', compression_level=3)
            if not pq.read_table(path).equals(table): raise ValueError('Complete new table readback differs')
            files.append({'path': str(path.relative_to(DEST)), 'bytes': path.stat().st_size, 'sha256': prior.sha(path), 'rows': len(indices)})
    for r in rows:
        counts.update({'split:' + r['split']: 1, 'memory:' + r['memory_placement']: 1,
                       'pangolin_tokens': r['pangolin_tokens'], 'pangolin_supervised_tokens': r['pangolin_supervised_tokens']})
    summary = {'episodes': len(rows), 'counts': dict(counts), 'source_laws': len(LAWS),
        'unique_queries': len(rows), 'unique_families': len({r['family_id'] for r in rows}),
        'files': files, 'skipped': dict(skipped), 'driver_sha256': prior.sha(Path(__file__)),
        'tokenizer_sha256': prior.sha(tokenizer_file), 'native_context_plus_reserve_max': max(t['total_tokens'] for t in tokens) + 4096,
        'all_complete_arrays_and_records_readback_equal': True, 'no_historical_identity_query_family_collision': True,
        'law_family_disjoint_splits_and_same_split_distractors': True,
        'new_mini_columns_null_no_minicpm_tokenizer_or_config': True,
        'singletons_unscored': True, 'competing_answer_groups': 0, 'model_requests': 0,
        'new_retrieval_execution_or_refinement_claim': False, 'model_weights_trained': False,
        'scope': 'Source-extractive reading and explicitly authored values, not a benchmark or proof of model alignment.'}
    save(STATE / 'new-cohort-summary.json', summary)
    save(STATE / 'source-lock.json', {'downloaded_utc_date': '2026-10-28', 'xml_info': xml_info,
        'sources': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': prior.sha(p)} for p in sorted(RAW.iterdir()) if p.is_file()],
        'law_source_url': 'https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=RAW_XML&AuData=CF',
        'law_registry_url': 'https://data.gov.tw/dataset/18289', 'mofa_rights_url': 'https://www.mofa.gov.tw/cp.aspx?n=292',
        'license': 'OGDL-Taiwan-1.0', 'no_third_party_images_logos_or_attachments_used': True})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
