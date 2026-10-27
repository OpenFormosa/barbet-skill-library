#!/usr/bin/env python3
"""Prepare public inference inputs separately from evaluation-only labels.

No model answers are used for selection. Never print final test questions/gold.
Public-source passages are a permissible knowledge corpus, not a QA answer bank.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from testtime.execution import run_python
from testtime.retrieval import Document, LexicalIndex
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer

WORK = Path(__file__).resolve().parent / 'work'
PINS = {
    'drcd': 'b944790de5af02c5fbb7cd9cb1473d27d169eebf',
    'openai/gsm8k': '740312add88f781978c0658806c59bc2815b9866',
    'google-research-datasets/mbpp': '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
    'ikala/tmmluplus': '639a64db464b4661feb91817b9edfe43932bdf18',
}
COUNTS = {'dev': 32, 'validation': 48, 'test': 128}
SEED = 20261027


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def old_template(text):
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()
    return sha(re.sub(r'\d+(?:[.,]\d+)*', '<N>', text))


class NearFilter:
    def __init__(self, prior, candidates):
        self.offset = len(prior)
        self.indices = list(range(self.offset))
        self.matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
            max_features=150_000, sublinear_tf=True).fit_transform(prior + candidates)
        self.rejections = 0

    def accept(self, i):
        j = self.offset + i
        if self.indices:
            scores = (self.matrix[j] @ self.matrix[self.indices].T).data
            if len(scores) and float(scores.max()) >= .90:
                self.rejections += 1
                return False
        self.indices.append(j)
        return True


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in value)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError('refusing to overwrite different frozen data: ' + str(path))
    path.write_text(payload)
    return {'path': str(path.relative_to(WORK)), 'rows': len(value), 'sha256': sha(payload)}


def parquet(repo, path):
    file = hf_hub_download(repo, path, repo_type='dataset', revision=PINS[repo])
    return pq.read_table(file).to_pylist()


def request(ident, prompt, contract, **kwargs):
    return {'id': ident, 'prompt': prompt, 'output_contract': contract, **kwargs}


def previous_questions():
    files = list(Path('/root/minicpm-skill-study/data').glob('*.jsonl'))
    files += [Path('/root/barbet-qwen-benchmark/runs/20261024-v26/questions-v1.jsonl')]
    rows = []
    for p in files:
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x]
    return rows


def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('selected_ids.json'))
    parser.add_argument('--original-selection', action='store_true',
        help='Repeat the original exclusion search; requires the archived earlier-study inputs.')
    args = parser.parse_args()
    WORK = args.work
    selection = None if args.original_selection else json.loads(args.selection.read_text()) if args.selection.exists() else None
    selected = {s: set(v) for s, v in selection['ids'].items()} if selection else None
    WORK.mkdir(parents=True, exist_ok=True)
    data = {s: [] for s in COUNTS}
    labels = {s: [] for s in COUNTS}
    diagnostics = {'pins': PINS, 'seed': SEED, 'selection_uses_model_outputs': False,
        'public_pretraining_overlap': 'unknown; this is not a decontaminated-pretraining claim',
        'exclusions': {}, 'license_notice': 'Source datasets retain their terms; downloaded texts/gold are not committed to the skill library.'}
    old = previous_questions() if selected is None else []
    old_texts = [r.get('user_prompt', r.get('question', r.get('prompt', ''))) for r in old]
    old_templates = {re.sub(r'\d+', '#', normalized(t)) for t in old_texts}
    old_template_hashes = {r.get('metadata', {}).get('template_sha256') for r in old}
    old_template_hashes |= {old_template(r.get('question', '')) for r in old if r.get('question')}
    seen = set()

    def add(split, req, domain, gold, group, **extra):
        key = normalized(req['prompt'])
        if key in seen:
            raise ValueError('duplicate question')
        seen.add(key)
        data[split].append(req)
        labels[split].append({'id': req['id'], 'domain': domain, 'gold': gold, 'group': group, **extra})

    # Traditional-Chinese Wikipedia passages: no question/answer strings in corpus.
    documents, qa = {}, []
    for part in ['training', 'dev', 'test']:
        target = WORK / 'sources' / f'DRCD_{part}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f'https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/{PINS["drcd"]}/DRCD_{part}.json'
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        raw = json.loads(target.read_text())
        for article in raw['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                text = paragraph['context']
                did = 'drcd-doc-' + sha(text)[:20]
                documents[did] = Document(did, text, f'DRCD@{PINS["drcd"]}/{part}', title)
                for q in paragraph['qas']:
                    if q.get('answers'):
                        qa.append((part, title, did, q))
    doc_list = sorted(documents.values(), key=lambda d: d.id)
    all_ids = [d.id for d in doc_list]
    grouped = {}
    for item in qa:
        grouped.setdefault(item[1], []).append(item)
    titles = sorted(grouped)
    random.Random(f'{SEED}:drcd').shuffle(titles)
    pointer = 0
    for split, count in COUNTS.items():
        for _ in range(count):
            title = titles[pointer]
            pointer += 1
            candidates = sorted(grouped[title], key=lambda x: x[3]['id'])
            part, _, did, q = random.Random(f'{SEED}:{title}').choice(candidates)
            rng = random.Random(f'{SEED}:{q["id"]}')
            pool = rng.sample([x for x in all_ids if x != did], 31) + [did]
            rng.shuffle(pool)
            ident = 'drcd:' + str(q['id'])
            add(split, request(ident, q['question'] + '\n依提供的來源回答，只輸出答案本身。',
                               'short_answer', document_ids=pool),
                'drcd', [x['text'] for x in q['answers']], 'drcd:' + title,
                sufficient_document_id=did, source_part=part)
    diagnostics['drcd'] = {'documents': len(doc_list), 'available_titles': len(titles),
        'split_method': 'article-title disjoint, one question/article; official partitions pooled and re-split',
        'packet': '32 source passages including the support passage; deterministic shuffled order; distractor passages contain no QA labels',
        'limitation': 'Known-answerable extractive QA, not evidence-absence detection or unrestricted web QA'}
    write(WORK / 'corpus.jsonl', [d.__dict__ for d in doc_list])

    # Math: new train questions for development, previously unused official test.
    excluded = 0
    math_pools = {s: parquet('openai/gsm8k', f'main/{s}-00000-of-00001.parquet') for s in ['train', 'test']}
    math_near = None if selected else NearFilter([t for t, r in zip(old_texts, old) if r.get('task') == 'gsm8k'],
        [r['question'] for part in ['train', 'test'] for r in math_pools[part]])
    old_ids = {str(r.get('id')) for r in old if r.get('task') == 'gsm8k'}
    used_math = set()
    for split, count in COUNTS.items():
        part = 'test' if split == 'test' else 'train'
        indices = list(range(len(math_pools[part])))
        random.Random(f'{SEED}:math:{split}').shuffle(indices)
        picked = 0
        for i in indices:
            row = math_pools[part][i]
            if selected and f'gsm8k:{part}:{i}' not in selected[split]:
                continue
            key = re.sub(r'\d+', '#', normalized(row['question']))
            if ((part, i) in used_math or key in old_templates or old_template(row['question']) in old_template_hashes
                    or (part == 'test' and f'gsm8k:{i}' in old_ids)):
                excluded += 1
                continue
            if math_near is not None and not math_near.accept(i + (len(math_pools['train']) if part == 'test' else 0)):
                continue
            used_math.add((part, i))
            old_templates.add(key)
            add(split, request(f'gsm8k:{part}:{i}', row['question'] + '\nEnd with #### and the final numerical answer.', 'number'),
                'gsm8k', row['answer'].rsplit('####', 1)[-1].strip(), 'math:' + sha(key))
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise RuntimeError('insufficient new math questions')
    diagnostics['exclusions']['math_exact_ids_and_numeric_templates'] = excluded
    diagnostics['exclusions']['math_char_ngram_near_duplicates'] = math_near.rejections if math_near else 'frozen-ID replay; see committed original data_audit.json'

    # MBPP: one public test; remaining tests are scorer-only. Verify references
    # in the same constrained executor before accepting a task, independently of
    # all model outputs. Report every exclusion instead of calling this full MBPP.
    code_exclusions = []
    code_used = set()
    code_used_ids = set()
    code_pools = {p: parquet('google-research-datasets/mbpp', f'sanitized/{p}-00000-of-00001.parquet')
                  for p in ['train', 'validation', 'test']}
    code_all = [r for p in ['train', 'validation', 'test'] for r in code_pools[p]]
    code_pos = {r['task_id']: i for i, r in enumerate(code_all)}
    code_near = None if selected else NearFilter([], [r.get('prompt', r.get('text', '')) for r in code_all])
    for split, count in COUNTS.items():
        part = {'dev': 'train', 'validation': 'validation', 'test': 'test'}[split]
        rows = list(code_pools[part])
        if split == 'validation':
            rows += code_pools['train']  # sanitized validation alone has <48 tasks
        random.Random(f'{SEED}:mbpp:{split}').shuffle(rows)
        picked = 0
        for row in rows:
            text = row.get('prompt', row.get('text', ''))
            tests = row['test_list']
            setup = '\n'.join(row.get('test_imports', []))
            ident = 'mbpp:' + str(row['task_id'])
            if selected and ident not in selected[split]:
                continue
            if row['task_id'] in code_used_ids or normalized(text) in code_used or len(tests) < 2:
                code_exclusions.append({'id': ident, 'reason': 'duplicate_or_insufficient_checks'})
                continue
            receipt = run_python(row['code'], tests, setup)
            if not receipt['passed']:
                code_exclusions.append({'id': ident, 'reason': receipt['status'], 'error': receipt.get('error', '')})
                continue
            if code_near is not None and not code_near.accept(code_pos[row['task_id']]):
                code_exclusions.append({'id': ident, 'reason': 'near_duplicate'})
                continue
            code_used.add(normalized(text))
            code_used_ids.add(row['task_id'])
            prompt = text + '\nImplement the function in Python. Return only Python code.\nPublic example:\n' + tests[0]
            add(split, request(ident, prompt, 'python', public_tests=tests[:1], setup=setup),
                'mbpp', {'tests': tests[1:], 'setup': setup}, 'mbpp:' + str(row['task_id']))
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise RuntimeError(f'insufficient executable MBPP {split}: {picked}/{count}')
    diagnostics['exclusions']['mbpp_reference_execution'] = code_exclusions
    diagnostics['mbpp_split'] = 'dev from sanitized train; validation from unused sanitized train/validation; final from sanitized test; reference-executable subset only; exact/near duplicates excluded'

    # Typed tool family holdout: two functions and five arguments, synthetic.
    families = [
        ('catalogue', '目錄匯出', 'compress', '是否壓縮輸出', '壓縮', '不要壓縮'),
        ('queue', '佇列建立', 'persistent', '是否持久保存', '持久保存', '不要持久保存'),
        ('audio', '音訊轉換', 'normalize', '是否正規化音量', '正規化音量', '不要正規化音量'),
        ('ticket', '票券整理', 'reverse', '是否反向排列', '反向排列', '維持原順序'),
        ('image', '影像輸出', 'grayscale', '是否轉成灰階', '轉成灰階', '保留彩色'),
        ('archive', '封存工作', 'encrypted', '是否加密', '加密', '不要加密'),
        ('table', '資料表輸出', 'distinct', '是否移除重複項目', '移除重複項目', '保留所有重複項目'),
        ('log', '紀錄匯出', 'include_header', '是否包含標頭', '包含標頭', '不要包含標頭'),
        ('invoice', '單據產生', 'rounded', '是否四捨五入', '四捨五入', '保留原始精度'),
        ('report', '報表輸出', 'ascending', '是否由小到大排序', '由小到大排序', '由大到小排序'),
        ('message', '訊息發佈', 'notify', '是否發出通知', '發出通知', '靜默處理，不發通知'),
        ('sample', '樣本整理', 'allow_missing', '是否允許缺漏', '允許缺漏', '拒絕缺漏'),
    ]
    for split, count in COUNTS.items():
        family_ids = {'dev': range(0, 3), 'validation': range(3, 6), 'test': range(6, 12)}[split]
        for i in range(count):
            f = list(family_ids)[i % len(family_ids)]
            tag, desc, boolean, bdesc, yes, no = families[f]
            rng = random.Random(f'{SEED}:{split}:tool:{i}')
            suffix = sha(f'{SEED}:{split}:{i}')[:7]
            fname = tag + '_' + suffix
            flag = bool(rng.randrange(2))
            vals = [rng.randint(-9, 40) for _ in range(3)]
            gold = {'record_id': f'案-{suffix} Ａ', 'values': vals, 'limit': rng.randint(2, 19),
                    'factor': rng.choice([0.25, 0.5, 1.5, 2.25]), boolean: flag}
            properties = {'record_id': {'type': 'string', 'description': '原樣保留指定識別碼，包括空白、全形字元'},
                'values': {'type': 'array', 'items': {'type': 'integer'}}, 'limit': {'type': 'integer'},
                'factor': {'type': 'number'}, boolean: {'type': 'boolean', 'description': bdesc}}
            schema = {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
            tools = [{'type': 'function', 'function': {'name': fname, 'description': desc + '正式作業', 'parameters': schema}},
                     {'type': 'function', 'function': {'name': 'preview_' + suffix, 'description': '僅預覽，不執行正式作業', 'parameters': schema}}]
            rng.shuffle(tools)
            prompt = f'請執行{desc}正式作業，使用識別碼「{gold["record_id"]}」、數列 {vals}、上限 {gold["limit"]}、倍率 {gold["factor"]}；{yes if flag else no}。只呼叫適用工具一次。'
            add(split, request(f'tool:{split}:{i}', prompt, 'tool_call', tools=tools),
                'tools', {'name': fname, 'arguments': gold}, 'tool-family:' + tag)
    diagnostics['tools'] = {'family_disjoint': True, 'limitation': 'Synthetic single-call typed semantics; not a full agent benchmark or a real external side effect.'}

    # Broad knowledge retention: fresh subjects, no oracle subject in public input.
    api = HfApi()
    names = sorted(s.rfilename for s in api.dataset_info('ikala/tmmluplus', revision=PINS['ikala/tmmluplus']).siblings if s.rfilename.endswith('_test.csv'))
    old_mc_ids = {str(r.get('id')) for r in old if r.get('task') == 'tmmluplus'}
    mc_candidates = []
    mc_by_subject = {}
    for name in names:
        subject = Path(name).name.removesuffix('_test.csv')
        path = hf_hub_download('ikala/tmmluplus', name, repo_type='dataset', revision=PINS['ikala/tmmluplus'])
        rows = list(csv.reader(open(path)))
        if rows and rows[0][0].lower() == 'question':
            rows = rows[1:]
        mc_by_subject[subject] = (rows, len(mc_candidates))
        mc_candidates += [r[0] + '\n' + '\n'.join(f'{k}. {v}' for k, v in zip('ABCD', r[1:5])) for r in rows]
    mc_near = None if selected else NearFilter([t for t, r in zip(old_texts, old) if r.get('task') == 'tmmluplus'], mc_candidates)
    for subject, (rows, mc_offset) in mc_by_subject.items():
        indices = list(range(len(rows)))
        random.Random(f'{SEED}:mc:{subject}').shuffle(indices)
        for split, count in [('dev', 1), ('validation', 1), ('test', 2)]:
            picked = 0
            while indices and picked < count:
                i = indices.pop()
                row = rows[i]
                ident = f'tmmluplus:{subject}:{i}'
                if selected and ident not in selected[split]:
                    continue
                text = row[0] + '\n' + '\n'.join(f'{k}. {v}' for k, v in zip('ABCD', row[1:5]))
                key = re.sub(r'\d+', '#', normalized(text))
                if ident in old_mc_ids or key in old_templates or old_template(text) in old_template_hashes:
                    continue
                if mc_near is not None and not mc_near.accept(mc_offset + i):
                    continue
                old_templates.add(key)
                add(split, request(ident, text + '\n只輸出正確選項的英文字母 A、B、C 或 D。', 'choice'),
                    'tmmluplus', row[5].strip(), 'subject:' + subject)
                picked += 1
            if picked != count:
                raise RuntimeError('insufficient fresh subject questions')
    diagnostics['exclusions']['tmmlu_char_ngram_near_duplicates'] = mc_near.rejections if mc_near else 'frozen-ID replay; see committed original data_audit.json'

    manifest = []
    for split in COUNTS:
        order = list(range(len(data[split])))
        random.Random(f'{SEED}:shuffle:{split}').shuffle(order)
        data[split] = [data[split][i] for i in order]
        labels[split] = [labels[split][i] for i in order]
        manifest.append(write(WORK / f'{split}.inputs.jsonl', data[split]))
        manifest.append(write(WORK / f'{split}.labels.jsonl', labels[split]))
    diagnostics['manifest'] = manifest
    if selection:
        if manifest != selection['manifest']:
            raise RuntimeError('replayed dataset hashes differ from original frozen selection')
        if sha((WORK / 'corpus.jsonl').read_bytes()) != selection['corpus_sha256']:
            raise RuntimeError('replayed corpus hash differs')
        diagnostics['replayed_frozen_ids'] = True
    diagnostics['test_status'] = 'sealed: do not read test labels or generate test answers before final selection'
    (WORK / 'data_audit.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(json.dumps({'event': 'data_prepared', 'counts': {s: len(v) for s, v in data.items()},
        'corpus_documents': len(doc_list), 'manifest': manifest}, ensure_ascii=False))


if __name__ == '__main__':
    main()
