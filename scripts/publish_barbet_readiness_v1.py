"""Bounded-cache query pruning, verified staging, and atomic main publication.

Original exports, scores and reviews stay immutable. Only generated staging
cache files are reclaimed, after anonymous byte verification at immutable commits.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import fcntl

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, DatasetCard

from scripts import publish_barbet_only_v1 as parent
from scripts import audit_barbet_continuation_v1 as audit
from scripts import resume_v26_recovery_readback_v2 as bounded

STATE = parent.base.WORK / 'reports/memory-harness-v2-barbet-readiness-v1'
DEST = parent.base.WORK / 'exports/memory-harness-v2-barbet-readiness-v1'
CACHE = STATE / 'verified-staging-cache'
BRANCH = 'quality-v26-20261028-v1'
FOLDER = 'reports/v26-readiness-v1'
BOUNDS = [4096, 8192, 16384, 32768, 65536, 131072]


def save(path, value): parent.base.save(path, value)
def read(path): return parent.read(path)
def sha(path): return parent.base.file_sha(path)


def removed_stats(row):
    result = Counter({'accepted': 1, 'split:' + row['split']: 1,
                      'pangolin_tokens': row['pangolin_tokens'],
                      'pangolin_supervised_tokens': row['pangolin_supervised_tokens'],
                      'assistant_turns': row['assistant_turns'], 'tool_observations': row['tool_observations'],
                      'actual_retrieval': int(row['actual_retrieval']),
                      'recorded_error_followup': int(row['recorded_error_followup'])})
    for key in ['domain', 'source', 'upstream_source', 'license', 'verification_tier',
                'memory_placement', 'observation_origin']:
        result[key + ':' + row[key]] += 1
    result['length_bucket:' + str(next(b for b in BOUNDS if row['pangolin_tokens'] <= b))] += 1
    refs = json.loads(row['source_refs_json'])
    result['selection:' + refs['answer_selection']['method']] += 1
    if refs.get('query_schema_id'): result['query_schema:' + refs['query_schema_id']] += 1
    return result


def scan(item, holds):
    pa.set_cpu_count(1)
    path = parent.parent.DEST / item['path']
    table = pq.read_table(path, columns=['record_id', 'query_hash', 'source', 'domain',
                                         'pangolin_tokens', 'pangolin_supervised_tokens'])
    groups = table.group_by(['source', 'domain']).aggregate([
        ('record_id', 'count'), ('pangolin_tokens', 'sum'), ('pangolin_supervised_tokens', 'sum')]).to_pylist()
    selected = np.flatnonzero(np.asarray(pc.is_in(table['query_hash'], value_set=pa.array(holds))))
    if not len(selected): return {'path': item['path'], 'groups': groups, 'excluded': 0}
    columns = [name for name in pq.read_schema(path).names if name not in
               ['messages_json', 'tools_json', 'text', 'mini_text', 'memory_json', 'skills_json',
                'target_spans_json', 'mini_target_spans_json']]
    removed = pq.read_table(path, columns=columns).take(pa.array(selected)).to_pylist()
    counts = Counter()
    removed_groups = {}
    for row in removed:
        counts.update(removed_stats(row))
        key = row['source'] + '|' + row['domain']
        removed_groups.setdefault(key, Counter()).update({'rows': 1, 'physical_tokens': row['pangolin_tokens'],
                                                         'supervised_tokens': row['pangolin_supervised_tokens']})
    return {'path': item['path'], 'groups': groups, 'excluded': len(selected),
            'exclude_indices': selected.tolist(), 'counts': dict(counts),
            'removed_ids': [r['record_id'] for r in removed],
            'removed_groups': {k: dict(v) for k, v in removed_groups.items()}}


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'plan.json').exists(): raise ValueError('Correction plan already frozen')
    if shutil.disk_usage(parent.base.WORK).free < 100 * 1024**3 + 2 * 1024**3:
        raise ValueError('Bounded staging cache reserve unavailable')
    prior = read(parent.STATE / 'publication-final-commit.json')
    if HfApi(token=False).repo_info(parent.DATASET, repo_type='dataset').sha != prior['final_head']:
        raise ValueError('Main changed before correction')
    old = read(parent.STATE / 'publication-plan.json')
    holds_file = audit.STATE / 'new-query-holds-v1.json'; holds = read(holds_file)['query_hashes']
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent.base.REPO, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=parent.base.REPO, text=True).strip():
        raise ValueError('Correction implementation must be committed')
    save(STATE / 'policy.json', {'driver_sha256': sha(Path(__file__)), 'library_commit': code,
        'parent': prior['final_head'], 'holds_sha256': sha(holds_file), 'max_cache_bytes': 2 * 1024**3,
        'minimum_free_bytes': 100 * 1024**3, 'max_rewrite_workers': 4,
        'source_arrays_reviews_and_scores_immutable': True, 'only_row_deletion_no_runner_up': True,
        'cache_reclaimed_only_after_pinned_anonymous_SHA256_readback': True})
    records = [x for x in old['manifest'] if '/records/' in x['path']]
    scans = []; removed = Counter(); groups = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, part in enumerate(pool.map(lambda x: scan(x, holds), records), 1):
            groups.extend(part.pop('groups')); removed.update(part.get('counts', {})); scans.append(part)
            if n % 100 == 0 or n == len(records):
                print(json.dumps({'scanned': n, 'total': len(records), 'rows_to_remove': removed['accepted']}), flush=True)
    db = sqlite3.connect(f'file:{parent.parent.STATE / "public-index.sqlite"}?mode=ro', uri=True)
    db.execute('PRAGMA temp_store=MEMORY'); db.execute('CREATE TEMP TABLE held(query_hash TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO held VALUES (?)', [(h,) for h in holds])
    count = db.execute('SELECT COUNT(*) FROM records WHERE query_hash IN (SELECT query_hash FROM held)').fetchone()[0]
    if count != removed['accepted']: raise ValueError('Index/query-wide deletion coverage differs')
    families = db.execute('SELECT COUNT(DISTINCT family_id) FROM records WHERE query_hash NOT IN (SELECT query_hash FROM held)').fetchone()[0]
    found_queries = db.execute('SELECT COUNT(DISTINCT query_hash) FROM records WHERE query_hash IN (SELECT query_hash FROM held)').fetchone()[0]
    if found_queries != len(holds): raise ValueError('A held query was not found in the parent')
    db.close()
    source = read(parent.STATE / 'summary.json'); counts = Counter(source['counts']); counts.subtract(removed)
    if any(v < 0 for k, v in counts.items() if k in removed): raise ValueError('Negative retained metric')
    counts['excluded:new_readiness_query_holds'] = removed['accepted']
    unique = {'record_id': counts['accepted'], 'query_hash': source['unique_counts']['query_hash'] - len(holds),
              'family_id': families}
    if counts['accepted'] < 10_000_000: raise ValueError('Correction falls below user count target')
    jobs = [x for x in scans if x['excluded']]
    lookup = {x['path']: x for x in old['manifest']}
    for job in jobs:
        job['input_bytes'] = sum(lookup[p]['bytes'] for p in [job['path'], job['path'].replace('/records/', '/tokenized/')])
    weights = {}
    for group in groups:
        key = group['source'] + '|' + group['domain']; current = weights.setdefault(key, Counter())
        current.update({'rows': group['record_id_count'], 'physical_tokens': group['pangolin_tokens_sum'],
                        'supervised_tokens': group['pangolin_supervised_tokens_sum']})
    retained_weights = {key: Counter(value) for key, value in weights.items()}
    for part in scans:
        for key, value in part.get('removed_groups', {}).items(): retained_weights[key].subtract(value)
    for field, expected in [('rows', counts['accepted']), ('physical_tokens', counts['pangolin_tokens']),
                            ('supervised_tokens', counts['pangolin_supervised_tokens'])]:
        if sum(v[field] for v in retained_weights.values()) != expected: raise ValueError('Retained token mix differs')
    for prefix in ['source:', 'domain:', 'length_bucket:', 'selection:']:
        if sum(v for k,v in counts.items() if k.startswith(prefix)) != counts['accepted']:
            raise ValueError('Retained episode accounting differs: ' + prefix)
    summary = {'counts': dict(counts), 'unique_counts': unique, 'removed_counts': dict(removed),
               'held_queries': len(holds), 'held_rows': removed['accepted'], 'changed_record_shards': len(jobs),
               'parent': prior, 'library_commit': code, 'whole_query_holds_no_runner_up': True,
               'retained_row_values_and_scores_unchanged': True, 'training_protocol': 'barbet_portable_v1',
               'original_source_domain_token_mix': {k: dict(v) for k, v in weights.items()},
               'source_domain_token_mix': {k: dict(v) for k, v in retained_weights.items()}}
    save(STATE / 'summary.json', summary)
    save(STATE / 'parent-tree.json', parent.tree(prior['final_head']))
    save(STATE / 'plan.json', {'parent': prior['final_head'], 'parent_plan_sha256': sha(parent.STATE / 'publication-plan.json'),
                              'holds_sha256': sha(holds_file), 'jobs': jobs, 'base_manifest': old['manifest'],
                              'library_commit': code, 'branch': BRANCH})
    print(json.dumps({'episodes_after_pruning': counts['accepted'], 'removed_rows': removed['accepted'],
                      'changed_record_shards': len(jobs), 'changed_input_bytes': sum(j['input_bytes'] for j in jobs)}), flush=True)


def rewrite(job):
    outputs = []
    for path in [job['path'], job['path'].replace('/records/', '/tokenized/')]:
        src = parent.parent.DEST / path; table = pq.read_table(src)
        keep = np.ones(len(table), dtype=bool); keep[job['exclude_indices']] = False
        expected = table.filter(pa.array(keep))
        target = CACHE / path; target.parent.mkdir(parents=True, exist_ok=True)
        if not len(expected):
            outputs.append({'path': path, 'delete': True}); continue
        pq.write_table(expected, target, compression='zstd', compression_level=3)
        reread = pq.read_table(target)
        if not reread.equals(expected): raise ValueError('Retained values changed on rewrite')
        outputs.append({'path': path, 'bytes': target.stat().st_size, 'sha256': sha(target), 'rows': len(expected)})
    if outputs[0].get('rows') != outputs[1].get('rows'): raise ValueError('Rewritten views misaligned')
    return {'source': job['path'], 'excluded': job['excluded'], 'files': outputs,
            'all_retained_columns_and_full_arrays_equal': True}


def stage():
    plan = read(STATE / 'plan.json'); api = HfApi()
    if sha(audit.STATE / 'new-query-holds-v1.json') != plan['holds_sha256']: raise ValueError('Holds changed')
    initial_path = STATE / 'staging-initial.json'
    if not initial_path.exists():
        api.create_branch(parent.DATASET, branch=BRANCH, revision=plan['parent'], repo_type='dataset', exist_ok=False)
        text = '# INCOMPLETE QUALITY STAGING — DO NOT TRAIN\n\nThis branch is being built; main remains at the parent until verification finishes.\n'
        result = api.create_commit(parent.DATASET, repo_type='dataset', revision=BRANCH, parent_commit=plan['parent'],
            operations=[CommitOperationAdd('README.md', text.encode()), CommitOperationDelete('PUBLICATION.json')],
            commit_message='Start incomplete bounded-cache quality correction staging')
        save(initial_path, {'head': result.oid})
    head = read(initial_path)['head']; jobs = plan['jobs']; batches = []; current = []; size = 0
    for job in jobs:
        if current and (len(current) >= 4 or size + job['input_bytes'] > 1024**3):
            batches.append(current); current = []; size = 0
        current.append(job); size += job['input_bytes']
    if current: batches.append(current)
    bounded.STATE = STATE
    for number, batch in enumerate(batches):
        receipt = STATE / 'staging-batches' / f'{number:04d}.json'
        if receipt.exists():
            data = read(receipt)
            if data['source_paths'] != [j['path'] for j in batch] or data['parent'] != head:
                raise ValueError('Staging resume binding changed')
            for report in data['reports']:
                for item in report['files']:
                    local = CACHE / item['path']
                    if local.exists() and not item.get('delete'):
                        checked = read(STATE / 'staged-readback' / data['head'] /
                                       (hashlib.sha256(item['path'].encode()).hexdigest() + '.json'))
                        if checked['sha256'] != item['sha256'] or sha(local) != item['sha256']:
                            raise ValueError('Interrupted cache reclamation binding differs')
                        local.unlink()
            head = data['head']; continue
        if api.repo_info(parent.DATASET, repo_type='dataset', revision=BRANCH).sha != head:
            raise ValueError('Staging head changed; preserve and diagnose')
        if shutil.disk_usage(parent.base.WORK).free < 100 * 1024**3 + 2 * 1024**3:
            raise ValueError('Staging cache reserve unavailable')
        with ThreadPoolExecutor(max_workers=4) as pool: reports = list(pool.map(rewrite, batch))
        files = [f for part in reports for f in part['files']]
        cache_bytes = sum(f.get('bytes', 0) for f in files)
        if cache_bytes > 2 * 1024**3: raise ValueError('Bounded cache exceeded')
        operations = [CommitOperationDelete(f['path']) if f.get('delete') else
                      CommitOperationAdd(f['path'], CACHE / f['path']) for f in files]
        result = api.create_commit(parent.DATASET, repo_type='dataset', revision=BRANCH, parent_commit=head,
            operations=operations, num_threads=4, commit_message=f'Prune frozen query holds from shard batch {number}')
        revision = result.oid
        checks = STATE / 'staged-readback' / revision; checks.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda f: bounded.verify({k: f[k] for k in ['path', 'bytes', 'sha256']}, revision, checks),
                          [f for f in files if not f.get('delete')]))
        payload = {'head': revision, 'parent': head, 'source_paths': [j['path'] for j in batch],
                   'reports': reports, 'all_new_payload_bytes_anonymously_verified': True,
                   'cache_bytes_reclaimed_after_verification': cache_bytes}
        save(receipt, payload)
        for f in files:
            if f.get('delete'): continue
            local = CACHE / f['path']
            if not local.resolve().is_relative_to(CACHE.resolve()) or sha(local) != f['sha256']:
                raise ValueError('Reclaim path/hash changed')
            local.unlink()
        head = revision
        print(json.dumps({'staged_batches': number + 1, 'total': len(batches),
                          'verified_payload_files': sum(not f.get('delete') for f in files),
                          'cache_reclaimed_bytes': cache_bytes}), flush=True)
    save(STATE / 'staging-complete.json', {'head': head, 'batches': len(batches), 'all_changed_shards_verified': True})


def metadata(plan, files):
    summary = read(STATE / 'summary.json'); source = read(parent.STATE / 'summary.json')
    report = {**source, **summary}
    scan = dict(source['barbet_token_scan'])
    for key, old in [('rows', 'accepted'), ('tokens', 'pangolin_tokens'), ('supervised_tokens', 'pangolin_supervised_tokens'),
                     ('split:train', 'split:train'), ('split:validation', 'split:validation')]: scan[key] = summary['counts'][old]
    for key, value in summary['counts'].items():
        if key.startswith('length_bucket:'): scan[key] = value
    report['barbet_token_scan'] = scan
    text = parent.card(report, plan['library_commit'])
    start, end = text.index('The prior release received full anonymous'), text.index('Original full-record')
    text = text[:start] + f'''This revision holds **{summary['held_rows']:,} episodes** across **{summary['held_queries']:,} queries**.
Affected Parquet shards are rewritten only to remove those rows; every retained
column and complete token/label array is compared exactly. Changed files receive
full anonymous SHA-256 readback on fixed staging commits. Main receives the same
objects by server-side copy; unchanged objects inherit the prior verification
chain. New metadata is separately downloaded and hashed. This is composed
verification, not a new full historical byte rescan. See [the quality report]({FOLDER}/report.html).

''' + text[end:]
    text = text.replace('Existing records and Barbet arrays\nare retained byte-for-byte; no question, target, split or PPL selection changed.',
                        'Retained row values and Barbet arrays are unchanged; affected Parquet files are rewritten only to remove held queries. No runner-up answer is substituted.')
    text += '''
## Continuation protocol

The exported training format is **barbet_portable_v1**, not the checkpoint's native
Jinja tool format. Use `BarbetSFTCollator` to preserve masks and right-pad by real
sequence length. Do not enable cross-example packing without verifying both
attention and Mamba state boundaries. The model shifts labels once.
Use `prepare_dataset_request(..., mode="barbet_completion")` or
`ContextAgent(..., prompt_format="barbet_portable_v1")` for matching inference.
Native v26 PPL ranking remains the frozen original criterion; it is not a claim
that the exported portable text has minimum PPL under a different template.

The fresh audit read every original token/label value, checked 186 stratified full
records and 522 causal target prefixes, and independently recomputed saved minima
for 586,883 candidate groups. Eleven manually inspected diagnostic packets exposed
additional defects; all seven first nonaccepts remain query-wide held. Those small
diagnostic counts are not a corpus error rate. The verified row-subset correction
removes native PAD occurrences and the new source-bound quality holds. Other
unreviewed semantic errors remain possible. Begin with a controlled continuation
pilot and frozen capability-retention evaluation before a full training run.
'''
    DatasetCard(text).validate(); (DEST / 'README.md').write_text(text)
    config = read(parent.DEST / 'recipe/barbet-training-v1.json')
    config.update(version=2, training_protocol='barbet_portable_v1', padding_side='right',
                  collator='barbet_inference.training.BarbetSFTCollator', packing=False,
                  inference_request_mode='barbet_completion', context_agent_prompt_format='barbet_portable_v1',
                  attention_mask_from_sequence_lengths=True, literal_native_PAD_allowed=False)
    save(DEST / 'recipe/barbet-training-v2.json', config)
    (DEST / 'TRAINING.md').write_text('''# Barbet continuation data

```python
from datasets import load_dataset
from barbet_inference.training import BarbetSFTCollator
train = load_dataset("OpenFormosa/barbet-long-context-sft", "tokenized", split="train", streaming=True)
collator = BarbetSFTCollator()
```

Use input_ids and the supplied labels without retokenizing or applying a second
chat template. The collator preserves assistant-only -100 masks and right-pads
labels with -100. The released Barbet model shifts labels once internally.
Cross-example packing is disabled until attention and Mamba state resets are
verified. This is data/collation setup, not an instruction to start training.

The tokenizer matches v26, but the exported portable JSON tool protocol differs
from its native chat template. For inference use prepare_dataset_request with
mode="barbet_completion", or ContextAgent with prompt_format="barbet_portable_v1".
Do not silently substitute the old native template after this SFT.
Frozen native-v26 PPL ranking is preserved; it was not measured on portable text.

The quality revision excludes entire held queries without answer substitution.
Original snapshots and first failures remain reproducible in history. The dataset
is mechanically validated and supports a controlled pilot; semantic correctness
and model improvement are not guaranteed by lower PPL or matching labels.
''')
    save(DEST / FOLDER / 'summary.json', summary)
    holds = read(audit.STATE / 'new-query-holds-v1.json')
    save(DEST / FOLDER / 'query-holds.json', {'query_hashes': holds['query_hashes'],
         'reason_findings': holds['reason_findings'], 'all_first_nonaccepts_held': True,
         'no_runner_up_substitution': True})
    for filename in ['array-summary.json', 'content-sample-summary.json', 'saved-selection-and-folds.json',
                     'barbet-live-format-equivalence.json', 'manual-first-review-v1.json']:
        value = read(audit.STATE / filename)
        if filename == 'array-summary.json': value.pop('errors', None)
        save(DEST / FOLDER / filename, value)
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>v26 續訓資料品質修訂</title><style>body{{font:17px/1.8 system-ui;max-width:1100px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>v26 續訓資料品質修訂</h1><p>本次重新讀取全部 10,328,696 筆原始 Barbet arrays，未發現 vocab 越界、labels 位移、非空 supervision、BOS/EOS 或長度預算錯誤；另找到 62 筆正文含原生 PAD 的問題。全部 PAD query、首次內容非通過案例及來源規則所涵蓋的 query 持續隔離，共移除 {summary['held_rows']:,} 筆，保留 {summary['counts']['accepted']:,} 筆。</p><p>來源規則包含未提供訂位知識的 task574 對話補空家族、工具回答中的星期日期矛盾，以及宣告 JSON 介面卻輸出舊式 action 文字的案例。這些是保守隔離範圍，不代表每條來源都已證明答案錯誤。沒有改用次低 PPL 回答。</p><p>186 筆分層樣本的完整文字、token IDs、labels、工具順序與 522 個答案前綴均吻合；37 個有限紀錄查詢及 16 次實際關係工具 observation 重建通過。小型人工式 agent 檢視共 11 題，首次 7 個非通過保留為 hold，不能據此推估全資料錯誤率。所有 287 個 UNK 皆對應來源字面 &lt;unk&gt;，沒有一般字元被 UNK 取代的證據。</p><p>原生 v26 PPL 保存分數共 1,563,256 筆、586,883 組選答最低值重新核算通過；這包括歷史候選群組，不等同現役 scored episode 數。原始 query/family 沒有跨 split；刪除子集合不新增洩漏。原生排名模板與本訓練 portable 模板不同，排名條件照原規則保留，不宣稱 portable 字串的 PPL 也最小。</p><p>續訓需使用 BarbetSFTCollator 保留 labels、右側 padding、模型內 shift 一次；跨樣本 packing 在 attention 與 Mamba reset 未驗證前關閉。推論使用 barbet_completion 或 ContextAgent 的 barbet_portable_v1。這些介面通過 186 筆、522 個真實 target prefix 的完全相同 IDs 檢查。</p><p>檔案以固定 staging commit 完整匿名讀取 SHA-256，再由相同 Git 物件發布至 main；未變動檔案繼承既有驗證鏈。沒有重新下載所有歷史資料的宣稱。可以開始有退化檢查的小額續訓試跑；完整續訓與能力增益仍需實際 trainer、獨立留存集與回歸評估。本次沒有訓練權重或執行來源程式。</p><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k!='original_source_domain_token_mix'},ensure_ascii=False,indent=2))}</pre></html>'''
    (DEST / FOLDER / 'report.html').write_text(body)


def publish():
    plan = read(STATE / 'plan.json'); stage = read(STATE / 'staging-complete.json'); api = HfApi()
    if (STATE / 'publication-data-commit.json').exists(): raise ValueError('Correction already published')
    if api.repo_info(parent.DATASET, repo_type='dataset').sha != plan['parent']:
        raise ValueError('Main changed during staging')
    if api.repo_info(parent.DATASET, repo_type='dataset', revision=BRANCH).sha != stage['head']:
        raise ValueError('Staging branch changed')
    files = [f for p in sorted((STATE / 'staging-batches').glob('*.json')) for report in read(p)['reports'] for f in report['files']]
    metadata(plan, files)
    changes = {f['path']: f for f in files}; manifest = {x['path']: x for x in plan['base_manifest']}
    operations = []
    for path, item in changes.items():
        if item.get('delete'):
            manifest.pop(path); operations.append(CommitOperationDelete(path))
        else:
            manifest[path] = {k: item[k] for k in ['path', 'bytes', 'sha256']}
            operations.append(CommitOperationCopy(path, path, src_revision=stage['head']))
    new_metadata = []
    for path in sorted(DEST.rglob('*')):
        if not path.is_file(): continue
        relative = str(path.relative_to(DEST)); item = {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)}
        manifest[relative] = item; new_metadata.append(item)
        operations.append(CommitOperationAdd(relative, path))
    operations.append(CommitOperationDelete('PUBLICATION.json'))
    save(STATE / 'publication-plan.json', {'manifest': sorted(manifest.values(), key=lambda x:x['path']),
        'changed_payloads': files, 'new_metadata': new_metadata, 'parent': plan['parent'],
        'staging_head': stage['head'], 'library_commit': plan['library_commit']})
    result = api.create_commit(parent.DATASET, repo_type='dataset', parent_commit=plan['parent'],
        operations=operations, num_threads=4, commit_message='Quarantine readiness query defects; preserve retained answers and Barbet supervision')
    save(STATE / 'publication-data-commit.json', {'commit': result.oid, 'parent': plan['parent'],
        'plan_sha256': sha(STATE / 'publication-plan.json')})
    print(json.dumps({'data_commit': result.oid}), flush=True)


def verify():
    plan = read(STATE / 'publication-plan.json'); data = read(STATE / 'publication-data-commit.json')
    if data['plan_sha256'] != sha(STATE / 'publication-plan.json'): raise ValueError('Manifest changed')
    revision = data['commit']; current = parent.tree(revision); previous = read(STATE / 'parent-tree.json')
    stage = parent.tree(plan['staging_head']); metadata_paths = {f['path'] for f in plan['new_metadata']}
    changed_paths = {f['path'] for f in plan['changed_payloads']}
    expected = {f['path'] for f in plan['manifest']} | {'.gitattributes'}
    if set(current) != expected or any('/minicpm5_tokenized/' in p for p in current): raise ValueError('Main tree mismatch')
    for path in expected - metadata_paths:
        other = stage if path in changed_paths else previous
        if current[path] != other[path]: raise ValueError('Main object differs from verified source: ' + path)
    bounded.STATE = STATE; folder = STATE / 'main-metadata-readback'; folder.mkdir(exist_ok=True)
    for item in plan['new_metadata']: bounded.verify(item, revision, folder)
    summary = read(STATE / 'summary.json'); c = summary['counts']; u = summary['unique_counts']
    proof = {'dataset': parent.DATASET, 'data_commit': revision, 'parent': data['parent'],
        'active_new_episodes': c['accepted'], 'active_legacy_rows': 0, 'goal_complete': c['accepted'] >= 10_000_000,
        'unique_normalized_first_queries': u['query_hash'], 'unique_task_families': u['family_id'],
        'active_training_configs': ['tokenized','records'], 'default_config': 'tokenized',
        'training_model': 'Barbet', 'training_protocol': 'barbet_portable_v1', 'tokenizer_sha256': parent.TOKENIZER_SHA,
        'held_queries': summary['held_queries'], 'removed_episodes': summary['held_rows'],
        'native_PAD_in_retained_real_content': 0, 'literal_UNK_not_loss_of_source_characters': True,
        'verification_method': 'unchanged_parent_objects_plus_full_staging_byte_readback_and_main_copy_object_identity_plus_new_metadata_bytes',
        'anonymous_full_file_sha256_readback': False,
        'changed_payload_files_anonymously_verified': sum(not f.get('delete') for f in plan['changed_payloads']),
        'retained_unchanged_objects_verified': len(expected - metadata_paths - changed_paths),
        'verified_files': len(plan['manifest']), 'verified_bytes': sum(f['bytes'] for f in plan['manifest']),
        'staging_head': plan['staging_head'], 'library_commit': plan['library_commit'],
        'all_retained_row_values_and_arrays_exact': True, 'all_first_nonaccept_queries_held': True,
        'no_runner_up_answer_substitution': True, 'model_weights_trained': False, 'model_efficacy_measured': False,
        'readiness': 'Mechanically validated for a controlled continuation pilot with the documented collator and matching inference protocol; not a guarantee of semantic accuracy or model improvement.'}
    save(DEST / 'PUBLICATION.json', proof); api = HfApi()
    if api.repo_info(parent.DATASET, repo_type='dataset').sha != revision: raise ValueError('Head changed before proof')
    result = api.create_commit(parent.DATASET, repo_type='dataset', parent_commit=revision,
        operations=[CommitOperationAdd('PUBLICATION.json', DEST / 'PUBLICATION.json')],
        commit_message='Record verified quality pruning and conditional v26 continuation readiness')
    head = result.oid
    save(STATE / 'publication-metadata-commit.json', {'final_head': head, 'publication_sha256': sha(DEST/'PUBLICATION.json')})
    post = STATE / 'proof-readback'; post.mkdir(exist_ok=True)
    bounded.verify({'path':'PUBLICATION.json','bytes':(DEST/'PUBLICATION.json').stat().st_size,
                    'sha256':sha(DEST/'PUBLICATION.json')}, head, post)
    final = parent.tree(head)
    if set(final) != expected | {'PUBLICATION.json'} or any(final[p] != current[p] for p in current):
        raise ValueError('Proof commit changed data')
    save(STATE / 'publication-final-commit.json', {**proof, 'final_head': head, 'publication_receipt_anonymous_verified': True})
    print(json.dumps({'final_head':head,'episodes':c['accepted'],'held_rows':summary['held_rows']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare','stage','publish','verify'])
    args = parser.parse_args()
    with (parent.base.WORK/'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); globals()[args.phase]()
