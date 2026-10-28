"""Publish the user's Barbet-only training view as an auditable deletion revision.

Retained payloads inherit the previous full byte readback through exact Git object
identity. Only new metadata is downloaded again; this is not a new full rescan.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, DatasetCard, HfApi

from scripts import build_memory_sft_v2 as base
from scripts import publish_conceptnet_relations_v1 as parent
from scripts import resume_v26_recovery_readback_v2 as bounded

DATASET = 'OpenFormosa/barbet-long-context-sft'
STATE = base.WORK / 'reports/memory-harness-v2-barbet-only-v1'
DEST = base.WORK / 'exports/memory-harness-v2-barbet-only-v1'
FOLDER = 'reports/barbet-only-v1'
TOKENIZER_SHA = 'f84754e42cc4b08d5573a6b7306635e86ed04fff68ee707662d76d7044917176'
MODEL_REVISION = '67131067ff1825179d5e38e91d499159669e9c66'


def read(path):
    return json.loads(path.read_text())


def tree(revision):
    result = {}
    for item in HfApi(token=False).list_repo_tree(DATASET, repo_type='dataset',
                                               revision=revision, recursive=True):
        if hasattr(item, 'blob_id'):
            result[item.path] = {'blob_id': item.blob_id, 'bytes': item.size}
    return result


def scan_tokens(item):
    path = parent.DEST / item['path']
    if base.file_sha(path) != item['sha256']:
        raise ValueError('Barbet tokenized payload changed: ' + item['path'])
    table = pq.read_table(path, columns=['total_tokens', 'supervised_tokens'])
    lengths = table['total_tokens']
    counts = {'rows': len(table), 'tokens': pc.sum(lengths).as_py(),
              'supervised_tokens': pc.sum(table['supervised_tokens']).as_py()}
    if counts['rows'] != item['rows'] or pc.max(lengths).as_py() + 4096 > 131072:
        raise ValueError('Barbet count or context budget differs')
    lower = 0
    for bound in [4096, 8192, 16384, 32768, 65536, 131072]:
        counts['length_bucket:' + str(bound)] = pc.sum(pc.and_(pc.greater(lengths, lower),
                                                               pc.less_equal(lengths, bound))).as_py()
        lower = bound
    counts['split:' + item['split']] = len(table)
    return counts


def card(summary, code):
    c, u = summary['counts'], summary['unique_counts']
    sources = '\n'.join(f'| {k[7:]} | {v:,} |' for k, v in sorted(c.items()) if k.startswith('source:'))
    buckets = '\n'.join(f'| ≤ {k.split(":")[1]} | {v:,} |' for k, v in sorted(
        summary['barbet_token_scan'].items(), key=lambda x: int(x[0].split(':')[1]) if x[0].startswith('length_bucket:') else 0)
        if k.startswith('length_bucket:'))
    return f'''---
license: other
license_name: mixed-source-licenses
license_link: https://huggingface.co/datasets/{DATASET}/blob/main/NOTICE.md
task_categories:
- text-generation
language:
- en
- zh
configs:
- config_name: tokenized
  default: true
  data_files:
  - split: train
    path: data/wave-*/tokenized/train-*.parquet
  - split: validation
    path: data/wave-*/tokenized/validation-*.parquet
- config_name: records
  data_files:
  - split: train
    path: data/wave-*/records/train-*.parquet
  - split: validation
    path: data/wave-*/records/validation-*.parquet
---

# Barbet memory, skills and harness SFT

**{c['accepted']:,} accepted new episodes**, **{u['query_hash']:,} distinct normalized first queries**,
and **{u['family_id']:,} task families**. Train: **{c['split:train']:,}**; validation:
**{c['split:validation']:,}**. Active legacy rows: **0**.

The training target is **Barbet**. The default `tokenized` config contains the existing
Barbet/Pangolin `input_ids` and assistant-only `labels`. Its tokenizer JSON SHA-256 is
`{TOKENIZER_SHA}`, identical to the frozen [Barbet v26 tokenizer](https://huggingface.co/voidful/barbet-skill/tree/{MODEL_REVISION}).
There is no MiniCPM tokenized config in this revision. See [TRAINING.md](TRAINING.md).

```python
from datasets import load_dataset
train = load_dataset("{DATASET}", "tokenized", split="train", streaming=True)
```

The `records` config retains source-attributed messages, tools, memory, full Barbet
training text and provenance. Its historical auxiliary `mini_*`/`minicpm_*` columns
remain as provenance and are not needed for Barbet training. Loading `tokenized`
does not load or require a MiniCPM tokenizer. Existing records and Barbet arrays
are retained byte-for-byte; no question, target, split or PPL selection changed.

## Standard answers

Multiple eligible answers to an identical complete input use the lowest conditional
supervised-answer PPL under frozen native Barbet v26. Supervised assistant tokens
and assistant EOS are scored; prompt, memory, observations, unsupervised history
and padding are excluded. Native batch size is 1 without truncation. Every
competitor must have a finite score; equal scores use stable record-ID order.
Different literal inputs or observations are separate conditions.

**{c['selection:minimum_v26_conditional_answer_ppl']:,} episodes** were selected from
scored alternatives; **{c['selection:only_eligible_answer']:,}** have one eligible answer
and are explicitly unscored. Source and quality holds remain in force. A newly
held selected question is removed entirely, without substituting a runner-up.
Low PPL measures likelihood, not truth.

## Barbet training size

| Measure | Count |
|---|---:|
| Complete Barbet tokens | {c['pangolin_tokens']:,} |
| Supervised Barbet tokens | {c['pangolin_supervised_tokens']:,} |
| Assistant turns | {c['assistant_turns']:,} |
| Tool observations | {c['tool_observations']:,} |

The following buckets are recomputed from **Barbet token lengths only**. Each row
belongs to its smallest fitting bucket. All rows retain 4,096 output tokens within
the existing 131,072-token export budget. Unrelated padding and truncation are not used.

| Complete Barbet token upper bound | Episodes |
|---|---:|
{buckets}

| Source | Episodes |
|---|---:|
{sources}

## Memory and inference

Memory is supplied before the answer, at the front/back of user context or through
actual registered-tool observations. Prompt, memory and tool observations have
labels `-100`; only supervised assistant positions train the model. Causal
trainers shift labels once. Do not retokenize existing IDs or apply another chat template.

The [skill library and harness](https://github.com/OpenFormosa/barbet-skill-library/tree/{code})
provide answer-free dataset requests, skill retrieval, explicit typed callbacks,
portable relation snapshots and bounded verifier-driven refinement. Provenance-only
construction traces must not be added to prompts. To evaluate retrieval from the
start, select the first supervised assistant turn, not an answer after future observations.

## Validation and limits

The prior release received full anonymous file-byte SHA-256 readback. This revision
removes {summary['removed_files']:,} MiniCPM tokenized files ({summary['removed_bytes']:,} bytes).
Every retained file is checked against the parent Git object and the prior byte
manifest. New metadata is anonymously downloaded and hashed. This is composed
verification from the fixed parent, not a new full historical byte rescan. See
[PUBLICATION.json](PUBLICATION.json) and [the revision report]({FOLDER}/summary.json).
Older reports describe their frozen historical releases and may mention three configs.

Original full-record, complete array, assistant mask, source-fold, query-fold and
answer-free Chat/Completion budget checks remain bound to unchanged payloads.
New graph tasks use the source training graph; SFT splits by anchor may share
other context facts and do not form entity-disjoint graph evaluation. Source-code
examples were not executed. Sampled semantic diagnoses are not corpus accuracy
estimates; finite-record SQL verification does not certify current-world truth.
Structured-query rows use a small number of query schemas over different anchors.
Training mixtures should consider source, task type and supervised-token budgets.

All first holds, source contracts and per-record licenses remain attached. See
[NOTICE.md](NOTICE.md). No model weights were trained and no new model capability
gain or small-model/9B equivalence is established by this format change.
'''


def prepare():
    STATE.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True)
    if (STATE / 'publication-plan.json').exists():
        raise ValueError('Plan is already frozen')
    if shutil.disk_usage(base.WORK).free < 100 * 1024**3 + 128 * 1024**2:
        raise ValueError('Metadata disk reserve unavailable')
    prior = read(parent.STATE / 'publication-final-commit.json')
    old = read(parent.STATE / 'publication-plan.json')
    summary = read(parent.STATE / 'summary.json')
    checks_path = parent.STATE / 'anonymous-readback/summary.json'
    if not prior['anonymous_full_file_sha256_readback'] or base.file_sha(checks_path) != prior['readback_sha256']:
        raise ValueError('Parent full-byte proof differs')
    checks = {x['path']: x for x in read(checks_path)['checks']}
    if len(checks) != len(old['manifest']):
        raise ValueError('Parent readback coverage differs')
    for item in old['manifest']:
        if checks[item['path']] != {**item, 'revision': prior['data_commit'], 'anonymous': True}:
            raise ValueError('Parent byte receipt differs')
    if HfApi(token=False).repo_info(DATASET, repo_type='dataset').sha != prior['final_head']:
        raise ValueError('HF head changed before Barbet-only preparation')
    parent_tree = tree(prior['final_head'])
    if set(parent_tree) != {x['path'] for x in old['manifest']} | {'.gitattributes', 'PUBLICATION.json'}:
        raise ValueError('Parent public tree differs')
    if base.file_sha(base.WORK / 'data/blobs/tokenizer/tokenizer.json') != TOKENIZER_SHA:
        raise ValueError('Barbet tokenizer mismatch')
    removed = [x for x in old['manifest'] if '/minicpm5_tokenized/' in x['path']]
    if not removed or any(not x['path'].startswith('data/') for x in removed):
        raise ValueError('Deletion scope is not exclusively MiniCPM data')
    retained = [x for x in old['manifest'] if x not in removed and x['path'] != 'README.md']
    tokens = [x for x in summary['files'] if x['config'] == 'tokenized']
    scan = Counter()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, counts in enumerate(pool.map(scan_tokens, tokens), 1):
            scan.update(counts)
            if n % 100 == 0 or n == len(tokens):
                print(json.dumps({'barbet_files_checked': n, 'total': len(tokens)}), flush=True)
    c = summary['counts']
    for field, expected in [('rows', c['accepted']), ('tokens', c['pangolin_tokens']),
                            ('supervised_tokens', c['pangolin_supervised_tokens']),
                            ('split:train', c['split:train']), ('split:validation', c['split:validation'])]:
        if scan[field] != expected:
            raise ValueError('Barbet aggregate differs: ' + field)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=base.REPO, text=True).strip():
        raise ValueError('Publish implementation must be committed first')
    active_counts = {k: v for k, v in c.items() if not k.startswith(('minicpm_', 'length_bucket:'))}
    active_counts.update({k: v for k, v in scan.items() if k.startswith('length_bucket:')})
    report = {'counts': active_counts, 'unique_counts': summary['unique_counts'],
              'active_training_configs': ['tokenized', 'records'], 'default_config': 'tokenized',
              'training_model': 'Barbet', 'tokenizer_sha256': TOKENIZER_SHA,
              'barbet_model_revision': MODEL_REVISION, 'barbet_token_scan': dict(scan),
              'removed_config': 'minicpm5_tokenized', 'removed_files': len(removed),
              'removed_bytes': sum(x['bytes'] for x in removed),
              'retained_payloads_unchanged': True, 'answer_selection_unchanged': True,
              'records_auxiliary_historical_columns_retained': True,
              'parent_full_byte_verified_release': prior, 'library_commit': code,
              'verification_method': 'parent_full_byte_readback_plus_retained_git_objects_plus_new_metadata_byte_readback',
              'new_full_historical_byte_rescan': False, 'model_weights_trained': False}
    base.save(STATE / 'summary.json', report)
    base.save(DEST / FOLDER / 'summary.json', report)
    base.save(DEST / FOLDER / 'prior-publication.json', prior)
    base.save(DEST / 'recipe/barbet-training-v1.json', {
        'dataset': DATASET, 'config': 'tokenized', 'model_family': 'Barbet',
        'tokenizer_sha256': TOKENIZER_SHA, 'model_reference_revision': MODEL_REVISION,
        'fields': ['input_ids', 'labels'], 'ignore_index': -100, 'labels_are_unshifted': True,
        'trainer_causal_shift_count': 1, 'max_full_tokens': 126976,
        'output_reserve_tokens': 4096, 'retokenization_required': False,
        'additional_chat_template_required': False, 'minicpm_tokenizer_required': False})
    (DEST / 'TRAINING.md').write_text(f'''# Barbet training data

```python
from datasets import load_dataset
train = load_dataset("{DATASET}", "tokenized", split="train", streaming=True)
validation = load_dataset("{DATASET}", "tokenized", split="validation", streaming=True)
row = next(iter(train))
example = {{"input_ids": row["input_ids"], "labels": row["labels"]}}
```

Use Barbet tokenizer JSON SHA-256 `{TOKENIZER_SHA}` (Barbet v26 revision
`{MODEL_REVISION}`), or explicitly verify tokenizer identity for your Barbet checkpoint.
The exported IDs historically called Pangolin are already the Barbet training IDs.
Keep the supplied assistant-only labels: prompt, memory and observations are `-100`.
Choose a collator that pads labels with `-100` and preserves existing masks.
Shift causal labels exactly once in the trainer/model. A collator that regenerates
labels from all input IDs would erase the supplied supervision masks.
Do not apply another chat template or tokenize these IDs again.

`records` is the optional source-text/provenance view. Historical auxiliary
MiniCPM columns in that view are not inputs for Barbet training. The active
`minicpm5_tokenized` config has been removed. No MiniCPM tokenizer is required.
These instructions describe data loading and masking, not new model training.
''')
    text = card(report, code); DatasetCard(text).validate(); (DEST / 'README.md').write_text(text)
    manifest = retained + [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size,
                            'sha256': base.file_sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
    if len({x['path'] for x in manifest}) != len(manifest):
        raise ValueError('Duplicate output path')
    base.save(STATE / 'parent-tree.json', parent_tree)
    base.save(STATE / 'publication-plan.json', {
        'parent_commit': prior['final_head'], 'parent_manifest_sha256': base.file_sha(parent.STATE / 'publication-plan.json'),
        'parent_readback_sha256': prior['readback_sha256'], 'parent_tree_sha256': base.file_sha(STATE / 'parent-tree.json'),
        'library_commit': code, 'manifest': manifest, 'retained': retained, 'removed': removed,
        'new_paths': [str(p.relative_to(DEST)) for p in sorted(DEST.rglob('*')) if p.is_file()]})
    print(json.dumps({'barbet_episodes': c['accepted'], 'removed_files': len(removed),
                      'removed_bytes': report['removed_bytes'], 'retained_and_new_files': len(manifest)}), flush=True)


def publish():
    plan = read(STATE / 'publication-plan.json'); api = HfApi()
    if (STATE / 'publication-data-commit.json').exists():
        raise ValueError('Already committed')
    if api.repo_info(DATASET, repo_type='dataset').sha != plan['parent_commit']:
        raise ValueError('Remote head changed')
    if tree(plan['parent_commit']) != read(STATE / 'parent-tree.json'):
        raise ValueError('Parent objects changed')
    folders = sorted({'/'.join(x['path'].split('/')[:3]) for x in plan['removed']})
    if any(not x.endswith('/minicpm5_tokenized') for x in folders):
        raise ValueError('Non-MiniCPM deletion requested')
    operations = [CommitOperationDelete(x, is_folder=True) for x in folders]
    operations.append(CommitOperationDelete('PUBLICATION.json'))
    for item in plan['manifest']:
        if item['path'] in plan['new_paths']:
            path = DEST / item['path']
            if base.file_sha(path) != item['sha256']:
                raise ValueError('Metadata changed after plan')
            operations.append(CommitOperationAdd(item['path'], path))
    commit = api.create_commit(DATASET, repo_type='dataset', parent_commit=plan['parent_commit'],
                               operations=operations, commit_message='Use Barbet tokenized training by default; remove MiniCPM tokenized exports')
    base.save(STATE / 'publication-data-commit.json', {'commit': commit.oid, 'parent': plan['parent_commit'],
                                                   'plan_sha256': base.file_sha(STATE / 'publication-plan.json')})
    print(json.dumps({'data_commit': commit.oid}), flush=True)


def verify():
    plan = read(STATE / 'publication-plan.json'); data = read(STATE / 'publication-data-commit.json')
    summary = read(STATE / 'summary.json'); revision = data['commit']
    if data['plan_sha256'] != base.file_sha(STATE / 'publication-plan.json'):
        raise ValueError('Frozen plan changed')
    current, prior = tree(revision), read(STATE / 'parent-tree.json')
    expected = {x['path'] for x in plan['manifest']} | {'.gitattributes'}
    if set(current) != expected or any('/minicpm5_tokenized/' in p for p in current):
        raise ValueError('New tree or MiniCPM removal differs')
    if current['.gitattributes'] != prior['.gitattributes']:
        raise ValueError('Repository attributes changed')
    for item in plan['retained']:
        if current[item['path']] != prior[item['path']] or current[item['path']]['bytes'] != item['bytes']:
            raise ValueError('Retained payload object changed: ' + item['path'])
    bounded.STATE = STATE
    receipts = STATE / 'anonymous-readback/files'; receipts.mkdir(parents=True, exist_ok=True)
    additions = [x for x in plan['manifest'] if x['path'] in plan['new_paths']]
    for item in additions:
        bounded.verify(item, revision, receipts)
    byte_checks = [read(receipts / (hashlib.sha256(x['path'].encode()).hexdigest() + '.json')) for x in additions]
    for item, check in zip(additions, byte_checks):
        if check != {**item, 'revision': revision, 'anonymous': True}:
            raise ValueError('New metadata byte receipt differs')
    base.save(STATE / 'verification.json', {'data_commit': revision, 'retained_git_objects_checked': len(plan['retained']),
                                          'new_metadata_byte_checks': byte_checks, 'removed_paths_absent': True,
                                          'parent_full_byte_readback_sha256': plan['parent_readback_sha256']})
    c, u = summary['counts'], summary['unique_counts']
    proof = {'dataset': DATASET, 'data_commit': revision, 'parent': data['parent'],
             'active_new_episodes': c['accepted'], 'active_legacy_rows': 0, 'goal_complete': True,
             'unique_normalized_first_queries': u['query_hash'], 'unique_task_families': u['family_id'],
             'active_training_configs': ['tokenized', 'records'], 'default_config': 'tokenized',
             'training_model': 'Barbet', 'tokenizer_sha256': TOKENIZER_SHA,
             'removed_minicpm_files': summary['removed_files'], 'removed_minicpm_bytes': summary['removed_bytes'],
             'verified_files': len(plan['manifest']), 'verified_bytes': sum(x['bytes'] for x in plan['manifest']),
             'anonymous_full_file_sha256_readback': False,
             'verification_method': summary['verification_method'],
             'parent_full_anonymous_byte_readback_verified': True,
             'retained_parent_git_objects_identical': True,
             'fresh_anonymous_byte_readback_files': len(additions),
             'verification_sha256': base.file_sha(STATE / 'verification.json'),
             'library_commit': plan['library_commit'], 'answer_selection_unchanged': True,
             'answer_selection': 'minimum eligible conditional answer PPL under frozen native Barbet v26; singleton answers explicitly unscored',
             'model_weights_trained': False, 'model_efficacy_measured': False}
    base.save(DEST / 'PUBLICATION.json', proof)
    api = HfApi()
    receipt_path = STATE / 'publication-metadata-commit.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['publication_sha256'] != base.file_sha(DEST / 'PUBLICATION.json'):
            raise ValueError('Proof changed on resume')
        head = receipt['final_head']
    else:
        if api.repo_info(DATASET, repo_type='dataset').sha != revision:
            raise ValueError('Head changed before publication proof')
        result = api.create_commit(DATASET, repo_type='dataset', parent_commit=revision,
                                   operations=[CommitOperationAdd('PUBLICATION.json', DEST / 'PUBLICATION.json')],
                                   commit_message='Record Barbet-only deletion revision verification and unchanged training payloads')
        head = result.oid
        base.save(receipt_path, {'final_head': head, 'publication_sha256': base.file_sha(DEST / 'PUBLICATION.json')})
    post = STATE / 'anonymous-readback/metadata'; post.mkdir(exist_ok=True)
    bounded.verify({'path': 'PUBLICATION.json', 'bytes': (DEST / 'PUBLICATION.json').stat().st_size,
                    'sha256': base.file_sha(DEST / 'PUBLICATION.json')}, head, post)
    final_tree = tree(head)
    if set(final_tree) != expected | {'PUBLICATION.json'} or any(final_tree[p] != current[p] for p in current):
        raise ValueError('Final proof commit changed payloads')
    base.save(STATE / 'publication-final-commit.json', {**proof, 'final_head': head,
                                                     'publication_receipt_anonymous_verified': True})
    print(json.dumps({'final_head': head, 'episodes': c['accepted'], 'default_config': 'tokenized',
                      'removed_minicpm_files': summary['removed_files']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'publish', 'verify'])
    args = parser.parse_args()
    with (base.WORK / 'reports/memory-harness-v2/publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.phase]()
