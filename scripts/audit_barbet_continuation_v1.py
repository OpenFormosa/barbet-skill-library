"""Fresh read-only v26 data audit: all arrays/metadata, bounded content samples.

No model training, source-program execution, score replacement or review replay.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from scripts import publish_barbet_only_v1 as release
from barbet_inference.sft import pangolin_render, tokenize_spans
from barbet_inference.answer_selection import NativeAnswerRenderer

STATE = release.base.WORK / 'reports/v26-continuation-readiness-v1'
MODEL = Path('/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots') / release.MODEL_REVISION
TOKENIZER = None
NATIVE = None
SPECIAL = None


def init():
    global TOKENIZER, NATIVE, SPECIAL
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZER = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    NATIVE = NativeAnswerRenderer((MODEL / 'chat_template.jinja').read_text(), TOKENIZER)
    SPECIAL = release.read(MODEL / 'config.json')
    pa.set_cpu_count(1)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def job(item):
    key = digest(item['path'])
    report_path = STATE / 'files' / (key + '.json')
    if report_path.exists():
        return release.read(report_path)
    record_path = release.parent.DEST / item['path']
    token_path = Path(str(record_path).replace('/records/', '/tokenized/'))
    counts = Counter(); errors = []; samples = []
    record_meta = pq.read_table(record_path, columns=[
        'record_id', 'family_id', 'query_hash', 'split', 'source', 'domain',
        'memory_placement', 'actual_retrieval', 'recorded_error_followup',
        'assistant_turns', 'pangolin_tokens', 'pangolin_supervised_tokens', 'verification_tier'])
    token_rows = pq.ParquetFile(token_path)
    if len(record_meta) != token_rows.metadata.num_rows:
        raise ValueError('Aligned row counts differ')
    expected_token = next(x for x in release.read(release.STATE / 'publication-plan.json')['manifest']
                          if x['path'] == item['path'].replace('/records/', '/tokenized/'))
    for path, expected in [(record_path, item), (token_path, expected_token)]:
        if release.base.file_sha(path) != expected['sha256']:
            raise ValueError('Current retained payload changed')
    meta = record_meta.to_pydict()
    sample_indices = {}
    bounds = [4096, 8192, 16384, 32768, 65536, 131072]
    for i in range(len(record_meta)):
        rid = meta['record_id'][i]
        bound = next(b for b in bounds if meta['pangolin_tokens'][i] <= b)
        stratum = '|'.join(str(meta[k][i]) for k in ['source', 'domain', 'memory_placement', 'split']) + '|' + str(bound)
        rank = digest('20261028:' + rid)
        if stratum not in sample_indices or rank < sample_indices[stratum][0]:
            sample_indices[stratum] = (rank, i)
        for k in ['source', 'domain', 'memory_placement', 'split', 'verification_tier']:
            counts[k + ':' + str(meta[k][i])] += 1
        counts['actual_retrieval'] += int(meta['actual_retrieval'][i])
        counts['recorded_error_followup'] += int(meta['recorded_error_followup'][i])
    offset = 0
    for batch in token_rows.iter_batches(batch_size=256):
        t = pa.Table.from_batches([batch]).combine_chunks()
        ids_array = t['input_ids'].chunk(0); labels_array = t['labels'].chunk(0)
        offsets = np.asarray(ids_array.offsets)
        ids = np.asarray(ids_array.values); labels = np.asarray(labels_array.values)
        lengths = np.diff(offsets)
        valid = labels != -100
        supervised = np.add.reduceat(valid.astype(np.int32), offsets[:-1])
        bos = np.add.reduceat((ids == SPECIAL['bos_token_id']).astype(np.int32), offsets[:-1])
        eos = np.add.reduceat(((labels == SPECIAL['eos_token_id']) & valid).astype(np.int32), offsets[:-1])
        bad_token = (ids < 0) | (ids >= SPECIAL['vocab_size'])
        bad_label = valid & (labels != ids)
        bad_pad = ids == SPECIAL['pad_token_id']
        positions = np.flatnonzero(valid)
        row_ends = np.searchsorted(positions, offsets[1:], side='left') - 1
        last_supervised = labels[positions[row_ends]]
        row_checks = {
            'length_matches': lengths == np.asarray(t['total_tokens']),
            'supervised_count_matches': supervised == np.asarray(t['supervised_tokens']),
            'nonempty_supervision': supervised > 0,
            'output_budget': lengths + 4096 <= 131072,
            'bos_once': bos == 1,
            'bos_first': ids[offsets[:-1]] == SPECIAL['bos_token_id'],
            'first_label_masked': labels[offsets[:-1]] == -100,
            'final_supervised_is_eos': last_supervised == SPECIAL['eos_token_id'],
            'assistant_eos_count': eos == np.asarray(meta['assistant_turns'][offset:offset + len(t)]),
            'record_length_matches': lengths == np.asarray(meta['pangolin_tokens'][offset:offset + len(t)]),
            'record_supervised_matches': supervised == np.asarray(meta['pangolin_supervised_tokens'][offset:offset + len(t)]),
        }
        if not np.array_equal(offsets, np.asarray(labels_array.offsets)):
            raise ValueError('ID/label offsets differ')
        if t['record_id'].to_pylist() != meta['record_id'][offset:offset + len(t)]:
            raise ValueError('Record/token identity order differs')
        for name, mask in [('out_of_vocab', bad_token), ('label_not_input_id', bad_label),
                           ('literal_pad_in_unpadded_data', bad_pad)]:
            counts['tokens:' + name] += int(mask.sum())
            rows = np.unique(np.searchsorted(offsets[1:], np.flatnonzero(mask), side='right'))
            counts['rows:' + name] += len(rows)
            for row in rows[:8]:
                errors.append({'record_id': meta['record_id'][offset + int(row)], 'check': name})
        for name, checks in row_checks.items():
            counts['failed:' + name] += int((~checks).sum())
            for i in np.flatnonzero(~checks)[:8]:
                errors.append({'record_id': meta['record_id'][offset + int(i)], 'check': name})
        counts['rows'] += len(t); counts['tokens'] += len(ids)
        counts['supervised_tokens'] += int(valid.sum())
        counts['unknown_tokens'] += int((ids == SPECIAL['unk_token_id']).sum())
        offset += len(t)
    # Choose pointers from every source/domain/placement/split/length combination.
    for stratum, (rank, index) in sample_indices.items():
        samples.append({'stratum': stratum, 'rank': rank, 'path': item['path'],
                        'row': index, 'record_id': meta['record_id'][index]})
    result = {'path': item['path'], 'counts': dict(counts), 'errors': errors[:40], 'samples': samples}
    release.base.save(report_path, result)
    return result


def arrays(workers):
    STATE.mkdir(exist_ok=True); (STATE / 'files').mkdir(exist_ok=True)
    if release.shutil.disk_usage(release.base.WORK).free < 100 * 1024**3 + 512 * 1024**2:
        raise ValueError('Read-only audit report reserve unavailable')
    if (STATE / 'array-summary.json').exists():
        raise ValueError('First full array audit already complete')
    proof = release.read(release.STATE / 'publication-final-commit.json')
    plan = release.read(release.STATE / 'publication-plan.json')
    inputs = [x for x in plan['manifest'] if '/records/' in x['path']]
    policy = {'dataset_revision': proof['final_head'], 'driver_sha256': release.base.file_sha(Path(__file__)),
              'current_plan_sha256': release.base.file_sha(release.STATE / 'publication-plan.json'),
              'tokenizer_sha256': release.base.file_sha(MODEL / 'tokenizer.json'),
              'native_template_sha256': release.base.file_sha(MODEL / 'chat_template.jinja'),
              'full_record_and_token_file_hashes': True, 'full_array_values_checked': True,
              'source_content_semantics_checked_by_sampling_only': True, 'sample_seed': '20261028',
              'cpu_workers': workers, 'no_model_training': True, 'no_source_program_execution': True,
              'first_holds_and_saved_scores_unchanged': True}
    pp = STATE / 'policy.json'
    if pp.exists() and release.read(pp) != policy:
        raise ValueError('Frozen audit policy changed')
    if not pp.exists(): release.base.save(pp, policy)
    counts = Counter(); candidates = {}; errors = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        tasks = [pool.submit(job, item) for item in inputs]
        for n, future in enumerate(as_completed(tasks), 1):
            part = future.result(); counts.update(part['counts']); errors.extend(part['errors'])
            for pointer in part['samples']:
                key = pointer['stratum']
                if key not in candidates or pointer['rank'] < candidates[key]['rank']:
                    candidates[key] = pointer
            if n % 25 == 0 or n == len(tasks):
                print(json.dumps({'files': n, 'total': len(tasks), 'rows': counts['rows'],
                                  'hard_check_failures': sum(v for k, v in counts.items() if k.startswith('failed:'))}), flush=True)
    release.base.save(STATE / 'array-summary.json', {'counts': dict(counts), 'errors': errors,
                                                   'files_checked': len(inputs), 'all_values_read': True})
    release.base.save(STATE / 'sample-pointers.json', sorted(candidates.values(), key=lambda x: x['stratum']))
    print(json.dumps({'rows': counts['rows'], 'stratified_samples': len(candidates), 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16: raise ValueError('Worker resource bound')
    arrays(args.workers)
