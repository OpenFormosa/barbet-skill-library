"""Join every scored row to public provenance and reject missing/duplicate rows."""
from collections import defaultdict
from pathlib import Path
import json
import math
import shutil
import sqlite3
import time
import pyarrow as pa
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, save_new, sha_file, digest
from barbet_pipeline.gpu_release import FAMILIES, ScoreTotals, validate_score_receipt, measured_curriculum_policy

STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
GPU = STATE / 'gpu-quality'


def merge_totals(parts):
    """Merge exact numerators/counts; never average file/rank means."""
    result = {'rows': 0, 'input_tokens': 0, 'supervised_tokens': 0, 'teacher_forced_top1_correct': 0,
              'nll_sum': 0., 'length_bucket_upper_bounds': {}, 'nll_histogram_upper_bounds': {}}
    example_sum = norm_sum = 0.
    cosine_sums = {family: 0. for family in FAMILIES}
    minima, maxima = [], []
    for p in parts:
        for key in ['rows', 'input_tokens', 'supervised_tokens', 'teacher_forced_top1_correct', 'nll_sum']:
            result[key] += p[key]
        for key in ['length_bucket_upper_bounds', 'nll_histogram_upper_bounds']:
            for group, count in p[key].items():
                result[key][group] = result[key].get(group, 0) + count
        example_sum += p['example_mean_nll'] * p['rows']
        norm_sum += p['mean_readout_gradient_sketch_norm'] * p['rows']
        for family in FAMILIES:
            cosine_sums[family] += p['mean_semantic_family_cosines'][family] * p['rows']
        minima.append(p['nll_min']); maxima.append(p['nll_max'])
    assert result['rows'] and result['supervised_tokens']
    result.update(token_weighted_nll=result['nll_sum'] / result['supervised_tokens'],
        example_mean_nll=example_sum / result['rows'],
        teacher_forced_top1_fraction=result['teacher_forced_top1_correct'] / result['supervised_tokens'],
        mean_readout_gradient_sketch_norm=norm_sum / result['rows'],
        mean_semantic_family_cosines={f: v / result['rows'] for f, v in cosine_sums.items()},
        nll_min=min(minima), nll_max=max(maxima))
    return result


def main():
    helper_sha = sha_file('barbet_pipeline/gpu_release.py')
    while not (GPU / 'provenance-index.json').exists() or not (GPU / 'full-score-plan.json').exists():
        time.sleep(5)
    plan = read_json(GPU / 'full-score-plan.json')
    index = read_json(GPU / 'provenance-index.json')
    assert sha_file(GPU / 'provenance.sqlite3') == index['database_sha256']
    assert sha_file('barbet_pipeline/gpu_quality.py') == plan['scorer_sha256']
    assert sha_file('scripts/run_full_v4_gpu_quality.py') == plan['runner_sha256']
    inputs = {digest(item['remote']): item for assignment in plan['assignments'] for item in assignment}
    assert len(inputs) == plan['files']
    db = sqlite3.connect(GPU / 'aggregate-progress.sqlite3', uri=True)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('ATTACH DATABASE ? AS provenance', ('file:' + str(GPU / 'provenance.sqlite3') + '?mode=ro',))
    db.execute('CREATE TABLE IF NOT EXISTS seen (record_id BLOB PRIMARY KEY) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS completed (key TEXT PRIMARY KEY, receipt TEXT NOT NULL) WITHOUT ROWID')
    parts = {key: json.loads(receipt) for key, receipt in db.execute('SELECT key,receipt FROM completed')}
    for key, part in parts.items():
        assert key in inputs and part['input_sha256'] == inputs[key]['sha256']
        assert part['aggregator_sha256'] == sha_file(__file__)
        assert part['aggregation_helper_sha256'] == helper_sha
        assert sha_file(part['file']['local']) == part['file']['sha256']
        target = GPU / 'aggregation-receipts' / (key + '.json')
        if not target.exists():
            save_new(target, part)
    while len(parts) < len(inputs):
        ready = [key for key in sorted(set(inputs) - set(parts)) if (GPU / 'score-receipts' / (key + '.json')).exists()]
        if not ready:
            if (GPU / 'workers-exited.json').exists() and not read_json(GPU / 'workers-exited.json')['all_success']:
                raise RuntimeError('GPU worker failed; successful measurements and joins remain resumable')
            time.sleep(5)
            continue
        for key in ready:
            if shutil.disk_usage(GPU).free < 50 * 1024**3:
                raise RuntimeError('GPU aggregation reached the 50 GiB free-space floor')
            item = inputs[key]
            receipt = read_json(GPU / 'score-receipts' / (key + '.json'))
            raw_path = validate_score_receipt(item, receipt, plan, GPU)
            status = item['remote'].split('/')[1]
            split = Path(item['remote']).name.split('-', 1)[0]
            output = GPU / 'measurements' / status / (split + '-' + key + '.parquet')
            output.parent.mkdir(exist_ok=True, parents=True)
            if output.exists():
                archive = GPU / 'incomplete-aggregation-attempts'
                archive.mkdir(exist_ok=True)
                output.rename(archive / (key + '-' + str(time.time_ns()) + '.parquet'))
            total = ScoreTotals()
            by_source = defaultdict(ScoreTotals)
            writer = None
            try:
                with db:
                    for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=512):
                        rows = batch.to_pylist()
                        ids = [bytes.fromhex(r['record_id']) for r in rows]
                        marks = ','.join('?' for _ in ids)
                        provenance = {r[0]: r[1:] for r in db.execute(
                            'SELECT record_id,parent_record_id,source,split,status,root_task_id,query_hash,verification_scope FROM provenance.records WHERE record_id IN (' + marks + ')', ids)}
                        if len(provenance) != len(rows):
                            raise ValueError('Score IDs are duplicated or absent from the actual exported parents')
                        values = []
                        for row, ident in zip(rows, ids):
                            parent, source, actual_split, actual_status, root, query, scope = provenance[ident]
                            if actual_status != status or actual_split != split:
                                raise ValueError('Scored row moved between source quality status or split')
                            total.add(row)
                            by_source['|'.join([status, source, split])].add(row)
                            values.append({**row, 'parent_record_id': parent, 'source': source, 'split': split,
                                'processing_status': status, 'root_task_id': root, 'query_hash': query,
                                'verification_scope': scope, 'model_revision': receipt['model_revision'],
                                'automatic_training_admission': False})
                        db.executemany('INSERT INTO seen VALUES (?)', [(i,) for i in ids])
                        table = pa.Table.from_pylist(values)
                        if writer is None:
                            writer = pq.ParquetWriter(output, table.schema, compression='zstd')
                        writer.write_table(table)
                    if writer is not None:
                        writer.close(); writer = None
                    totals = total.result()
                    for field in ['rows', 'input_tokens', 'supervised_tokens', 'teacher_forced_top1_correct']:
                        assert totals[field] == receipt[field]
                    assert math.isclose(totals['nll_sum'], receipt['nll_sum'], rel_tol=1e-10, abs_tol=1e-6)
                    assert pq.ParquetFile(output).metadata.num_rows == item['rows']
                    file = {'local': str(output), 'remote': 'full-v4/gpu-quality/measurements/' + status + '/' + output.name,
                        'rows': totals['rows'], 'bytes': output.stat().st_size, 'sha256': sha_file(output)}
                    part = {'input_path': item['remote'], 'input_sha256': item['sha256'],
                        'raw_score_receipt_sha256': sha_file(GPU / 'score-receipts' / (key + '.json')),
                        'totals': totals, 'source_groups': {name: v.result() for name, v in by_source.items()},
                        'file': file, 'aggregator_sha256': sha_file(__file__), 'aggregation_helper_sha256': helper_sha}
                    db.execute('INSERT INTO completed VALUES (?,?)', (key, json.dumps(part)))
                save_new(GPU / 'aggregation-receipts' / (key + '.json'), part)
                parts[key] = part
            finally:
                if writer is not None:
                    writer.close()
            print(json.dumps({'joined_files': len(parts), 'total_files': len(inputs),
                'joined_rows': sum(p['totals']['rows'] for p in parts.values())}), flush=True)
    while not (GPU / 'workers-exited.json').exists():
        time.sleep(5)
    exited = read_json(GPU / 'workers-exited.json')
    assert exited['all_success'] and set(exited['exit_codes']) == set(map(str, range(8)))
    for rank in range(8):
        worker = read_json(GPU / 'workers' / (str(rank) + '.json'))
        assert worker['complete'] and worker['plan_sha256'] == sha_file(GPU / 'full-score-plan.json')
        assert worker['rows'] == sum(i['rows'] for i in plan['assignments'][rank])
    assert db.execute('SELECT COUNT(*) FROM seen').fetchone()[0] == plan['rows']
    assert db.execute('SELECT COUNT(*) FROM provenance.records p LEFT JOIN seen s ON s.record_id=p.record_id WHERE s.record_id IS NULL').fetchone()[0] == 0
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    totals = merge_totals([p['totals'] for p in parts.values()])
    for field in ['rows', 'input_tokens', 'supervised_tokens']:
        assert totals[field] == plan[field]
    groups = defaultdict(list)
    for part in parts.values():
        for name, values in part['source_groups'].items():
            groups[name].append(values)
    save_new(GPU / 'measurement-files.json', {'files': [parts[k]['file'] for k in sorted(parts)]})
    save_new(GPU / 'curriculum-policy.json', measured_curriculum_policy(read_json(GPU / 'context-probe-summary.json')))
    save_new(GPU / 'complete-summary.json', {'complete': True, 'plan_sha256': sha_file(GPU / 'full-score-plan.json'),
        'files': len(parts), 'totals': totals, 'source_groups': {k: merge_totals(v) for k, v in sorted(groups.items())},
        'all_ids_match_public_export_exactly_once': True, 'no_truncation': True, 'training_updates': 0,
        'weight_sha256': plan['model_files']['model.safetensors'],
        'measurement_manifest_sha256': sha_file(GPU / 'measurement-files.json'),
        'aggregator_sha256': sha_file(__file__), 'aggregation_helper_sha256': helper_sha,
        'scope': 'Exhaustive frozen-v26 likelihood and readout-gradient proxy measurements, not answer truth or demonstrated learnability.',
        'source_quality_statuses_changed_by_gpu': 0, 'targets_changed_by_gpu': 0,
        'free_generation_task_success': 'NOT_MEASURED', 'training_transfer_retention': 'NOT_RUN'})
    print(json.dumps({'complete': True, 'files': len(parts), 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
