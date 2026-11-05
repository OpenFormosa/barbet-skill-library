"""Real-v26 control for the copied original scoring loop before a switch."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
from pathlib import Path
import torch
sys.path.append('/root/barbet-context-sft-data/.venv/lib/python3.12/site-packages')
import pyarrow as pa
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, save_new, sha_file, digest
from barbet_pipeline.gpu_quality import FrozenScorer
from barbet_pipeline.parallel_recovery import writer_core
from scripts import run_full_v4_gpu_parallel_recovery as recovery


def main():
    torch.set_num_threads(2)
    root = recovery.GPU
    output = root / 'parallel-recovery-controls-20261105'
    output.mkdir(exist_ok=False)
    code = recovery.CODE
    assert writer_core(code / 'scripts/run_full_v4_gpu_quality.py') == writer_core(Path(recovery.__file__))
    original_plan = read_json(root / 'full-score-plan.json')
    assert sha_file(code / 'barbet_pipeline/gpu_quality.py') == original_plan['scorer_sha256']
    for name, expected in original_plan['model_files'].items():
        assert sha_file(recovery.MODEL / name) == expected
    source = root / 'batching-control-input.parquet'
    profile = read_json(root / 'batching-profile.json')
    assert sha_file(source) == profile['source_input_sha256']
    rows = next(pq.ParquetFile(source).iter_batches(batch_size=32)).to_pylist()
    assert len(rows) == 32
    scratch = output / 'scratch'
    scratch.mkdir()
    (output / 'scores').mkdir()
    item = {'remote': 'control-fixtures/parallel-recovery-32.parquet', 'rows': 32,
        'revision': recovery.MODEL.name}
    local = scratch / (digest(item['remote']) + '.parquet')
    pq.write_table(pa.Table.from_pylist(rows), local, compression='zstd')
    item.update(sha256=sha_file(local), bytes=local.stat().st_size)
    save_new(output / 'plan.json', {'scope': '32 real input rows; numerical control only',
        'item': item, 'source_input_sha256': sha_file(source)})
    sys.path.insert(0, str(recovery.MODEL))
    from load_barbet import load_barbet
    model, tokenizer = load_barbet(recovery.MODEL, 'cuda:0')
    scorer = FrozenScorer(model)
    reference = {row['record_id']: scorer.score([row])[0] for row in rows}
    features = {}

    class Recorder:
        def score(self, batch):
            values = scorer.score(batch)
            features.update({v['record_id']: v['_feature'].clone() for v in values})
            return values

    anchor = read_json(root / 'anchors.json')
    families = sorted(anchor['prototypes'])
    prototypes = torch.tensor([anchor['prototypes'][f] for f in families])
    schema = pa.schema([('record_id',pa.string()),('input_tokens',pa.int64()),('supervised_tokens',pa.int64()),('nll_sum',pa.float64()),('nll',pa.float64()),('teacher_forced_top1_correct',pa.int64()),('readout_gradient_sketch_norm',pa.float64()),('semantic_family_cosines',pa.list_(pa.float32(),len(families)))])
    # All products stay under the explicit numerical-control directory.
    recovery.GPU = output
    recovery.RECOVERY = output
    count = recovery.process_file(item, 1, Recorder(), prototypes, families, schema,
        pq, pa, original_plan, scratch, output / 'scores')
    actual = pq.read_table(output / 'scores' / (digest(item['remote']) + '.parquet')).to_pylist()
    tolerance = read_json(root / 'control-tolerances.json')
    nll_errors, cosines = [], []
    for value in actual:
        ident = value['record_id']; expected = reference[ident]
        error = abs(value['nll'] - expected['nll'])
        cosine = float(features[ident] @ expected['_feature'])
        assert error <= tolerance['nll_absolute'] + tolerance['nll_relative'] * abs(expected['nll'])
        assert cosine >= tolerance['sketch_cosine_minimum']
        assert value['teacher_forced_top1_correct'] == expected['teacher_forced_top1_correct']
        nll_errors.append(error); cosines.append(cosine)
    assert count == len(actual) == len(reference) == 32
    result = {'passed': True, 'rows': count, 'original_scoring_core_ast_identical': True,
        'original_plan_sha256': sha_file(root / 'full-score-plan.json'),
        'original_runner_sha256': sha_file(code / 'scripts/run_full_v4_gpu_quality.py'),
        'recovery_runner_sha256': sha_file(recovery.__file__),
        'scorer_sha256': original_plan['scorer_sha256'],
        'weight_sha256': original_plan['model_files']['model.safetensors'],
        'source_input_sha256': sha_file(source), 'control_input_sha256': item['sha256'],
        'maximum_nll_absolute_error': max(nll_errors), 'minimum_readout_feature_cosine': min(cosines),
        'teacher_forced_top1_counts_identical': True, 'preexisting_tolerances': tolerance,
        'control_implementation_sha256': sha_file(__file__), 'training_updates': 0,
        'scope': 'Real frozen-v26 numerical equivalence on 32 rows; not capability or training validation.'}
    save_new(output / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
