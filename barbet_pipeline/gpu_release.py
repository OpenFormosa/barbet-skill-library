"""Exact accounting and honest interpretation of frozen-student measurements."""
from collections import Counter
from pathlib import Path
import math
from .common import read_json, sha_file, digest

FAMILIES = ['binding', 'comparison', 'evidence_state', 'operation_state', 'ordering', 'rule_version', 'units']
LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
NLL_BINS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]


def validate_measurement(row):
    if len(bytes.fromhex(row['record_id'])) != 32:
        raise ValueError('Invalid record identity')
    n, supervised = row['input_tokens'], row['supervised_tokens']
    if not 0 < supervised < n <= 131072:
        raise ValueError('Invalid complete-sequence or assistant-token counts')
    if not 0 <= row['teacher_forced_top1_correct'] <= supervised:
        raise ValueError('Teacher-forced correct count exceeds the actual targets')
    for name in ['nll_sum', 'nll', 'readout_gradient_sketch_norm']:
        if not math.isfinite(row[name]) or row[name] < 0:
            raise ValueError('Invalid measurement: ' + name)
    if not math.isclose(row['nll'] * supervised, row['nll_sum'], rel_tol=2e-6, abs_tol=1e-5):
        raise ValueError('Mean target NLL differs from the measured token sum')
    if len(row['semantic_family_cosines']) != len(FAMILIES):
        raise ValueError('Semantic prototype dimension differs')
    if any(not math.isfinite(v) or abs(v) > 1.0001 for v in row['semantic_family_cosines']):
        raise ValueError('Invalid semantic-prototype cosine')


class ScoreTotals:
    def __init__(self):
        self.rows = self.input_tokens = self.supervised_tokens = self.correct = 0
        self.nll_sum = self.example_nll_sum = self.norm_sum = 0.0
        self.cosine_sums = [0.0] * len(FAMILIES)
        self.lengths, self.histogram = Counter(), Counter()
        self.nll_min, self.nll_max = None, None

    def add(self, row):
        validate_measurement(row)
        self.rows += 1
        self.input_tokens += row['input_tokens']
        self.supervised_tokens += row['supervised_tokens']
        self.correct += row['teacher_forced_top1_correct']
        self.nll_sum += row['nll_sum']
        self.example_nll_sum += row['nll']
        self.norm_sum += row['readout_gradient_sketch_norm']
        self.cosine_sums = [a + b for a, b in zip(self.cosine_sums, row['semantic_family_cosines'])]
        self.lengths[str(next(n for n in LENGTHS if row['input_tokens'] <= n))] += 1
        self.histogram[next((str(n) for n in NLL_BINS if row['nll'] <= n), '>32')] += 1
        self.nll_min = row['nll'] if self.nll_min is None else min(self.nll_min, row['nll'])
        self.nll_max = row['nll'] if self.nll_max is None else max(self.nll_max, row['nll'])

    def result(self):
        return {'rows': self.rows, 'input_tokens': self.input_tokens, 'supervised_tokens': self.supervised_tokens,
            'nll_sum': self.nll_sum, 'token_weighted_nll': self.nll_sum / self.supervised_tokens if self.supervised_tokens else None,
            'example_mean_nll': self.example_nll_sum / self.rows if self.rows else None,
            'teacher_forced_top1_correct': self.correct,
            'teacher_forced_top1_fraction': self.correct / self.supervised_tokens if self.supervised_tokens else None,
            'mean_readout_gradient_sketch_norm': self.norm_sum / self.rows if self.rows else None,
            'mean_semantic_family_cosines': dict(zip(FAMILIES, [v / self.rows if self.rows else None for v in self.cosine_sums])),
            'length_bucket_upper_bounds': dict(self.lengths), 'nll_histogram_upper_bounds': dict(self.histogram),
            'nll_min': self.nll_min, 'nll_max': self.nll_max}


def validate_score_receipt(item, receipt, plan, gpu):
    expected = {'input_path': item['remote'], 'input_sha256': item['sha256'],
        'input_revision': item['revision'], 'rows': item['rows'], 'scorer_sha256': plan['scorer_sha256'],
        'anchor_sha256': plan['anchor_sha256'], 'training_updates': 0, 'no_truncation': True,
        'semantic_family_order': FAMILIES}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError('GPU receipt identity mismatch: ' + key)
    path = Path(receipt['output_path'])
    if path != Path(gpu) / 'scores' / (digest(item['remote']) + '.parquet'):
        raise ValueError('Unexpected GPU output path')
    if path.stat().st_size != receipt['output_bytes'] or sha_file(path) != receipt['output_sha256']:
        raise ValueError('GPU score bytes differ from their receipt')
    return path


def measured_curriculum_policy(probe_summary):
    families = []
    for row in probe_summary['families']:
        pairs = row['discriminating_first_action_pairs']
        families.append({'family': row['family'], 'train_roots_probed': row['roots'],
            'discriminating_first_decision_pairs': pairs,
            'pairs_correct_sum_logprob': row['sum_logprob_pairs_correct'],
            'pairs_correct_mean_logprob': row['mean_logprob_pairs_correct'],
            'candidate_pair_failure_fraction': (pairs - row['sum_logprob_pairs_correct']) / pairs if pairs else None,
            'diagnostic_only': True,
            'policy': 'Keep both semantic conditions together; never keep only the lower-NLL side.' if pairs else
                      'First action is identical; assess later observed-state decisions before assigning weakness.'})
    return {'status': 'MEASURED_STUDENT_DIFFICULTY_NOT_TRAINING_VALIDATED',
        'model': 'voidful/barbet-skill@67131067ff1825179d5e38e91d499159669e9c66',
        'answer_correctness_gate_precedes_student_scores': True,
        'reference_only_automatic_admission': False,
        'low_nll_is_not_semantic_mastery': True,
        'high_nll_is_not_automatic_rejection': True,
        'value_aligned_records_are_not_removed_for_student_disagreement': True,
        'do_not_sort_the_complete_corpus_by_nll_or_prototype_cosine': True,
        'same_input_candidate_selection': 'Preserve the original frozen-v26 winner and complete input; new scores do not reselect it.',
        'counterfactual_contract': 'Different input conditions are separate correct targets, not competing answer versions for one input.',
        'semantic_family_policies': families,
        'proposed_exposure': {'unit': 'supervised tokens and unique roots', 'general_fraction': 0.5,
            'semantic_fraction': 0.5, 'validated': False, 'graph_dominance_cap_required': True,
            'reason': 'The full corpus remains graph-heavy; semantic anchors are not a broad general-capability retention set.'},
        'gradient_scope': 'Detached readout-copy sketch; not the full tied-embedding or full-model training gradient.',
        'gradient_anchors': 'All 2000 authored TRAIN roots, seven semantic families; no DEV/calibration/TEST use.',
        'unknowns': ['actual learning gain', 'unseen-schema/composition transfer', 'same-checkpoint capability retention',
                     'free-generation task success', '1M-context capability'],
        'training_updates': 0}


def collect_complete_gpu_release(state, data):
    """No partial-row, partial-file or unmatched-checkpoint completion claim."""
    state = Path(state)
    gpu = state / 'gpu-quality'
    plan = read_json(gpu / 'full-score-plan.json')
    summary = read_json(gpu / 'complete-summary.json')
    processing = read_json(state / 'processing-summary.json')
    expected = {f['remote']: f for f in data if '/tokenized/' in f['remote']}
    planned = [f for a in plan['assignments'] for f in a]
    if len(planned) != len(expected) or {f['remote'] for f in planned} != set(expected):
        raise ValueError('GPU plan does not cover every exported input shard exactly once')
    for f in planned:
        if any(f[k] != expected[f['remote']][k] for k in ['sha256', 'bytes', 'rows']):
            raise ValueError('GPU input identity differs from the complete CPU export')
    if plan['source_processing_summary_sha256'] != sha_file(state / 'processing-summary.json'):
        raise ValueError('GPU plan processing parent changed')
    if summary['plan_sha256'] != sha_file(gpu / 'full-score-plan.json') or not summary['complete']:
        raise ValueError('GPU completion is not bound to the actual full plan')
    if (summary['files'] != len(expected) or summary['weight_sha256'] != plan['model_files']['model.safetensors']
            or summary['training_updates'] != 0 or not summary['no_truncation']
            or not summary['all_ids_match_public_export_exactly_once']):
        raise ValueError('GPU completion scope differs from the frozen measurement contract')
    for field in ['rows', 'input_tokens', 'supervised_tokens']:
        wanted = processing['counts']['exported' if field == 'rows' else field]
        if summary['totals'][field] != wanted or plan[field] != wanted:
            raise ValueError('GPU whole-corpus coverage mismatch: ' + field)
    controls = read_json(gpu / 'numerical-controls.json')
    if not controls['passed'] or controls['scorer_sha256'] != plan['scorer_sha256']:
        raise ValueError('Frozen scorer lacks matching numerical controls')
    if controls['weight_sha256'] != plan['model_files']['model.safetensors']:
        raise ValueError('Controls use different model weights')
    probe_controls = read_json(gpu / 'context-probe-controls.json')
    if not probe_controls['passed'] or probe_controls['score_sha256'] != sha_file(gpu / 'context-probe-scores.json'):
        raise ValueError('Semantic probe candidate construction was not verified')
    index = read_json(gpu / 'provenance-index.json')
    if index['rows'] != plan['rows'] or index['processing_sha256'] != plan['source_processing_summary_sha256']:
        raise ValueError('GPU provenance coverage differs')
    if index['database_sha256'] != sha_file(gpu / 'provenance.sqlite3'):
        raise ValueError('Original GPU provenance index changed')
    exited = read_json(gpu / 'workers-exited.json')
    if exited['exit_codes'] != {str(i): 0 for i in range(8)} or not exited['all_success']:
        raise ValueError('Not all eight frozen-scoring workers completed successfully')
    for rank in range(8):
        worker = read_json(gpu / 'workers' / (str(rank) + '.json'))
        if (not worker['complete'] or worker['plan_sha256'] != summary['plan_sha256']
                or worker['rows'] != sum(f['rows'] for f in plan['assignments'][rank])):
            raise ValueError('GPU worker completion differs from its exact assignment')
    if summary['measurement_manifest_sha256'] != sha_file(gpu / 'measurement-files.json'):
        raise ValueError('Enriched measurement manifest changed')
    items = read_json(gpu / 'measurement-files.json')['files']
    if sum(i['rows'] for i in items) != plan['rows'] or len(items) != len(expected):
        raise ValueError('Enriched score publication misses complete input files')
    expected_measurements = {}
    for f in planned:
        status, split = f['remote'].split('/')[1], Path(f['remote']).name.split('-', 1)[0]
        remote = 'full-v4/gpu-quality/measurements/' + status + '/' + split + '-' + digest(f['remote']) + '.parquet'
        expected_measurements[remote] = f
    if len({i['remote'] for i in items}) != len(items) or {i['remote'] for i in items} != set(expected_measurements):
        raise ValueError('GPU measurement paths are duplicated or incomplete')
    for item in items:
        source = expected_measurements[item['remote']]
        if item['rows'] != source['rows']:
            raise ValueError('GPU measurement row count differs from its actual input file')
        key = digest(source['remote'])
        receipt = read_json(gpu / 'score-receipts' / (key + '.json'))
        validate_score_receipt(source, receipt, plan, gpu)
        joined = read_json(gpu / 'aggregation-receipts' / (key + '.json'))
        if (joined['file'] != item or joined['input_sha256'] != source['sha256']
                or joined['raw_score_receipt_sha256'] != sha_file(gpu / 'score-receipts' / (key + '.json'))):
            raise ValueError('GPU enriched data is not bound to its actual source and score receipt')
        if sha_file(item['local']) != item['sha256'] or Path(item['local']).stat().st_size != item['bytes']:
            raise ValueError('Enriched score publication changed')
    return summary, items
