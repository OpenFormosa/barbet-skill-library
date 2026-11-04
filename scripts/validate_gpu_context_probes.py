"""Independently bind every scored candidate to its actual public source prefix."""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import json
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, save_new, sha_file
from barbet_pipeline.export import load_tokenizer
from barbet_inference.sft import pangolin_render, tokenize_spans

STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4/gpu-quality')


def main():
    plan = read_json(STATE / 'context-probe-plan.json')
    scores = read_json(STATE / 'context-probe-scores.json')
    assert sha_file(STATE / 'context-probes.parquet') == plan['file_sha256'] == scores['input_sha256']
    source = {}
    for item in plan['sources']:
        assert sha_file(item['file']) == item['sha256']
        for row in pq.read_table(item['file']).to_pylist():
            assert row['record_id'] not in source
            source[row['record_id']] = row
    roots = defaultdict(dict)
    for row in source.values():
        roots[row['root_task_id']][row['variant']] = row
    token_path = '/root/.cache/huggingface/hub/models--voidful--barbet-skill/snapshots/67131067ff1825179d5e38e91d499159669e9c66/tokenizer.json'
    tokenizer = load_tokenizer(token_path)
    score_index = {row['record_id']: row for row in scores['rows']}
    assert len(score_index) == len(scores['rows'])
    checked = Counter()
    for probe in pq.read_table(STATE / 'context-probes.parquet').to_pylist():
        row = source[probe['source_record_id']]
        assert row['split'] == 'train' and row['root_task_id'] == probe['root_task_id']
        assert row['semantic_family_id'] == probe['semantic_family_id']
        scored = score_index.pop(probe['record_id'])
        for key in ['root_task_id', 'semantic_family_id', 'condition', 'correct_candidate', 'source_record_id']:
            assert scored[key] == probe[key]
        assert scored['input_tokens'] == len(probe['input_ids'])
        assert scored['supervised_tokens'] == sum(t != -100 for t in probe['labels'])
        if probe['condition'] in ['original', 'counterfactual', 'invariance']:
            messages = json.loads(row['messages_json'])
            target = messages[2]
            if not probe['correct_candidate']:
                other = 'counterfactual' if probe['condition'] == 'original' else 'original'
                target = json.loads(roots[row['root_task_id']][other]['messages_json'])[2]
            text, spans = pangolin_render(messages[:2] + [target], json.loads(row['tools_json']))
            ids, labels = tokenize_spans(tokenizer, text, spans)
            assert ids == probe['input_ids'] and labels == probe['labels']
            checked['exact_public_prefix_and_target_reconstructed'] += 1
        checked['scored_candidates_bound_to_inputs'] += 1
    assert not score_index
    examples = []
    counts = Counter()
    for root_id, views in roots.items():
        family = views['original']['semantic_family_id']
        if family not in ['comparison', 'ordering', 'rule_version']:
            continue
        a, b = [json.loads(views[v]['messages_json'])[2] for v in ['original', 'counterfactual']]
        a, b = copy.deepcopy(a), copy.deepcopy(b)
        op_a = a['tool_calls'][0]['function']['arguments'].pop('operation')
        op_b = b['tool_calls'][0]['function']['arguments'].pop('operation')
        assert a == b and op_a != op_b
        counts[family] += 1
        if not any(e['family'] == family for e in examples):
            values = [r for r in scores['rows'] if r['root_task_id'] == root_id
                      and r['condition'] in ['original', 'counterfactual']]
            examples.append({'family': family, 'root_task_id': root_id,
                'original_task': json.loads(views['original']['public_episode_json'])['task'],
                'counterfactual_task': json.loads(views['counterfactual']['public_episode_json'])['task'],
                'original_operation': op_a, 'counterfactual_operation': op_b,
                'candidate_scores': [{k: r[k] for k in ['condition', 'correct_candidate', 'nll', 'nll_sum', 'supervised_tokens']}
                                     for r in values]})
    result = {'passed': True, 'all_scored_candidates': plan['rows'], 'checks': dict(checked),
        'operation_is_only_target_difference': dict(counts), 'examples': examples,
        'score_sha256': sha_file(STATE / 'context-probe-scores.json'),
        'probe_sha256': plan['file_sha256'], 'validator_sha256': sha_file(__file__),
        'semantic_source_correctness': 'Inherited v3 executable task/oracle checks; this validation independently verifies candidate construction.',
        'accuracy_scope': 'Candidate preference on TRAIN roots; not free-generation or unseen-task accuracy.'}
    save_new(STATE / 'context-probe-controls.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}), flush=True)


if __name__ == '__main__':
    main()
