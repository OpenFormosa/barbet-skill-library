"""Read-only diagnostics of complete IFM alternative scores; no model requests."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import statistics

from scripts import build_memory_sft_v2 as base
from scripts.normalize_ifm_math_memory_v1 import STATE as SOURCE

STATE = SOURCE / 'v26-selection-v1'


def quantiles(values):
    values = sorted(values)
    return {'minimum': values[0], 'p25': values[int((len(values) - 1) * .25)],
            'median': statistics.median(values), 'p75': values[int((len(values) - 1) * .75)],
            'p95': values[int((len(values) - 1) * .95)], 'maximum': values[-1]}


def main():
    output = STATE / 'answer-selection-diagnostics-v1.json'
    if output.exists(): raise ValueError('score diagnostic already frozen')
    counts = Counter(); margins = []; chosen_tokens = []; alternative_tokens = []; chosen_ppl = []; ratios = []
    bindings = []
    for bucket in range(8):
        folder = STATE / 'scores/canonical-v2'; receipt = json.loads((folder / f'bucket-{bucket}.summary.json').read_text())
        if receipt['status'] != 'BUCKET_COMPLETE' or receipt['limited_pilot']: raise ValueError('full bucket required')
        path = folder / f'bucket-{bucket}.sqlite'; bindings.append({'bucket': bucket, 'sha256': base.file_sha(path)})
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        groups = {}; decisions = {}
        for group, status, winner in db.execute('SELECT input_hash,status,winner FROM decisions'):
            decisions[group] = (status, winner)
        for group, raw in db.execute('SELECT input_hash,score_json FROM scores'):
            groups.setdefault(group, []).append(json.loads(raw))
        db.close()
        for group, (status, winner) in decisions.items():
            counts['groups:' + status] += 1
            if status != 'selected': continue
            rows = sorted(groups[group], key=lambda r: (r['mean_nll'], r['record_id']))
            if rows[0]['candidate_identity'] != winner: raise ValueError('decision differs from actual minimum')
            counts['alternatives_scored'] += len(rows)
            best = rows[0]; lengths = [r['answer_tokens'] for r in rows]
            margins.append(rows[1]['mean_nll'] - best['mean_nll'])
            chosen_tokens.append(best['answer_tokens']); alternative_tokens.extend(lengths); chosen_ppl.append(best['ppl'])
            ratios.append(best['answer_tokens'] / statistics.mean(lengths))
            counts['winner_is_a_shortest_answer'] += best['answer_tokens'] == min(lengths)
            counts['winner_is_a_longest_answer'] += best['answer_tokens'] == max(lengths)
            counts['all_answer_lengths_equal'] += min(lengths) == max(lengths)
            counts['exact_nll_tie'] += rows[1]['mean_nll'] == best['mean_nll']
            counts['nll_margin_below_1e-4'] += margins[-1] < 1e-4
    result = {'counts': dict(counts), 'runner_up_minus_winner_mean_nll': quantiles(margins),
        'winner_answer_tokens': quantiles(chosen_tokens), 'all_alternative_answer_tokens': quantiles(alternative_tokens),
        'winner_ppl': quantiles(chosen_ppl), 'winner_tokens_over_group_mean_tokens': quantiles(ratios),
        'average_winner_answer_tokens': statistics.mean(chosen_tokens),
        'average_alternative_answer_tokens': statistics.mean(alternative_tokens),
        'scoring_database_bindings': bindings, 'driver_sha256': base.file_sha(Path(__file__)),
        'standard_answers_changed': False, 'model_requests': 0,
        'scope': 'Only the 24,933 competing math groups; excludes singletons and does not assess correctness, training gains or model capability.',
        'length_is_not_the_ranking_criterion': True, 'lowest_ppl_is_not_correctness_proof': True}
    base.save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scoring_database_bindings'}), flush=True)


if __name__ == '__main__': main()
