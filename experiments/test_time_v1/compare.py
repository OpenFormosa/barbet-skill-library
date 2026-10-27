#!/usr/bin/env python3
"""Paired, question-level uncertainty, with repeated seeds averaged first."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY = ['drcd', 'gsm8k', 'mbpp', 'tools']


def by_question(report):
    rows = defaultdict(list)
    for row in report['scores']:
        rows[row['id']].append(row)
    return rows


def paired(a, b, repetitions=10000):
    """B minus A in percentage points. No bootstrap across correlated seeds."""
    qa, qb = by_question(a), by_question(b)
    if qa.keys() != qb.keys():
        raise ValueError('comparison requires the same question set')
    domains = defaultdict(list)
    for ident in sorted(qa):
        ar, br = qa[ident], qb[ident]
        if {r['seed'] for r in ar} != {r['seed'] for r in br}:
            raise ValueError('comparison requires the same seed set')
        if len({r['seed'] for r in ar}) != len(ar) or len({r['seed'] for r in br}) != len(br):
            raise ValueError('duplicate seed/question')
        if ar[0]['domain'] != br[0]['domain'] or ar[0]['group'] != br[0]['group']:
            raise ValueError('mismatched domain/group')
        delta = np.mean([r['correct'] for r in br]) - np.mean([r['correct'] for r in ar])
        domains[ar[0]['domain']].append((float(delta), ar[0]['group']))
    rng = np.random.default_rng(20261027)
    domain_results, draws, cluster_draws = {}, {}, {}
    for domain, values in domains.items():
        vector = np.array([x[0] for x in values])
        draws[domain] = vector[rng.integers(len(vector), size=(repetitions, len(vector)))].mean(axis=1)
        groups = defaultdict(list)
        for delta, group in values:
            groups[group].append(delta)
        # Cluster resampling preserves all question-level weights within each
        # sampled family/document/subject. Small group counts remain disclosed.
        group_values = list(groups.values())
        sums = np.array([sum(x) for x in group_values])
        counts = np.array([len(x) for x in group_values])
        indices = rng.integers(len(sums), size=(repetitions, len(sums)))
        cluster_draws[domain] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        domain_results[domain] = {'questions': len(vector), 'clusters': len(groups),
            'delta_pp': float(vector.mean() * 100),
            'paired_95ci_pp': [float(x * 100) for x in np.quantile(draws[domain], [.025, .975])],
            'cluster_95ci_pp': [float(x * 100) for x in np.quantile(cluster_draws[domain], [.025, .975])]}
    result = {'comparison': f'{b["model"]}/{b["policy"]} minus {a["model"]}/{a["policy"]}',
        'method': 'mean within question over matching seeds, paired stratified question bootstrap; percentile 95% CI; cluster sensitivity separately',
        'bootstrap_replicates': repetitions, 'domains': domain_results}
    if all(d in draws for d in PRIMARY):
        boot = np.mean([draws[d] for d in PRIMARY], axis=0)
        cluster = np.mean([cluster_draws[d] for d in PRIMARY], axis=0)
        result['primary_macro'] = {
            'delta_pp': float(np.mean([domain_results[d]['delta_pp'] for d in PRIMARY])),
            'paired_95ci_pp': [float(x * 100) for x in np.quantile(boot, [.025, .975])],
            'cluster_95ci_pp': [float(x * 100) for x in np.quantile(cluster, [.025, .975])]}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = paired(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
