"""Offline outcomes and grouped paired comparisons; never actor feedback."""
from __future__ import annotations

from collections import defaultdict
import random


def paired_comparison(parent, candidate, *, seed=20261104, resamples=2000,
                      noninferiority_margin=None, min_roots=None):
    """Maps example IDs to {'root_task_id': str, 'correct': bool}.

Bootstrap resamples roots, not correlated views. No promotion claim is made
without an explicitly registered margin and minimum sample requirement.
"""
    if not parent or set(parent) != set(candidate):
        raise ValueError("Paired evaluations require exactly the same example IDs")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("At least 100 explicit root-bootstrap resamples required")
    if noninferiority_margin is not None and (type(noninferiority_margin) not in {int,float} or not 0 <= noninferiority_margin < 1):
        raise ValueError("Invalid noninferiority margin")
    if min_roots is not None and (type(min_roots) is not int or min_roots < 2):
        raise ValueError("Invalid registered minimum root count")
    groups = defaultdict(list)
    lost = gained = 0
    for ident in sorted(parent):
        a, b = parent[ident], candidate[ident]
        if a["root_task_id"] != b["root_task_id"] or type(a["correct"]) is not bool or type(b["correct"]) is not bool:
            raise ValueError("Paired roots or outcome types differ")
        groups[a["root_task_id"]].append(int(b["correct"]) - int(a["correct"]))
        lost += a["correct"] and not b["correct"]
        gained += b["correct"] and not a["correct"]
    roots = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        draws = [groups[rng.choice(roots)] for _ in roots]
        deltas.append(sum(sum(x) for x in draws) / sum(len(x) for x in draws))
    deltas.sort()
    low = deltas[int(.05 * (len(deltas) - 1))]
    if noninferiority_margin is None or min_roots is None:
        verdict = "UNREGISTERED_NO_NONINFERIORITY_DECISION"
    elif len(roots) < min_roots:
        verdict = "UNCERTAIN_INSUFFICIENT_ROOTS"
    else:
        verdict = "noninferiority_bound_met" if low >= -noninferiority_margin else "noninferiority_not_established"
    return {"examples": len(parent), "roots": len(roots), "lost": lost, "gained": gained,
            "paired_accuracy_delta": (gained - lost) / len(parent), "one_sided_95_lower": low,
            "root_group_bootstrap_seed": seed, "resamples": resamples, "verdict": verdict}


def paired_condition_score(rows):
    groups = defaultdict(dict)
    for r in rows:
        g = groups[r["counterfactual_group_id"]]
        if r["variant"] in g:
            raise ValueError("Duplicate paired variant")
        if type(r["correct"]) is not bool:
            raise ValueError("Outcome must be boolean")
        g[r["variant"]] = r["correct"]
    if not groups or any(not {"original", "counterfactual", "invariance"} <= set(g) for g in groups.values()):
        raise ValueError("Incomplete semantic pair")
    return {"groups": len(groups), "all_three_correct": sum(all(g.values()) for g in groups.values()),
            "pair_correct": sum(g["original"] and g["counterfactual"] for g in groups.values())}


def risk_coverage(rows):
    if not rows:
        raise ValueError("Calibration outcomes required")
    if any(type(r["correct"]) is not bool or type(r["confidence"]) not in {float, int} or not 0 <= r["confidence"] <= 1 for r in rows):
        raise ValueError("Invalid calibrated confidence or correctness")
    curve = [{"threshold": None, "coverage": 0.0, "risk": None}]
    for threshold in sorted({r["confidence"] for r in rows}, reverse=True):
        selected = [r for r in rows if r["confidence"] >= threshold]
        curve.append({"threshold": threshold, "coverage": len(selected) / len(rows),
                      "risk": sum(not r["correct"] for r in selected) / len(selected)})
    return curve


def evaluation_record(*, condition, raw_proposal, host_decision, execution_receipt, semantic_correct, naturally_ended):
    if condition not in {"O0", "O1", "O2", "O3", "O4", "O5"}:
        raise ValueError("Unknown diagnostic condition")
    return {"condition": condition, "raw_proposal": raw_proposal, "host_decision": host_decision,
            "execution_receipt": execution_receipt, "semantic_correct": semantic_correct,
            "naturally_ended": naturally_ended,
            "eligible_for_main_score": condition == "O0", "oracle_assisted": condition != "O0"}
