"""Private semantic checks independent of the runtime's selection algorithm.

No functions in this module are registered as actor tools. Rankings are checked
by pairwise ranks (not the runtime sort); unit answers use rational arithmetic.
"""
from __future__ import annotations

from fractions import Fraction
import copy
from .common import canonical


def solve(spec):
    kind = spec["kind"]
    if kind in {"comparison", "ordering", "rule_version"}:
        values = [(r["id"], Fraction(r[spec["field"]])) for r in spec["rows"]]
        op = spec["operation"]
        if op in {"gt", "ge", "lt", "le"}:
            t = Fraction(spec["threshold"])
            expected = []
            for row_id, value in values:
                delta = value - t
                accept = {"gt": delta > 0, "ge": delta >= 0, "lt": delta < 0, "le": delta <= 0}[op]
                if accept:
                    expected.append(row_id)
            return sorted(expected)
        if op not in {"nth_smallest", "nth_largest", "top_smallest", "top_largest"}:
            raise ValueError("Unsupported private semantic operator")
        k = spec["k"]
        if type(k) is not int or not 1 <= k <= len(values):
            raise ValueError("Invalid private rank")
        largest = op.endswith("largest")
        rank = {}
        for row_id, value in values:
            rank[row_id] = 1 + sum((other > value if largest else other < value) or
                                   (other == value and other_id < row_id)
                                   for other_id, other in values if other_id != row_id)
        if op.startswith("nth"):
            return [row_id for row_id, r in rank.items() if r == k]
        return [row_id for r in range(1, k + 1) for row_id, found in rank.items() if found == r]
    if kind == "binding":
        matches = [r for r in spec["rows"] if r["id"] == spec["row_id"]]
        if len(matches) != 1:
            raise ValueError("Ambiguous private binding")
        return copy.deepcopy(matches[0][spec["field"]])
    if kind == "units":
        scale = {"mm": Fraction(1, 1000), "cm": Fraction(1, 100), "m": Fraction(1),
                 "g": Fraction(1, 1000), "kg": Fraction(1), "s": Fraction(1), "min": Fraction(60), "h": Fraction(3600)}
        amount = next(r[spec["field"]] for r in spec["rows"] if r["id"] == spec["row_id"])
        value = Fraction(amount) * scale[spec["from_unit"]] / scale[spec["to_unit"]]
        return {"numerator": value.numerator, "denominator": value.denominator, "unit": spec["to_unit"]}
    if kind == "evidence_state":
        if spec["state"] == "present":
            return {"behavior": "answer", "value": spec["answer"]}
        if spec["state"] == "missing_rule":
            return {"behavior": "clarify", "missing": "applicable_rule"}
        if spec["state"] == "conflicting_rules":
            return {"behavior": "clarify", "missing": "rule_precedence"}
        if spec["state"] == "retrievable":
            return {"behavior": "retrieve_then_answer", "value": spec["answer"]}
        if spec["state"] == "unavailable":
            return {"behavior": "report_unavailable", "missing": "source_document"}
        raise ValueError("Unknown information state")
    if kind == "operation_state":
        return {"status": spec["status"], "completed": spec["status"] == "completed"}
    raise ValueError("Unknown semantic kind")


def verify_execution(spec, result):
    """Verify actual execution values, not just schema or tool-name validity."""
    if not result.get("ok"):
        return False
    expected = solve(spec)
    actual = result.get("result")
    if not isinstance(actual, dict):
        return False
    if spec["kind"] in {"comparison", "ordering", "rule_version"}:
        return actual.get("ids") == expected
    if spec["kind"] == "binding":
        if "value" not in actual:
            return False
        value = actual.get("value")
        return type(value) is type(expected) and value == expected
    if spec["kind"] == "units":
        return (actual.get("unit") == expected["unit"] and
                Fraction(actual["value"]) == Fraction(expected["numerator"], expected["denominator"]))
    if spec["kind"] == "operation_state":
        return actual.get("status") == expected["status"]
    return False


def require_discriminating(original, counterfactual, invariant):
    a, b, c = solve(original), solve(counterfactual), solve(invariant)
    if canonical(a) == canonical(b):
        raise ValueError("Counterfactual does not change the semantic answer")
    if canonical(a) != canonical(c):
        raise ValueError("Invariance transformation changed the semantic answer")
    return {"counterfactual_answer_changed": True, "invariance_answer_preserved": True}
