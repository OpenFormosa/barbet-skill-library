import copy
import unittest

from barbet_pipeline.executors import Executor, StateLedger, ToolError
from barbet_pipeline.oracle import solve, verify_execution, require_discriminating
from barbet_pipeline.visibility import actor_messages, visible_prefix
from barbet_pipeline.evaluation import paired_comparison, paired_condition_score, risk_coverage, evaluation_record


class SemanticToolsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": i, "score": n, "literal": v} for i, n, v in
                     [("A", "9", "001"), ("B", "2", False), ("C", "6", None), ("D", "4", "0")]]
        self.executor = Executor(tables={"t": self.rows})

    def spec(self, **kwargs):
        return {"kind": "comparison", "rows": self.rows, "field": "score", "operation": "gt", "threshold": "6", **kwargs}

    def test_gt_ge_discriminate_and_wrong_operator_fails_semantics(self):
        for op, expected in [("gt", ["A"]), ("ge", ["A", "C"]), ("lt", ["B", "D"]), ("le", ["B", "C", "D"])]:
            result = self.executor.call("table_select", {"table_id": "t", "field": "score", "operation": op, "threshold": "6"})
            self.assertEqual(result["result"]["ids"], expected)
            self.assertTrue(verify_execution(self.spec(operation=op), result))
        wrong = self.executor.call("table_select", {"table_id": "t", "field": "score", "operation": "ge", "threshold": "6"})
        self.assertFalse(verify_execution(self.spec(), wrong))

    def test_independent_rank_oracle_ties_negatives_and_zero(self):
        rows = [{"id": str(i), "score": str(v)} for i, v in enumerate([0, -2, 3, 3, 8])]
        e = Executor(tables={"t": rows})
        for op in ["nth_smallest", "nth_largest", "top_smallest", "top_largest"]:
            for k in range(1, 6):
                s = {"kind": "ordering", "rows": rows, "field": "score", "operation": op, "k": k}
                out = e.call("table_select", {"table_id": "t", "field": "score", "operation": op, "k": k})
                self.assertTrue(verify_execution(s, out), (op, k))

    def test_non_discriminating_pair_is_rejected(self):
        s = self.spec(threshold="5")
        with self.assertRaises(ValueError):
            require_discriminating(s, {**s, "operation": "ge"}, s)
        original = self.spec()
        inv = copy.deepcopy(original); inv["rows"].reverse()
        self.assertTrue(require_discriminating(original, self.spec(operation="ge"), inv)["counterfactual_answer_changed"])

    def test_exact_types_and_missing_are_distinct(self):
        for row_id, expected in [("A", "001"), ("B", False), ("C", None), ("D", "0")]:
            out = self.executor.call("resolve_value", {"table_id": "t", "row_id": row_id, "field": "literal"})
            self.assertIs(type(out["result"]["value"]), type(expected)); self.assertEqual(out["result"]["value"], expected)
        out = self.executor.call("resolve_value", {"table_id": "t", "row_id": "A", "field": "missing"})
        self.assertFalse(out["ok"])

    def test_schema_rejection_is_receipted_no_silent_repair(self):
        args = {"table_id": "t", "field": "score", "operation": "nth_smallest", "k": True}
        self.assertFalse(self.executor.call("table_select", args)["ok"])
        self.assertIs(self.executor.receipts[-1]["arguments"]["k"], True)
        self.assertFalse(self.executor.call("unregistered", {})["ok"])

    def test_units_decimal_dates_and_empty_set(self):
        spec = {"kind": "units", "rows": self.rows, "row_id": "C", "field": "score", "from_unit": "m", "to_unit": "cm"}
        out = self.executor.call("convert_unit", {"value_ref": {"table_id": "t", "row_id": "C", "field": "score"}, "from_unit": "m", "to_unit": "cm"})
        self.assertTrue(verify_execution(spec, out))
        self.assertFalse(self.executor.call("convert_unit", {"value_ref": {"table_id": "t", "row_id": "C", "field": "score"}, "from_unit": "m", "to_unit": "kg"})["ok"])
        e = Executor(tables={"empty": [], "dates": [{"id": "x", "date": "2024-02-28"}]})
        self.assertEqual(e.call("table_select", {"table_id": "empty", "field": "x", "operation": "gt", "threshold": "0"})["result"]["ids"], [])
        self.assertEqual(e.call("date_add_days", {"date_ref": {"table_id": "dates", "row_id": "x", "field": "date"}, "days": 2})["result"]["date"], "2024-03-01")

    def test_search_failure_and_recovery_use_actual_public_store(self):
        e = Executor(evidence={"d": {"id": "d", "title": "供應 規則", "text": "有效版本為 v2", "revision": "2"}})
        self.assertEqual(e.call("search_evidence", {"query": "不存在", "k": 2})["result"], [])
        self.assertEqual(e.call("search_evidence", {"query": "供應", "k": 2})["result"][0]["id"], "d")
        self.assertFalse(e.call("read_evidence", {"document_id": "future"})["ok"])

    def test_cancelled_late_result_cannot_mark_completed(self):
        ledger = StateLedger(); ledger.submit("op", {"x": 1}); ledger.event("op", "running")
        ledger.event("op", "cancelled")
        self.assertEqual(ledger.event("op", "completed", {"ok": True})["status"], "late_result_ignored")
        self.assertEqual(ledger.get("op")["status"], "cancelled")
        with self.assertRaises(ToolError): ledger.submit("op", {"x": 2})


class BoundaryAndMetricsTest(unittest.TestCase):
    def test_private_fields_and_canaries_never_reach_actor(self):
        p = {"task": "Answer from the records.", "context": [], "skills": [], "tools": []}
        self.assertEqual(len(actor_messages(p)), 2)
        with self.assertRaises(ValueError): actor_messages({**p, "private_oracle": {"answer": "x"}})
        with self.assertRaises(ValueError): actor_messages({**p, "task": "PRIVATE-CANARY"}, forbidden_canaries=["PRIVATE-CANARY"])
        with self.assertRaises(ValueError): actor_messages({**p, "context": [{"id": "d", "title": "d", "text": "x", "revision": "1", "gold": "x"}]})

    def test_visibility_cuts_before_target_and_future(self):
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "target", "loss": True}, {"role": "tool", "content": "future"}]
        self.assertEqual(visible_prefix(messages, 1), messages[:1])

    def test_paired_losses_not_hidden_by_aggregate(self):
        parent = {str(i): {"root_task_id": str(i // 2), "correct": i < 4} for i in range(8)}
        candidate = {str(i): {"root_task_id": str(i // 2), "correct": 2 <= i < 6} for i in range(8)}
        r = paired_comparison(parent, candidate, resamples=100)
        self.assertEqual((r["lost"], r["gained"], r["paired_accuracy_delta"]), (2, 2, 0))
        self.assertEqual(r["verdict"], "UNREGISTERED_NO_NONINFERIORITY_DECISION")
        self.assertEqual(paired_comparison(parent, candidate, resamples=100, min_roots=100, noninferiority_margin=.02)["verdict"], "UNCERTAIN_INSUFFICIENT_ROOTS")

    def test_pair_and_risk_reject_empty_success(self):
        rows = [{"counterfactual_group_id": "r", "variant": v, "correct": v != "counterfactual"} for v in ["original", "counterfactual", "invariance"]]
        self.assertEqual(paired_condition_score(rows)["pair_correct"], 0)
        self.assertIsNone(risk_coverage([{"confidence": .9, "correct": False}])[0]["risk"])
        self.assertEqual(risk_coverage([{"confidence": .9, "correct": False}])[-1]["risk"], 1)
        self.assertFalse(evaluation_record(condition="O4", raw_proposal={}, host_decision={}, execution_receipt={}, semantic_correct=True, naturally_ended=True)["eligible_for_main_score"])


if __name__ == "__main__":
    unittest.main()
