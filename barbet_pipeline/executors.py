"""Deterministic public tools. No task oracle, candidate repair or source eval.

Receipts describe actual calls into these functions. The optional state ledger
is an explicit local simulator, not a claim of external side effects.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, localcontext
import operator

from .common import digest


class ToolError(ValueError):
    pass


def number(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ToolError("decimal_string_required")
    try:
        n = Decimal(value)
    except InvalidOperation:
        raise ToolError("invalid_decimal") from None
    if not n.is_finite() or n.adjusted() > 30 or n.as_tuple().exponent < -30:
        raise ToolError("decimal_out_of_supported_range")
    return n


def decimal_text(value):
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def exact_keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise ToolError("argument_contract")


def table_select(table, args):
    exact_keys(args, {"table_id", "field", "operation", "threshold", "k", "tie_rule"},
               {"table_id", "field", "operation"})
    if args.get("tie_rule", "id_ascending") != "id_ascending":
        raise ToolError("unsupported_tie_rule")
    op, field = args["operation"], args["field"]
    if not isinstance(field, str) or len(table) > 10000:
        raise ToolError("invalid_table_or_field")
    ids = [row.get("id") for row in table]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ToolError("duplicate_or_invalid_id")
    if any(field not in row or row[field] is None for row in table):
        raise ToolError("required_value_missing")
    values = [(row["id"], number(row[field])) for row in table]
    predicates = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}
    if op in predicates:
        if "threshold" not in args or "k" in args:
            raise ToolError("filter_requires_threshold_only")
        t = number(args["threshold"])
        selected = sorted(i for i, n in values if predicates[op](n, t))
    elif op in {"nth_smallest", "nth_largest", "top_smallest", "top_largest"}:
        k = args.get("k")
        if type(k) is not int or not 1 <= k <= len(values) or "threshold" in args:
            raise ToolError("rank_requires_valid_k_only")
        reverse = op.endswith("largest")
        ordered = sorted(values, key=lambda x: (-x[1] if reverse else x[1], x[0]))
        selected = [ordered[k - 1][0]] if op.startswith("nth") else [x[0] for x in ordered[:k]]
    else:
        raise ToolError("unsupported_operation")
    return {"ids": selected, "operation": op, "field": field, "tie_rule": "id_ascending"}


class StateLedger:
    """A finite simulator with idempotence and visible late-result handling."""
    def __init__(self):
        self.revision = 0
        self.operations = {}

    def submit(self, operation_id, payload):
        if not isinstance(operation_id, str) or not operation_id:
            raise ToolError("operation_id_required")
        identity = digest(payload)
        if operation_id in self.operations:
            if self.operations[operation_id]["payload_sha256"] != identity:
                raise ToolError("idempotency_conflict")
            return copy.deepcopy(self.operations[operation_id])
        self.revision += 1
        self.operations[operation_id] = {"operation_id": operation_id, "payload_sha256": identity,
                                         "status": "submitted", "revision": self.revision, "result": None}
        return copy.deepcopy(self.operations[operation_id])

    def event(self, operation_id, status, result=None):
        if operation_id not in self.operations:
            raise ToolError("unknown_operation")
        current = self.operations[operation_id]
        if current["status"] == "cancelled" and status in {"partial", "completed", "failed"}:
            return {"status": "late_result_ignored", "operation_id": operation_id, "revision": self.revision}
        transitions = {"submitted": {"running", "cancelled", "failed"},
                       "running": {"partial", "completed", "failed", "cancelled"},
                       "partial": {"partial", "completed", "failed", "cancelled"}}
        if status not in transitions.get(current["status"], set()):
            raise ToolError("invalid_state_transition")
        self.revision += 1
        current.update(status=status, result=copy.deepcopy(result), revision=self.revision)
        return copy.deepcopy(current)

    def get(self, operation_id):
        if operation_id not in self.operations:
            raise ToolError("unknown_operation")
        return copy.deepcopy(self.operations[operation_id])


class Executor:
    """Only public world state is accepted. Task answers never enter this class."""
    def __init__(self, *, tables=None, evidence=None, skills=None, ledger=None):
        self.tables = copy.deepcopy(tables or {})
        self.evidence = copy.deepcopy(evidence or {})
        self.skills = copy.deepcopy(skills or {})
        self.ledger = ledger or StateLedger()
        self.receipts = []

    def resolve(self, ref):
        exact_keys(ref, {"table_id", "row_id", "field"}, {"table_id", "row_id", "field"})
        table = self.tables.get(ref["table_id"])
        if table is None:
            raise ToolError("unknown_table")
        matches = [r for r in table if r.get("id") == ref["row_id"]]
        if len(matches) != 1 or ref["field"] not in matches[0]:
            raise ToolError("missing_or_ambiguous_reference")
        return copy.deepcopy(matches[0][ref["field"]])

    def call(self, name, args):
        original = copy.deepcopy(args)
        try:
            from jsonschema import Draft202012Validator
            try:
                schema = tool_schema(name)["function"]["parameters"]
            except (ValueError, TypeError):
                raise ToolError("unregistered_tool") from None
            if next(Draft202012Validator(schema).iter_errors(args), None) is not None:
                raise ToolError("argument_contract")
            result = self._dispatch(name, args)
            output = {"ok": True, "result": result}
        except (ToolError, ZeroDivisionError) as exc:
            output = {"ok": False, "error": str(exc) if isinstance(exc, ToolError) else "division_by_zero"}
        receipt = {"executor": "barbet-public-tools-v3", "origin": "local_deterministic_execution",
                   "ordinal": len(self.receipts), "tool": name, "arguments": original,
                   "arguments_sha256": digest(original), "output": copy.deepcopy(output),
                   "output_sha256": digest(output), "state_revision": self.ledger.revision}
        self.receipts.append(receipt)
        return output

    def _dispatch(self, name, args):
        if name == "table_select":
            table = self.tables.get(args.get("table_id")) if isinstance(args, dict) else None
            if table is None:
                raise ToolError("unknown_table")
            return table_select(table, args)
        if name == "resolve_value":
            return {"value": self.resolve(args)}
        if name == "decimal_compute":
            exact_keys(args, {"operation", "operand_refs"}, {"operation", "operand_refs"})
            refs = args["operand_refs"]
            if not isinstance(refs, list) or not 2 <= len(refs) <= 20:
                raise ToolError("requires_2_to_20_operands")
            values = [number(self.resolve(r)) for r in refs]
            operations = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul, "divide": operator.truediv}
            if args["operation"] not in operations:
                raise ToolError("unsupported_arithmetic")
            with localcontext() as context:
                context.prec = 80
                value = values[0]
                for n in values[1:]:
                    value = operations[args["operation"]](value, n)
            return {"value": decimal_text(value), "precision": 80}
        if name == "convert_unit":
            exact_keys(args, {"value_ref", "from_unit", "to_unit"}, {"value_ref", "from_unit", "to_unit"})
            units = {"mm": ("length", "0.001"), "cm": ("length", "0.01"), "m": ("length", "1"),
                     "g": ("mass", "0.001"), "kg": ("mass", "1"),
                     "s": ("time", "1"), "min": ("time", "60"), "h": ("time", "3600")}
            a, b = units.get(args["from_unit"]), units.get(args["to_unit"])
            if a is None or b is None or a[0] != b[0]:
                raise ToolError("incompatible_units")
            with localcontext() as context:
                context.prec = 80
                value = number(self.resolve(args["value_ref"])) * Decimal(a[1]) / Decimal(b[1])
            return {"value": decimal_text(value), "unit": args["to_unit"]}
        if name == "date_add_days":
            exact_keys(args, {"date_ref", "days"}, {"date_ref", "days"})
            if type(args["days"]) is not int or abs(args["days"]) > 36600:
                raise ToolError("days_out_of_range")
            value = self.resolve(args["date_ref"])
            try:
                result = date.fromisoformat(value) + timedelta(days=args["days"])
            except (ValueError, TypeError, OverflowError):
                raise ToolError("invalid_iso_date") from None
            return {"date": result.isoformat()}
        if name == "read_evidence":
            exact_keys(args, {"document_id"}, {"document_id"})
            if args["document_id"] not in self.evidence:
                raise ToolError("document_unavailable")
            return copy.deepcopy(self.evidence[args["document_id"]])
        if name == "search_evidence":
            exact_keys(args, {"query", "k"}, {"query", "k"})
            if not isinstance(args["query"], str) or type(args["k"]) is not int or not 1 <= args["k"] <= 16:
                raise ToolError("invalid_search")
            words = args["query"].casefold().split()
            if not words:
                raise ToolError("empty_search")
            ranked = []
            for key, doc in self.evidence.items():
                body = (doc.get("title", "") + " " + doc.get("text", "")).casefold()
                score = sum(w in body for w in words)
                if score:
                    ranked.append((-score, key))
            return [copy.deepcopy(self.evidence[k]) for _, k in sorted(ranked)[:args["k"]]]
        if name == "read_skill":
            exact_keys(args, {"skill_id", "version"}, {"skill_id", "version"})
            key = args["skill_id"] + "@" + args["version"]
            if key not in self.skills:
                raise ToolError("skill_version_unavailable")
            return copy.deepcopy(self.skills[key])
        if name == "read_operation_state":
            exact_keys(args, {"operation_id"}, {"operation_id"})
            return self.ledger.get(args["operation_id"])
        if name == "cancel_operation":
            exact_keys(args, {"operation_id"}, {"operation_id"})
            return self.ledger.event(args["operation_id"], "cancelled")
        raise ToolError("unregistered_tool")


def tool_schema(name):
    string = {"type": "string"}
    reference = {"type": "object", "properties": {k: string for k in ["table_id", "row_id", "field"]},
                 "required": ["table_id", "row_id", "field"], "additionalProperties": False}
    definitions = {
        "table_select": ("Select visible table row IDs. gt/ge/lt/le compare to a Decimal threshold; nth/top use k. Ties use id_ascending; nth ranks rows, not distinct values.",
            {"table_id": string, "field": string, "operation": {"type": "string", "enum": ["gt", "ge", "lt", "le", "nth_smallest", "nth_largest", "top_smallest", "top_largest"]},
             "threshold": string, "k": {"type": "integer", "minimum": 1}, "tie_rule": {"type": "string", "enum": ["id_ascending"]}}, ["table_id", "field", "operation"]),
        "resolve_value": ("Copy exactly the selected visible field. Missing, null, false, numeric strings and IDs are distinct.", reference["properties"], reference["required"]),
        "decimal_compute": ("Compute in listed operand order using visible references and Decimal precision 80.",
            {"operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}, "operand_refs": {"type": "array", "items": reference}}, ["operation", "operand_refs"]),
        "convert_unit": ("Convert compatible mm/cm/m, g/kg, s/min/h units using a visible numeric reference.",
            {"value_ref": reference, "from_unit": string, "to_unit": string}, ["value_ref", "from_unit", "to_unit"]),
        "date_add_days": ("Add signed calendar days to a visible ISO date.", {"date_ref": reference, "days": {"type": "integer"}}, ["date_ref", "days"]),
        "read_evidence": ("Read a currently available document. Missing documents return document_unavailable.", {"document_id": string}, ["document_id"]),
        "search_evidence": ("Search currently available public evidence by literal words; returns matched documents only.", {"query": string, "k": {"type": "integer", "minimum": 1, "maximum": 16}}, ["query", "k"]),
        "read_skill": ("Read an exact registered skill version; never silently substitute another version.", {"skill_id": string, "version": string}, ["skill_id", "version"]),
        "read_operation_state": ("Read observable operation status; submitted or running does not imply completed.", {"operation_id": string}, ["operation_id"]),
        "cancel_operation": ("Cancel a pending simulated operation; late results cannot undo cancellation.", {"operation_id": string}, ["operation_id"]),
    }
    if name not in definitions:
        raise ValueError("No implemented tool: " + name)
    description, properties, required = definitions[name]
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
