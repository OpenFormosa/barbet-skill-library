"""Stream every active v2 row into a separately versioned curriculum index.

This is a use/verification inventory, not new semantic acceptance. Original
answers, PPL choices, splits, arrays and all prior query holds are immutable.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from .common import canonical, digest, read_json, save_new, sha_file


def classify(row):
    refs = json.loads(row["source_refs_json"])
    source = row["source"]
    query = refs.get("query")
    if isinstance(query, dict) and source.startswith(("Wikidata5M/", "ConceptNet/")):
        branches = query.get("branches", [])
        program = {"operation": query.get("operation"), "branches": [
            [{"direction": step.get("direction"), "relation_slot": n}
             for n, step in enumerate(branch.get("path", []))] for branch in branches]}
        semantic = refs.get("query_schema_id") or canonical(program)
        role = "external_memory_and_bounded_operator_candidate"
        signature = "finite_record/" + semantic
        program_id = digest(program)
        status = "source_bound_semantics_previously_verified_no_new_learning_evidence"
    elif source == "OpenFormosa/taiwan-civic-v1":
        role, signature, program_id = "values_and_attributed_civic_candidate", None, None
        status = "prior_values_admission_preserved"
    elif source in {"hotpotqa/hotpot_qa", "rajpurkar/squad_v2"}:
        role, signature, program_id = "evidence_binding_and_retrieval_candidate", None, None
        status = "source_gold_span_or_support_checks_not_new_semantic_review"
    elif row["domain"] in {"tool_use", "coding"}:
        role, signature, program_id = "execution_reverification_required", None, None
        status = "upstream_only_no_new_execution_claim"
    elif row["domain"] == "math":
        role, signature, program_id = "problem_relation_and_solution_reverification_required", None, None
        status = "upstream_only_no_new_math_correctness_claim"
    else:
        role, signature, program_id = "general_content_completion_review_required", None, None
        status = "upstream_only_no_new_content_completion_claim"
    return {
        "record_id": row["record_id"], "root_task_id": row["family_id"],
        "query_hash": row["query_hash"], "source": source, "domain": row["domain"],
        "split": row["split"], "semantic_family_id": signature,
        "semantic_family_status": "derived_from_explicit_query" if signature else "UNRESOLVED",
        "program_graph_id": program_id, "curriculum_role": role,
        "verification_tier_inherited": row["verification_tier"], "current_review_scope": status,
        "input_tokens": row["pangolin_tokens"], "supervised_tokens": row["pangolin_supervised_tokens"],
        "memory_placement": row["memory_placement"], "tool_observations": row["tool_observations"],
        "source_refs_sha256": digest(refs),
        "learning_status": "NOT_RUN", "transfer_status": "NOT_RUN", "retention_status": "NOT_RUN",
        "eligible_for_automatic_training": False,
    }


def inventory_jobs(work):
    work = Path(work)
    prior = read_json(work / "reports/memory-harness-v2-barbet-readiness-v1/plan.json")
    holds = {}
    plans = [prior,
             read_json(work / "reports/taiwan-civic-v1/publication-plan.json"),
             read_json(work / "reports/value-alignment-v1/publication-plan.json")]
    for plan in plans:
        for job in plan["jobs"]:
            holds.setdefault(job["path"], set()).update(job["exclude_indices"])
    jobs = []
    for item in prior["base_manifest"]:
        if "/records/" in item["path"]:
            jobs.append({**item, "local": str(work / "exports/memory-harness-v2-cumulative-0010" / item["path"]),
                         "excluded_indices": sorted(holds.get(item["path"], set()))})
    civic = read_json(work / "reports/taiwan-civic-v1/new-cohort-summary.json")
    for item in civic["files"]:
        if "/records/" in item["path"]:
            jobs.append({**item, "local": str(work / "exports/taiwan-civic-v1" / item["path"]),
                         "excluded_indices": sorted(holds.get(item["path"], set()))})
    if len({j["path"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate input shard")
    return sorted(jobs, key=lambda j: j["path"])


def _scan(job, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = Path(job["local"])
    if sha_file(source) != job["sha256"]:
        raise ValueError("Original source changed: " + job["path"])
    name = digest(job["path"])
    target = Path(output) / "index" / (name + ".parquet")
    receipt = Path(output) / "receipts" / (name + ".json")
    binding = digest(job)
    if receipt.exists():
        result = read_json(receipt)
        if result["binding"] != binding or sha_file(target) != result["sha256"]:
            raise ValueError("Frozen shard receipt changed")
        return result
    if target.exists():
        raise ValueError("Unreceipted output exists; retain it and investigate")
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["record_id", "family_id", "query_hash", "source", "domain", "split", "pangolin_tokens",
               "pangolin_supervised_tokens", "source_refs_json", "memory_placement", "verification_tier", "tool_observations"]
    excluded = set(job["excluded_indices"])
    counts, mixtures, schema_counts = Counter(), {}, Counter()
    writer, offset = None, 0
    schema = pa.schema([(k, pa.string()) for k in ["record_id", "root_task_id", "query_hash", "source", "domain", "split",
                       "semantic_family_id", "semantic_family_status", "program_graph_id", "curriculum_role",
                       "verification_tier_inherited", "current_review_scope"]] +
                       [("input_tokens", pa.int64()), ("supervised_tokens", pa.int64()), ("memory_placement", pa.string()),
                        ("tool_observations", pa.int64()), ("source_refs_sha256", pa.string()),
                        ("learning_status", pa.string()), ("transfer_status", pa.string()), ("retention_status", pa.string()),
                        ("eligible_for_automatic_training", pa.bool_())])
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=2048, columns=columns):
            selected = []
            for n, row in enumerate(batch.to_pylist(), offset):
                if n in excluded:
                    counts["excluded_prior_hold"] += 1
                    continue
                value = classify(row)
                selected.append(value)
                counts.update({"rows": 1, "input_tokens": value["input_tokens"], "supervised_tokens": value["supervised_tokens"],
                               "split:" + value["split"]: 1, "source:" + value["source"]: 1,
                               "role:" + value["curriculum_role"]: 1, "semantic_status:" + value["semantic_family_status"]: 1})
                key = value["source"] + "|" + value["domain"]
                mixtures.setdefault(key, Counter()).update({"rows": 1, "input_tokens": value["input_tokens"],
                                                           "supervised_tokens": value["supervised_tokens"]})
                if value["semantic_family_id"]:
                    schema_counts[value["semantic_family_id"]] += 1
            offset += len(batch)
            if selected:
                if writer is None:
                    writer = pq.ParquetWriter(target, schema, compression="zstd")
                writer.write_table(pa.Table.from_pylist(selected, schema=schema))
    finally:
        if writer is not None:
            writer.close()
    if excluded and max(excluded) >= offset:
        raise ValueError("Exclusion outside original shard")
    if counts["excluded_prior_hold"] != len(excluded):
        raise ValueError("Hold application differs")
    if writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=schema), target, compression="zstd")
    if pq.ParquetFile(target).metadata.num_rows != counts["rows"]:
        raise ValueError("Saved index row count differs")
    result = {"binding": binding, "source_path": job["path"], "source_sha256": job["sha256"],
              "output_path": str(target), "sha256": sha_file(target), "bytes": target.stat().st_size,
              "counts": dict(counts), "mixtures": {k: dict(v) for k, v in mixtures.items()},
              "semantic_families": dict(schema_counts)}
    save_new(receipt, result)
    return result


def audit_corpus(work, output, workers=4):
    work, output = Path(work), Path(output)
    if shutil.disk_usage(output.parent).free < 50 * 1024**3:
        raise ValueError("Need at least 50 GiB free before indexing")
    jobs = inventory_jobs(work)
    output.mkdir(parents=True, exist_ok=True)
    frozen = {"jobs": jobs, "dataset_revision": "b88f4ff7171bb6fb8c1b775ae21de814ea2123fb",
              "driver_sha256": sha_file(__file__), "scope": "all active rows, no target changes or new semantic certification"}
    plan = output / "plan.json"
    if plan.exists():
        if read_json(plan) != frozen:
            raise ValueError("Frozen corpus plan changed")
    else:
        save_new(plan, frozen)
    counts, mixtures, families, receipts = Counter(), {}, Counter(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(lambda j: _scan(j, output), jobs), 1):
            receipts.append({k: result[k] for k in ["source_path", "output_path", "sha256", "bytes"]})
            counts.update(result["counts"])
            families.update(result["semantic_families"])
            for k, v in result["mixtures"].items():
                mixtures.setdefault(k, Counter()).update(v)
            if n % 100 == 0 or n == len(jobs):
                print(json.dumps({"shards": n, "total_shards": len(jobs), "active_rows": counts["rows"]}), flush=True)
    expected = read_json(work / "reports/value-alignment-v1/summary.json")["counts"]
    for k, key in [("rows", "accepted"), ("input_tokens", "pangolin_tokens"), ("supervised_tokens", "pangolin_supervised_tokens")]:
        if counts[k] != expected[key]:
            raise ValueError("Full release total differs: " + k)
    for k, value in expected.items():
        if k.startswith(("source:", "split:")) and counts[k] != value:
            raise ValueError("Full release partition differs: " + k)
    result = {"dataset_revision": frozen["dataset_revision"], "plan_sha256": sha_file(plan),
              "counts": dict(counts), "mixtures": {k: dict(v) for k, v in mixtures.items()},
              "explicit_semantic_families": dict(families), "files": receipts,
              "all_active_rows_accounted_for": True, "old_holds_applied": True,
              "targets_ppl_splits_and_arrays_unchanged": True, "new_semantic_acceptance": False,
              "learning_transfer_retention": "NOT_RUN"}
    save_new(output / "summary.json", result)
    return result
