from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import ir_measures
from scipy.stats import kendalltau

from .baselines import _read_approved_trec_run
from .benchmark import SNAPSHOT_ID, load_jsonl
from .contract import validate_record
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .smoke import _read_qrels, _verify_manifest_outputs
from .verify import verify_source


DEPTHS = (20, 30, 40, 50)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _system_order(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda system: (-scores[system], system))


def _order_tau(reference: list[str], candidate: list[str]) -> float:
    candidate_rank = {system: rank for rank, system in enumerate(candidate)}
    value = kendalltau(
        list(range(len(reference))),
        [candidate_rank[system] for system in reference],
        variant="b",
    ).statistic
    return float(value)


def _agent_pool_is_stable(depth_rows: list[dict[str, Any]]) -> bool:
    return len(depth_rows) >= 3 and all(
        row["new_grade_2_or_3_yield"] < 0.01
        and row["leave_one_run_out_min_kendall_tau"] >= 0.95
        for row in depth_rows[-2:]
    )


def _annotation_label(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["retrieval_grade"],
        tuple(row["selected_source_span_ids"]),
        tuple(row["selected_nugget_ids"]),
    )


def _negative_classes(candidate: dict[str, Any], query: dict[str, Any]) -> list[str]:
    if query["no_answer_reason"] == "wrong_version":
        return ["wrong_version"]
    mapping = {
        "wrong_module_candidate": "wrong_module",
        "lexical_candidate": "lexical_collision",
        "semantic_candidate": "semantic_nearest",
        "hybrid_candidate": "baseline_false_positive",
        "rerank_candidate": "baseline_false_positive",
    }
    return sorted({mapping[value] for value in candidate["provenance"] if value in mapping})


def _reported_annotation_sha(report: dict[str, Any]) -> str | None:
    return (
        report.get("annotation_sha256")
        or report.get("annotation_file_sha256")
        or report.get("adjudicator_sha256")
        or report.get("input_hashes", {}).get("annotation")
        or next(
            (
                output.get("sha256")
                for output in report.get("bound_outputs", [])
                if output.get("path", "").endswith("/adjudicator.jsonl")
            ),
            None,
        )
    )


def _write_adjudicated_review_outputs(
    pool_root: Path,
    annotation_sources: list[tuple[int, Path]],
    candidates: dict[str, dict[str, Any]],
    paths: LabPaths,
) -> dict[str, str]:
    if [depth for depth, _ in annotation_sources] != list(DEPTHS):
        return {}
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    queries = {row["id"]: row for row in load_jsonl(benchmark_root / "queries.jsonl")}
    all_rows = []
    queue_by_id: dict[str, dict[str, Any]] = {}
    agreed: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    annotation_hashes = {}
    for depth, adjudicator_path in annotation_sources:
        annotation_dir = adjudicator_path.parent
        frontier_root = annotation_dir.parent
        context_path = (
            pool_root / "active_frontier_context.jsonl"
            if depth == 20
            else frontier_root / "context.jsonl"
        )
        contexts = {row["id"]: row for row in load_jsonl(context_path)}
        files = {
            "annotator_a": annotation_dir / "annotator_a.jsonl",
            "annotator_b": annotation_dir / "annotator_b.jsonl",
            "adjudicator": adjudicator_path,
        }
        annotations = {
            name: {row["id"]: row for row in load_jsonl(path)}
            for name, path in files.items()
        }
        annotation_hashes.update(
            {
                f"depth{depth}/{name}.jsonl": sha256_file(path)
                for name, path in files.items()
            }
        )
        for item_id in sorted(annotations["adjudicator"]):
            a = annotations["annotator_a"][item_id]
            b = annotations["annotator_b"][item_id]
            adjudicated = annotations["adjudicator"][item_id]
            candidate = candidates[item_id]
            query = queries[adjudicated["query_id"]]
            context = contexts[item_id]
            row = {
                "id": "human-review-" + stable_id(item_id),
                "pool_item_id": item_id,
                "depth": depth,
                "query": query,
                "candidate": candidate,
                "context": context,
                "annotator_a": a,
                "annotator_b": b,
                "adjudicator": adjudicated,
                "selection_reasons": [],
            }
            all_rows.append(row)
            if _annotation_label(a) != _annotation_label(b):
                row["selection_reasons"].append("agent_disagreement")
            if query["answerability"] == "no_answer":
                row["selection_reasons"].append("no_answer")
            if row["selection_reasons"]:
                queue_by_id[item_id] = row
            elif query["answerability"] == "answerable":
                agreed[(adjudicated["retrieval_grade"], query["intent"])].append(row)

    sampled = 0
    strata = {key: sorted(rows, key=lambda row: row["pool_item_id"]) for key, rows in agreed.items()}
    while sampled < 15 and any(strata.values()):
        for key in sorted(strata):
            if sampled >= 15:
                break
            if strata[key]:
                row = strata[key].pop(0)
                row["selection_reasons"].append("stratified_agreement_sample")
                queue_by_id[row["pool_item_id"]] = row
                sampled += 1
    if sampled < 15:
        raise RuntimeError("Human review package could not produce 15 agreed stratified samples")

    selected_negatives = []
    coverage = []
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_query[row["query"]["id"]].append(row)
    for query_id in sorted(queries):
        query = queries[query_id]
        nonpositive = [
            row for row in by_query[query_id] if row["adjudicator"]["retrieval_grade"] <= 1
        ]
        chosen: list[tuple[dict[str, Any], str]] = []
        used_items = set()
        if query["answerability"] == "answerable":
            class_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in nonpositive:
                for provenance in _negative_classes(row["candidate"], query):
                    class_rows[provenance].append(row)
            for provenance in sorted(class_rows):
                candidates_for_class = sorted(
                    class_rows[provenance],
                    key=lambda row: (
                        -row["adjudicator"]["retrieval_grade"],
                        row["candidate"]["minimum_rank"],
                        row["pool_item_id"],
                    ),
                )
                row = next(
                    (candidate for candidate in candidates_for_class if candidate["pool_item_id"] not in used_items),
                    None,
                )
                if row is not None:
                    chosen.append((row, provenance))
                    used_items.add(row["pool_item_id"])
                if len(chosen) == 3:
                    break
            if len(chosen) < 3:
                for row in sorted(nonpositive, key=lambda item: item["pool_item_id"]):
                    if row["pool_item_id"] in used_items:
                        continue
                    classes = _negative_classes(row["candidate"], query)
                    if classes:
                        chosen.append((row, classes[0]))
                        used_items.add(row["pool_item_id"])
                    if len(chosen) == 3:
                        break
            for row, provenance in chosen:
                source_span_id = sorted(
                    {
                        span_id
                        for fragment in row["context"]["span_fragments"]
                        for span_id in fragment["source_span_ids"]
                    }
                )[0]
                negative = {
                    "id": "negative-" + stable_id(query_id, row["pool_item_id"], provenance),
                    "schema_version": 1,
                    "query_id": query_id,
                    "source_span_id": source_span_id,
                    "grade": row["adjudicator"]["retrieval_grade"],
                    "provenance": provenance,
                    "status": "provisional",
                }
                validate_record("HardNegative", negative, paths)
                selected_negatives.append(negative)
        provenance_classes = sorted({provenance for _, provenance in chosen})
        coverage.append(
            {
                "query_id": query_id,
                "answerability": query["answerability"],
                "judged_nonpositive_count": len(nonpositive),
                "selected_hard_negative_count": len(chosen),
                "selected_provenance_classes": provenance_classes,
                "hard_negative_requirement_met": (
                    len(chosen) >= 3 and len(provenance_classes) >= 2
                    if query["answerability"] == "answerable"
                    else None
                ),
                "no_answer_five_judgments_met": (
                    len(nonpositive) >= 5 if query["answerability"] == "no_answer" else None
                ),
            }
        )
    if not all(
        row["hard_negative_requirement_met"] is not False
        and row["no_answer_five_judgments_met"] is not False
        for row in coverage
    ):
        raise RuntimeError("Adjudicated pool does not meet hard-negative/no-answer coverage")

    adjudicated_root = pool_root / "adjudicated"
    write_jsonl(adjudicated_root / "hard_negatives.jsonl", selected_negatives, paths=paths)
    write_jsonl(adjudicated_root / "coverage_report.jsonl", coverage, paths=paths)
    queue = sorted(queue_by_id.values(), key=lambda row: (row["query"]["id"], row["depth"], row["pool_item_id"]))
    human_root = pool_root / "human_review"
    write_jsonl(human_root / "queue.jsonl", queue, paths=paths)
    write_jsonl(
        human_root / "decisions.template.jsonl",
        (
            {
                "id": row["id"],
                "query_id": row["query"]["id"],
                "pool_item_id": row["pool_item_id"],
                "selection_reasons": row["selection_reasons"],
                "human_grade": None,
                "selected_source_span_ids": [],
                "selected_nugget_ids": [],
                "decision": "PENDING",
                "reviewer": None,
                "rationale": "",
            }
            for row in queue
        ),
        paths=paths,
    )
    queue_counts = {
        "agent_disagreements": sum("agent_disagreement" in row["selection_reasons"] for row in queue),
        "no_answer_candidates": sum("no_answer" in row["selection_reasons"] for row in queue),
        "stratified_agreement_samples": sampled,
        "total_unique_candidates": len(queue),
    }
    receipt_template = {
        "schema_version": 1,
        "phase": "seed50-pooled-human-review",
        "decision": "PENDING",
        "reviewer": None,
        "human_review_complete": False,
        "reviewed_ids": [],
        "corrections": [],
        "required_scope": queue_counts,
    }
    write_json(human_root / "receipt.template.json", receipt_template, paths)
    output_paths = {
        "adjudicated/hard_negatives.jsonl": adjudicated_root / "hard_negatives.jsonl",
        "adjudicated/coverage_report.jsonl": adjudicated_root / "coverage_report.jsonl",
        "human_review/queue.jsonl": human_root / "queue.jsonl",
        "human_review/decisions.template.jsonl": human_root / "decisions.template.jsonl",
        "human_review/receipt.template.json": human_root / "receipt.template.json",
    }
    output_hashes = {name: sha256_file(path) for name, path in output_paths.items()}
    write_json(
        adjudicated_root / "manifest.json",
        {
            "id": stable_id(canonical_json(annotation_hashes), canonical_json(output_hashes), length=40),
            "schema_version": 1,
            "status": "agent_diagnostic_stable_human_review_pending",
            "annotation_hashes": annotation_hashes,
            "output_hashes": output_hashes,
            "human_review_queue_counts": queue_counts,
            "hard_negative_requirements_met": True,
            "no_answer_requirements_met": True,
            "human_review_complete": False,
            "pooling_stable": False,
            "seed_frozen": False,
        },
        paths,
    )
    return {**output_hashes, "adjudicated/manifest.json": sha256_file(adjudicated_root / "manifest.json")}


def _seal_completed_annotations(
    output: Path,
    manifest: dict[str, Any],
    benchmark_root: Path,
    evidence_root: Path,
    paths: LabPaths,
) -> dict[str, Any]:
    annotation_dir = output / "annotations"
    data_names = ("annotator_a.jsonl", "annotator_b.jsonl", "adjudicator.jsonl")
    report_names = ("annotator_a.report.json", "annotator_b.report.json", "agreement_report.json")
    completed = [annotation_dir / name for name in data_names + report_names]
    if not any(path.is_file() for path in completed):
        return manifest
    if not all(path.is_file() for path in completed):
        missing = [path.name for path in completed if not path.is_file()]
        raise RuntimeError(f"Incomplete annotation package: {', '.join(missing)}")

    context_ids = [row["id"] for row in load_jsonl(output / "active_frontier_context.jsonl")]
    for name in data_names:
        rows = load_jsonl(annotation_dir / name)
        ids = [row["id"] for row in rows]
        if ids != context_ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"Annotation coverage/order mismatch: {name}")
        for row in rows:
            grade = row["retrieval_grade"]
            if grade in (0, 1) and row["selected_nugget_ids"]:
                raise RuntimeError(f"Grade {grade} cannot bind nuggets: {row['id']}")
            if grade >= 2 and (not row["selected_nugget_ids"] or not row["selected_source_span_ids"]):
                raise RuntimeError(f"Positive annotation lacks evidence binding: {row['id']}")

    reports = {
        name: json.loads((annotation_dir / name).read_text(encoding="utf-8"))
        for name in report_names
    }
    for data_name, report_name in zip(data_names, report_names):
        if _reported_annotation_sha(reports[report_name]) != sha256_file(annotation_dir / data_name):
            raise RuntimeError(f"Annotation report hash mismatch: {report_name}")

    input_paths = {
        "retrieval_pool.jsonl": output / "retrieval_pool.jsonl",
        "active_frontier_context.jsonl": output / "active_frontier_context.jsonl",
        "annotator_a.template.jsonl": annotation_dir / "annotator_a.template.jsonl",
        "annotator_b.template.jsonl": annotation_dir / "annotator_b.template.jsonl",
        "adjudicator.template.jsonl": annotation_dir / "adjudicator.template.jsonl",
        "queries.jsonl": benchmark_root / "queries.jsonl",
        "nuggets.jsonl": benchmark_root / "nuggets.jsonl",
        "judgments.jsonl": benchmark_root / "judgments.jsonl",
        "evidence_units.jsonl": evidence_root / "evidence_units.jsonl",
        "source_spans.jsonl": evidence_root / "source_spans.jsonl",
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    pool_contract_sha256 = _json_digest(manifest)
    agreement = reports["agreement_report.json"]
    agreement["pool_contract_sha256"] = pool_contract_sha256
    agreement["input_hashes"] = input_hashes
    write_json(annotation_dir / "agreement_report.json", agreement, paths)

    annotation_hashes = {
        name: sha256_file(annotation_dir / name) for name in data_names + report_names
    }
    package = {
        "id": stable_id(pool_contract_sha256, canonical_json(annotation_hashes), length=40),
        "schema_version": 1,
        "status": "agent_provisional_human_review_pending",
        "pool_contract_sha256": pool_contract_sha256,
        "input_hashes": input_hashes,
        "annotation_hashes": annotation_hashes,
        "human_gold": False,
        "human_review_complete": False,
        "pooling_stable": False,
        "seed_frozen": False,
    }
    package_path = annotation_dir / "manifest.json"
    write_json(package_path, package, paths)
    manifest["pool_contract_sha256"] = pool_contract_sha256
    manifest["annotation_package_sha256"] = sha256_file(package_path)
    manifest["status"] = "provisional_depth20_agent_adjudicated_human_review_pending"
    manifest["output_hashes"].update(
        {f"annotations/{name}": digest for name, digest in annotation_hashes.items()}
    )
    manifest["output_hashes"]["annotations/manifest.json"] = sha256_file(package_path)
    return manifest


def _run_provenance(system: str) -> str:
    return {
        "E0-BM25": "lexical_candidate",
        "E1-dense-exact": "semantic_candidate",
        "E2-hybrid-rrf": "hybrid_candidate",
        "E3-rerank": "rerank_candidate",
    }[system]


def _application_family(source_path: str) -> str:
    parts = Path(source_path).as_posix().split("/")
    try:
        return parts[parts.index("applications") + 1]
    except (ValueError, IndexError):
        return ""


def _is_wrong_version_candidate(query: dict[str, Any], known_grade: int | None) -> bool:
    return query["no_answer_reason"] == "wrong_version" and known_grade is not None and known_grade <= 1


def _true_sibling_headings(
    unit: dict[str, Any],
    heading_by_section: dict[tuple[str, str], dict[str, Any]],
    headings_by_parent: dict[tuple[str, str | None], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    heading = heading_by_section.get((unit["source_document_id"], unit["section_id"]))
    if heading is None:
        return []
    siblings = headings_by_parent[(heading["source_document_id"], heading["parent_section_id"])]
    index = next(index for index, candidate in enumerate(siblings) if candidate["id"] == heading["id"])
    return siblings[max(0, index - 1) : index] + siblings[index + 1 : index + 2]


def _write_future_frontiers(
    output: Path,
    pool_manifest: dict[str, Any],
    pool_records: list[dict[str, Any]],
    chunks_by_config: dict[str, dict[str, dict[str, Any]]],
    matrix: dict[str, Any],
    paths: LabPaths,
) -> None:
    approval = require_approval("p3b-depth20-agent-annotations", paths)
    pool_manifest_path = output / "manifest.json"
    if approval["pool_manifest_sha256"] != sha256_file(pool_manifest_path):
        raise RuntimeError("Depth20 approval does not bind the current pool manifest")
    for depth in DEPTHS[1:]:
        contexts = []
        templates = []
        for item in pool_records:
            if item["first_pool_depth"] != depth:
                continue
            chunk = chunks_by_config[item["chunk_config_id"]][item["chunk_id"]]
            contexts.append(
                {
                    "id": item["id"],
                    "query_id": item["query_id"],
                    "chunk_config_hash": item["chunk_config_hash"],
                    "chunk_id": item["chunk_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "source_uri": chunk["source_uri"],
                    "span_fragments": chunk["span_fragments"],
                }
            )
            templates.append(
                {
                    "id": item["id"],
                    "query_id": item["query_id"],
                    "context_id": item["id"],
                    "chunk_config_hash": item["chunk_config_hash"],
                    "chunk_id": item["chunk_id"],
                    "annotator_id": None,
                    "retrieval_grade": None,
                    "selected_source_span_ids": [],
                    "selected_nugget_ids": [],
                    "rationale": "",
                    "annotation_status": "pending",
                    "adjudication_status": "pending_dual_annotation",
                }
            )
        root = output / "frontiers" / f"depth{depth}"
        write_jsonl(root / "context.jsonl", contexts, paths=paths)
        for name in ("annotator_a", "annotator_b", "adjudicator"):
            write_jsonl(root / "annotations" / f"{name}.template.jsonl", templates, paths=paths)
        output_hashes = {
            "context.jsonl": sha256_file(root / "context.jsonl"),
            **{
                f"annotations/{name}.template.jsonl": sha256_file(
                    root / "annotations" / f"{name}.template.jsonl"
                )
                for name in ("annotator_a", "annotator_b", "adjudicator")
            },
        }
        frontier_manifest = {
                "id": stable_id(pool_manifest["id"], str(depth), canonical_json(output_hashes), length=40),
                "schema_version": 1,
                "status": "pending_dual_agent_annotation",
                "depth": depth,
                "previous_depth": depth - 10,
                "item_count": len(contexts),
                "pool_id": pool_manifest["id"],
                "pool_manifest_sha256": sha256_file(pool_manifest_path),
                "retrieval_pool_sha256": sha256_file(output / "retrieval_pool.jsonl"),
                "output_hashes": output_hashes,
                "human_gold": False,
                "pooling_stable": False,
                "seed_frozen": False,
            }
        write_json(root / "manifest.json", frontier_manifest, paths)
        _seal_future_frontier(output, root, frontier_manifest, paths)
    _write_stability_report(output, pool_records, matrix, paths)


def _seal_future_frontier(
    pool_root: Path,
    frontier_root: Path,
    frontier_manifest: dict[str, Any],
    paths: LabPaths,
) -> None:
    annotation_dir = frontier_root / "annotations"
    data_names = ("annotator_a.jsonl", "annotator_b.jsonl", "adjudicator.jsonl")
    report_names = ("annotator_a.report.json", "annotator_b.report.json", "agreement_report.json")
    completed = [annotation_dir / name for name in data_names + report_names]
    if not any(path.is_file() for path in completed):
        return
    if not all(path.is_file() for path in completed):
        missing = [path.name for path in completed if not path.is_file()]
        raise RuntimeError(f"Incomplete depth{frontier_manifest['depth']} annotation package: {', '.join(missing)}")

    contexts = {row["id"]: row for row in load_jsonl(frontier_root / "context.jsonl")}
    context_ids = sorted(contexts)
    nuggets = {
        row["id"]: row
        for row in load_jsonl(paths.root / "benchmarks" / "seed50" / "provisional" / "nuggets.jsonl")
    }
    for name in data_names:
        rows = load_jsonl(annotation_dir / name)
        if [row["id"] for row in rows] != context_ids or len(rows) != len(contexts):
            raise RuntimeError(f"Annotation coverage/order mismatch: depth{frontier_manifest['depth']}/{name}")
        for row in rows:
            context = contexts[row["id"]]
            visible_spans = {
                span_id
                for fragment in context["span_fragments"]
                for span_id in fragment["source_span_ids"]
            }
            grade = row["retrieval_grade"]
            if grade == 0 and (row["selected_source_span_ids"] or row["selected_nugget_ids"]):
                raise RuntimeError(f"Grade 0 cannot bind evidence: {row['id']}")
            if grade == 1 and row["selected_nugget_ids"]:
                raise RuntimeError(f"Grade 1 cannot bind nuggets: {row['id']}")
            if grade >= 2 and (not row["selected_source_span_ids"] or not row["selected_nugget_ids"]):
                raise RuntimeError(f"Positive annotation lacks evidence binding: {row['id']}")
            if not set(row["selected_source_span_ids"]) <= visible_spans:
                raise RuntimeError(f"Annotation selects a non-visible source span: {row['id']}")
            if any(nuggets[nugget_id]["query_id"] != row["query_id"] for nugget_id in row["selected_nugget_ids"]):
                raise RuntimeError(f"Annotation selects a cross-query nugget: {row['id']}")

    reports = {
        name: json.loads((annotation_dir / name).read_text(encoding="utf-8"))
        for name in report_names
    }
    for data_name, report_name in zip(data_names, report_names):
        reported_sha = _reported_annotation_sha(reports[report_name])
        if reported_sha != sha256_file(annotation_dir / data_name):
            raise RuntimeError(f"Annotation report hash mismatch: depth{frontier_manifest['depth']}/{report_name}")

    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    input_paths = {
        "pool_manifest.json": pool_root / "manifest.json",
        "retrieval_pool.jsonl": pool_root / "retrieval_pool.jsonl",
        "context.jsonl": frontier_root / "context.jsonl",
        **{
            f"{name}.template.jsonl": annotation_dir / f"{name}.template.jsonl"
            for name in ("annotator_a", "annotator_b", "adjudicator")
        },
        "queries.jsonl": benchmark_root / "queries.jsonl",
        "nuggets.jsonl": benchmark_root / "nuggets.jsonl",
        "judgments.jsonl": benchmark_root / "judgments.jsonl",
        "evidence_units.jsonl": evidence_root / "evidence_units.jsonl",
        "source_spans.jsonl": evidence_root / "source_spans.jsonl",
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    frontier_contract_sha256 = _json_digest(frontier_manifest)
    agreement = reports["agreement_report.json"]
    agreement["frontier_contract_sha256"] = frontier_contract_sha256
    agreement["input_hashes"] = input_hashes
    write_json(annotation_dir / "agreement_report.json", agreement, paths)
    annotation_hashes = {
        name: sha256_file(annotation_dir / name) for name in data_names + report_names
    }
    package = {
        "id": stable_id(frontier_contract_sha256, canonical_json(annotation_hashes), length=40),
        "schema_version": 1,
        "status": "agent_provisional_human_review_pending",
        "depth": frontier_manifest["depth"],
        "frontier_contract_sha256": frontier_contract_sha256,
        "input_hashes": input_hashes,
        "annotation_hashes": annotation_hashes,
        "human_gold": False,
        "human_review_complete": False,
        "pooling_stable": False,
        "seed_frozen": False,
    }
    package_path = annotation_dir / "manifest.json"
    write_json(package_path, package, paths)
    frontier_manifest["status"] = "agent_adjudicated_human_review_pending"
    frontier_manifest["frontier_contract_sha256"] = frontier_contract_sha256
    frontier_manifest["annotation_package_sha256"] = sha256_file(package_path)
    frontier_manifest["output_hashes"].update(
        {f"annotations/{name}": digest for name, digest in annotation_hashes.items()}
    )
    frontier_manifest["output_hashes"]["annotations/manifest.json"] = sha256_file(package_path)
    write_json(frontier_root / "manifest.json", frontier_manifest, paths)


def _ndcg10(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[dict[str, Any]]],
) -> float:
    run = {
        query_id: {row["chunk_id"]: row["score"] for row in ranking}
        for query_id, ranking in rankings.items()
        if query_id in qrels
    }
    measure = ir_measures.nDCG @ 10
    return float(ir_measures.calc_aggregate([measure], qrels, run)[measure])


def _write_stability_report(
    pool_root: Path,
    pool_records: list[dict[str, Any]],
    matrix: dict[str, Any],
    paths: LabPaths,
) -> None:
    annotation_sources = [(20, pool_root / "annotations" / "adjudicator.jsonl")]
    package_paths = [pool_root / "annotations" / "manifest.json"]
    for depth in DEPTHS[1:]:
        frontier_root = pool_root / "frontiers" / f"depth{depth}"
        manifest_path = frontier_root / "manifest.json"
        if not manifest_path.is_file():
            break
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "agent_adjudicated_human_review_pending":
            break
        annotation_sources.append((depth, frontier_root / "annotations" / "adjudicator.jsonl"))
        package_paths.append(frontier_root / "annotations" / "manifest.json")

    candidate_by_id = {row["id"]: row for row in pool_records}
    answerable = {
        row["id"]
        for row in load_jsonl(paths.root / "benchmarks" / "seed50" / "provisional" / "queries.jsonl")
        if row["answerability"] == "answerable"
    }
    runs_root = paths.root / "artifacts" / "runs" / "seed50-provisional"
    rankings = {
        entry["run_id"]: _read_approved_trec_run(entry, runs_root / entry["run_id"])
        for entry in matrix["runs"]
    }
    rows: list[dict[str, Any]] = []
    cumulative_annotations: list[dict[str, Any]] = []
    for depth, annotation_path in annotation_sources:
        new_annotations = load_jsonl(annotation_path)
        cumulative_annotations.extend(new_annotations)
        new_positive = sum(row["retrieval_grade"] >= 2 for row in new_annotations)
        qrels_by_config: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for annotation in cumulative_annotations:
            candidate = candidate_by_id[annotation["id"]]
            if annotation["query_id"] in answerable:
                qrels_by_config[candidate["chunk_config_id"]][annotation["query_id"]][
                    candidate["chunk_id"]
                ] = annotation["retrieval_grade"]

        leaveouts = []
        full_orders: dict[str, list[str]] = {}
        for config_id in sorted(qrels_by_config):
            entries = [entry for entry in matrix["runs"] if entry["chunk_config_id"] == config_id]
            full_scores = {
                entry["system"]: _ndcg10(dict(qrels_by_config[config_id]), rankings[entry["run_id"]])
                for entry in entries
            }
            full_order = _system_order(full_scores)
            full_orders[config_id] = full_order
            for heldout in entries:
                exclusive = {
                    (candidate["query_id"], candidate["chunk_id"])
                    for candidate in pool_records
                    if candidate["chunk_config_id"] == config_id
                    and candidate["first_pool_depth"] <= depth
                    and {
                        hit["run_id"]
                        for hit in candidate["run_hits"]
                        if hit["rank"] <= depth
                    }
                    == {heldout["run_id"]}
                }
                reduced = {
                    query_id: {
                        chunk_id: grade
                        for chunk_id, grade in judgments.items()
                        if (query_id, chunk_id) not in exclusive
                    }
                    for query_id, judgments in qrels_by_config[config_id].items()
                }
                reduced_scores = {
                    entry["system"]: _ndcg10(reduced, rankings[entry["run_id"]])
                    for entry in entries
                }
                reduced_order = _system_order(reduced_scores)
                leaveouts.append(
                    {
                        "chunk_config_id": config_id,
                        "held_out_run_id": heldout["run_id"],
                        "held_out_system": heldout["system"],
                        "exclusive_judgments_removed": len(exclusive),
                        "kendall_tau": _order_tau(full_order, reduced_order),
                        "system_order": reduced_order,
                    }
                )
        minimum_tau = min(row["kendall_tau"] for row in leaveouts)
        rows.append(
            {
                "depth": depth,
                "new_judged_items": len(new_annotations),
                "new_grade_2_or_3_items": new_positive,
                "new_grade_2_or_3_yield": new_positive / len(new_annotations),
                "cumulative_judged_items": len(cumulative_annotations),
                "leave_one_run_out_min_kendall_tau": minimum_tau,
                "full_system_order_by_chunk_config": full_orders,
                "leave_one_run_out": leaveouts,
            }
        )

    agent_stable = _agent_pool_is_stable(rows)
    report = {
        "schema_version": 1,
        "status": "agent_diagnostic_stable_human_review_pending"
        if agent_stable
        else "pending_additional_adjudicated_depths",
        "depths_completed": [row["depth"] for row in rows],
        "stability_rule": {
            "consecutive_added_depths": 2,
            "new_grade_2_or_3_yield_below": 0.01,
            "leave_one_run_out_kendall_tau_at_least": 0.95,
        },
        "depth_results": rows,
        "agent_diagnostic_pooling_stable": agent_stable,
        "pooling_stable": False,
        "human_review_complete": False,
        "seed_frozen": False,
    }
    stability_root = pool_root / "stability"
    write_json(stability_root / "report.json", report, paths)
    adjudicated_outputs = _write_adjudicated_review_outputs(
        pool_root, annotation_sources, candidate_by_id, paths
    )
    package_hashes = {
        str(path.relative_to(pool_root)).replace("\\", "/"): sha256_file(path)
        for path in package_paths
    }
    write_json(
        stability_root / "manifest.json",
        {
            "id": stable_id(matrix["id"], canonical_json(package_hashes), canonical_json(report), length=40),
            "schema_version": 1,
            "status": report["status"],
            "matrix_id": matrix["id"],
            "annotation_package_hashes": package_hashes,
            "adjudicated_output_hashes": adjudicated_outputs,
            "report_sha256": sha256_file(stability_root / "report.json"),
            "agent_diagnostic_pooling_stable": agent_stable,
            "pooling_stable": False,
            "human_review_complete": False,
            "seed_frozen": False,
        },
        paths,
    )


def build_pool(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    approval = require_approval("p3b-matrix", paths)
    verify_source(paths)
    matrix_path = paths.root / "artifacts" / "matrices" / "seed50-provisional" / "manifest.json"
    if sha256_file(matrix_path) != approval["matrix_manifest_sha256"]:
        raise RuntimeError("Approved provisional matrix manifest hash mismatch")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["id"] != approval["matrix_id"] or matrix["status"] != "provisional_unpooled":
        raise RuntimeError("Approved provisional matrix identity/status mismatch")

    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    queries = load_jsonl(benchmark_root / "queries.jsonl")
    query_by_id = {query["id"]: query for query in queries}
    judgments = load_jsonl(benchmark_root / "judgments.jsonl")
    judgment_by_key = {(row["query_id"], row["source_span_id"]): row for row in judgments}

    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = load_jsonl(evidence_root / "evidence_units.jsonl")
    spans = {span["id"]: span for span in load_jsonl(evidence_root / "source_spans.jsonl")}
    span_units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    heading_by_section: dict[tuple[str, str], dict[str, Any]] = {}
    headings_by_parent: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        for span_id in unit["source_span_ids"]:
            span_units[span_id].append(unit)
        if unit["node_type"] == "heading":
            heading_by_section[(unit["source_document_id"], unit["section_id"])] = unit
            headings_by_parent[(unit["source_document_id"], unit["parent_section_id"])].append(unit)
    for headings in headings_by_parent.values():
        headings.sort(key=lambda unit: (unit["ordinal"], unit["id"]))

    positive_families: dict[str, set[str]] = defaultdict(set)
    for judgment in judgments:
        if judgment["grade"] >= 2:
            positive_families[judgment["query_id"]].add(
                _application_family(spans[judgment["source_span_id"]]["span_refs"][0]["source_path"])
            )

    chunks_by_config: dict[str, dict[str, dict[str, Any]]] = {}
    config_hashes: dict[str, str] = {}
    qrels_by_config: dict[str, dict[str, dict[str, int]]] = {}
    pool: dict[tuple[str, str, str], dict[str, Any]] = {}
    trec_hits = 0
    for run_entry in matrix["runs"]:
        config_id = run_entry["chunk_config_id"]
        if config_id not in chunks_by_config:
            chunk_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / config_id
            config_manifest = _verify_manifest_outputs(chunk_root)
            config_hashes[config_id] = config_manifest["chunk_config_hash"]
            chunks_by_config[config_id] = {
                chunk["id"]: chunk for chunk in load_jsonl(chunk_root / "chunks.jsonl")
            }
            qrels_by_config[config_id] = _read_qrels(
                benchmark_root / "derived" / config_id / "qrels.seed.trec"
            )
        rankings = _read_approved_trec_run(
            run_entry,
            paths.root
            / "artifacts"
            / "runs"
            / "seed50-provisional"
            / run_entry["run_id"],
        )
        for query_id, ranking in rankings.items():
            for hit in ranking[:50]:
                trec_hits += 1
                chunk = chunks_by_config[config_id][hit["chunk_id"]]
                key = query_id, config_hashes[config_id], chunk["id"]
                depth = next(value for value in DEPTHS if hit["rank"] <= value)
                item = pool.setdefault(
                    key,
                    {
                        "id": "retrieval-pool-" + stable_id(*key),
                        "query_id": query_id,
                        "query_text": query_by_id[query_id]["text"],
                        "answerability": query_by_id[query_id]["answerability"],
                        "chunk_config_id": config_id,
                        "chunk_config_hash": config_hashes[config_id],
                        "chunk_id": chunk["id"],
                        "chunk_text_sha256": chunk["text_sha256"],
                        "title": chunk["title"],
                        "source_uri": chunk["source_uri"],
                        "source_document_id": chunk["source_document_id"],
                        "first_pool_depth": depth,
                        "minimum_rank": hit["rank"],
                        "provenance": set(),
                        "run_hits": [],
                        "known_provisional_grade": qrels_by_config[config_id]
                        .get(query_id, {})
                        .get(chunk["id"]),
                    },
                )
                item["first_pool_depth"] = min(item["first_pool_depth"], depth)
                item["minimum_rank"] = min(item["minimum_rank"], hit["rank"])
                item["provenance"].add(_run_provenance(run_entry["system"]))
                source_path = spans[chunk["span_fragments"][0]["source_span_ids"][0]]["span_refs"][0][
                    "source_path"
                ]
                family = _application_family(source_path)
                if family and positive_families[query_id] and family not in positive_families[query_id]:
                    item["provenance"].add("wrong_module_candidate")
                item["run_hits"].append(
                    {
                        "run_id": run_entry["run_id"],
                        "system": run_entry["system"],
                        "rank": hit["rank"],
                        "score": hit["score"],
                    }
                )

    pool_records: list[dict[str, Any]] = []
    frontier_context: list[dict[str, Any]] = []
    for item in pool.values():
        item["provenance"] = sorted(item["provenance"])
        item["run_hits"] = sorted(item["run_hits"], key=lambda hit: (hit["system"], hit["rank"]))
        pool_records.append(item)
        if item["first_pool_depth"] == 20:
            chunk = chunks_by_config[item["chunk_config_id"]][item["chunk_id"]]
            frontier_context.append(
                {
                    "id": item["id"],
                    "query_id": item["query_id"],
                    "chunk_config_hash": item["chunk_config_hash"],
                    "chunk_id": item["chunk_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "source_uri": chunk["source_uri"],
                    "span_fragments": chunk["span_fragments"],
                }
            )
    pool_records.sort(
        key=lambda item: (
            item["query_id"],
            item["first_pool_depth"],
            item["chunk_config_id"],
            item["chunk_id"],
        )
    )
    frontier_context.sort(key=lambda item: item["id"])

    expert: dict[tuple[str, str], dict[str, Any]] = {}

    def add_expert(query_id: str, span_id: str, unit: dict[str, Any], provenance: str) -> None:
        key = query_id, span_id
        judgment = judgment_by_key.get(key)
        row = expert.setdefault(
            key,
            {
                "id": "expert-" + stable_id(*key),
                "query_id": query_id,
                "query_text": query_by_id[query_id]["text"],
                "source_span_id": span_id,
                "evidence_unit_id": unit["id"],
                "node_type": unit["node_type"],
                "source_uri": unit["source_uri"],
                "heading_path": unit["heading_path"],
                "text": unit["rendered_text"],
                "provenance": set(),
                "known_provisional_grade": judgment["grade"] if judgment else None,
            },
        )
        row["provenance"].add(provenance)

    for judgment in judgments:
        for unit in span_units[judgment["source_span_id"]]:
            add_expert(
                judgment["query_id"],
                judgment["source_span_id"],
                unit,
                "seeded_positive" if judgment["grade"] >= 2 else "seeded_agent_candidate",
            )
            if judgment["grade"] >= 2:
                for sibling in _true_sibling_headings(unit, heading_by_section, headings_by_parent):
                    for span_id in sibling["source_span_ids"]:
                        add_expert(judgment["query_id"], span_id, sibling, "sibling_heading")
    expert_records = sorted(expert.values(), key=lambda row: (row["query_id"], row["source_span_id"]))
    for row in expert_records:
        if _is_wrong_version_candidate(
            query_by_id[row["query_id"]], row["known_provisional_grade"]
        ):
            row["provenance"].add("wrong_version")
        row["provenance"] = sorted(row["provenance"])

    templates = []
    context_by_id = {context["id"]: context for context in frontier_context}
    for item in pool_records:
        if item["first_pool_depth"] != 20:
            continue
        context = context_by_id[item["id"]]
        known = item["known_provisional_grade"]
        selected_spans = []
        selected_nuggets = []
        if known is not None and known >= 1:
            chunk_spans = {
                span_id for fragment in context["span_fragments"] for span_id in fragment["source_span_ids"]
            }
            selected_spans = sorted(
                span_id
                for span_id in chunk_spans
                if (item["query_id"], span_id) in judgment_by_key
                and judgment_by_key[(item["query_id"], span_id)]["grade"] >= 1
            )
            selected_nuggets = sorted(
                {
                    nugget_id
                    for span_id in selected_spans
                    for nugget_id in judgment_by_key[(item["query_id"], span_id)]["nugget_ids"]
                }
            )
        templates.append(
            {
                "id": item["id"],
                "query_id": item["query_id"],
                "context_id": context["id"],
                "chunk_config_hash": item["chunk_config_hash"],
                "chunk_id": item["chunk_id"],
                "annotator_id": None,
                "retrieval_grade": known,
                "selected_source_span_ids": selected_spans,
                "selected_nugget_ids": selected_nuggets,
                "rationale": "" if known is None else "prepopulated from provisional canonical judgment",
                "annotation_status": "pending" if known is None else "prepopulated_provisional",
                "adjudication_status": "pending_dual_annotation",
            }
        )

    depth_report = []
    for depth in DEPTHS:
        included = [row for row in pool_records if row["first_pool_depth"] <= depth]
        new = [row for row in pool_records if row["first_pool_depth"] == depth]
        judged = [row for row in included if row["known_provisional_grade"] is not None]
        depth_report.append(
            {
                "depth": depth,
                "retrieval_pool_items": len(included),
                "new_retrieval_pool_items": len(new),
                "known_provisional_items": len(judged),
                "unjudged_items": len(included) - len(judged),
                "active_annotation_frontier": depth == 20,
                "relevance_yield": None,
                "leave_one_run_out_kendall_tau": None,
                "stability_status": "PENDING_COMPLETE_DUAL_ANNOTATION_AND_ADJUDICATION",
            }
        )

    per_query = [
        {
            "query_id": query["id"],
            "answerability": query["answerability"],
            "depth20_items": sum(
                row["query_id"] == query["id"] and row["first_pool_depth"] <= 20
                for row in pool_records
            ),
            "depth50_items": sum(row["query_id"] == query["id"] for row in pool_records),
            "hard_negative_requirement_met": False,
            "no_answer_five_judgments_met": False,
        }
        for query in queries
    ]

    output = paths.root / "benchmarks" / "seed50" / "pooling" / "provisional"
    for stale in ("candidates.jsonl", "annotation_template.jsonl"):
        stale_path = paths.require_write_path(output / stale)
        if stale_path.is_file():
            stale_path.unlink()
    write_jsonl(output / "retrieval_pool.jsonl", pool_records, paths=paths)
    write_jsonl(output / "active_frontier_context.jsonl", frontier_context, paths=paths)
    write_jsonl(output / "expert_candidates.jsonl", expert_records, paths=paths)
    for name in ("annotator_a", "annotator_b", "adjudicator"):
        write_jsonl(output / "annotations" / f"{name}.template.jsonl", templates, paths=paths)
    write_json(output / "depth_report.json", depth_report, paths)
    write_jsonl(output / "per_query_report.jsonl", per_query, paths=paths)
    output_names = (
        "retrieval_pool.jsonl",
        "active_frontier_context.jsonl",
        "expert_candidates.jsonl",
        "annotations/annotator_a.template.jsonl",
        "annotations/annotator_b.template.jsonl",
        "annotations/adjudicator.template.jsonl",
        "depth_report.json",
        "per_query_report.jsonl",
    )
    outputs = {name: sha256_file(output / name) for name in output_names}
    manifest = {
        "id": stable_id(matrix["id"], canonical_json(outputs), length=40),
        "schema_version": 1,
        "status": "provisional_depth20_pending_dual_annotation_and_human_review",
        "matrix_id": matrix["id"],
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "depths": list(DEPTHS),
        "trec_hits_mapped": trec_hits,
        "retrieval_pool_items_depth20": depth_report[0]["retrieval_pool_items"],
        "retrieval_pool_items_depth50": depth_report[-1]["retrieval_pool_items"],
        "active_frontier_items": len(frontier_context),
        "expert_candidate_count": len(expert_records),
        "wrong_version_candidate_count": sum(
            "wrong_version" in row["provenance"] for row in expert_records
        ),
        "pooling_stable": False,
        "seed_frozen": False,
        "human_review_complete": False,
        "sota_claims_allowed": False,
        "output_hashes": outputs,
    }
    manifest = _seal_completed_annotations(output, manifest, benchmark_root, evidence_root, paths)
    write_json(output / "manifest.json", manifest, paths)
    approval_path = paths.root / "reviews" / "p3b-depth20-agent-annotations" / "approval.json"
    if approval_path.is_file():
        _write_future_frontiers(output, manifest, pool_records, chunks_by_config, matrix, paths)
    return manifest
