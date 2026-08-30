from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baselines import _read_approved_trec_run
from .benchmark import SNAPSHOT_ID, load_jsonl
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .smoke import _read_qrels, _verify_manifest_outputs
from .verify import verify_source


DEPTHS = (20, 30, 40, 50)


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
    write_json(output / "manifest.json", manifest, paths)
    return manifest
