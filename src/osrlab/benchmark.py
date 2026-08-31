from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from beir.datasets.data_loader import GenericDataLoader

from .contract import validate_record
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths


SNAPSHOT_ID = "32a8b8d77833f22b4bc74ed4ea78b6a82b5338fd"
INTENT_COUNTS = {
    "exact_identifier": 6,
    "prerequisite": 8,
    "ordered_procedure": 8,
    "action_semantics": 6,
    "diagnosis": 6,
    "cross_module": 6,
    "comparison": 5,
    "version_applicability": 5,
}
TOPOLOGY_COUNTS = {"single_span": 26, "same_page_multi_span": 10, "cross_page": 4}
NO_ANSWER_REASONS = {
    "absent_from_corpus",
    "requires_live_instance",
    "wrong_version",
    "enterprise_or_excluded",
    "ambiguous_requires_clarification",
    "out_of_scope",
}
SMOKE_SLICES = {"lexical", "semantic", "multi_evidence", "no_answer"}
TOPIC_CONTRACT_CORRECTIONS = {"S007", "S012", "S026", "S041", "S042"}


def derive_retrieval_grade(
    *,
    topic_relevance: bool,
    required_nugget_hits: set[str] | list[str],
    required_nugget_ids: set[str] | list[str],
) -> int:
    """Derive the only allowed 0-3 grade from topic and atomic-nugget coverage."""
    hits, required = set(required_nugget_hits), set(required_nugget_ids)
    if not hits <= required:
        raise RuntimeError("Required nugget hits contain an unknown nugget")
    if not topic_relevance:
        if hits:
            raise RuntimeError("Irrelevant evidence cannot hit a required nugget")
        return 0
    if not hits:
        return 1
    return 3 if required and hits == required else 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def topic_briefs_path(paths: LabPaths | None = None) -> Path:
    paths = paths or LabPaths.discover()
    return paths.root / "benchmarks" / "seed50" / "topic_briefs.jsonl"


def validate_topic_briefs(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    briefs = load_jsonl(topic_briefs_path(paths))
    for brief in briefs:
        validate_record("TopicBrief", brief, paths)
    expected_ids = [f"S{index:03d}" for index in range(1, 51)]
    if [brief["brief_id"] for brief in briefs] != expected_ids:
        raise RuntimeError("Seed50 topic briefs must be ordered exactly S001..S050")
    normalized = [re.sub(r"\s+", " ", brief["query_text"].strip().casefold()) for brief in briefs]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("Seed50 contains duplicate normalized queries")
    intents = Counter(brief["intent"] for brief in briefs)
    answerability = Counter(brief["answerability"] for brief in briefs)
    topology = Counter(brief["evidence_topology"] for brief in briefs if brief["answerability"] == "answerable")
    reasons = Counter(brief["no_answer_reason"] for brief in briefs if brief["answerability"] == "no_answer")
    smoke = [brief for brief in briefs if brief["smoke"]]
    if dict(intents) != INTENT_COUNTS:
        raise RuntimeError(f"Seed50 intent distribution mismatch: {intents}")
    if answerability != Counter({"answerable": 40, "no_answer": 10}):
        raise RuntimeError(f"Seed50 answerability distribution mismatch: {answerability}")
    if dict(topology) != TOPOLOGY_COUNTS:
        raise RuntimeError(f"Seed50 topology distribution mismatch: {topology}")
    if set(reasons) != NO_ANSWER_REASONS:
        raise RuntimeError(f"Seed50 no-answer reason coverage mismatch: {reasons}")
    if len(smoke) != 12 or set(brief["smoke_slice"] for brief in smoke) != SMOKE_SLICES:
        raise RuntimeError("Smoke12 must contain 12 topics and all four smoke slices")
    source_documents = {
        record["source_path"]
        for record in load_jsonl(paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence" / "source_documents.jsonl")
    }
    for brief in briefs:
        answerable = brief["answerability"] == "answerable"
        if answerable != (brief["no_answer_reason"] is None and brief["evidence_topology"] is not None):
            raise RuntimeError(f"Answerability sidecar mismatch: {brief['brief_id']}")
        if answerable and not brief["source_hint_paths"]:
            raise RuntimeError(f"Answerable brief lacks source hints: {brief['brief_id']}")
        if any(path not in source_documents or not path.startswith("applications/") for path in brief["source_hint_paths"]):
            raise RuntimeError(f"Brief contains an invalid source hint: {brief['brief_id']}")
        if brief["smoke"] != (brief["smoke_slice"] is not None):
            raise RuntimeError(f"Smoke sidecar mismatch: {brief['brief_id']}")
    return {
        "topics": len(briefs),
        "intent_counts": dict(intents),
        "answerability_counts": dict(answerability),
        "topology_counts": dict(topology),
        "no_answer_reason_counts": dict(reasons),
        "smoke_topics": len(smoke),
    }


def _evidence_maps(paths: LabPaths) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = load_jsonl(evidence_root / "evidence_units.jsonl")
    spans = load_jsonl(evidence_root / "source_spans.jsonl")
    unit_by_span = {span_id: unit for unit in units for span_id in unit["source_span_ids"]}
    return unit_by_span, {span["id"]: span for span in spans}


def validate_annotation_file(
    path: Path,
    annotator_id: str,
    paths: LabPaths | None = None,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    validate_topic_briefs(paths)
    briefs = {brief["brief_id"]: brief for brief in load_jsonl(topic_briefs_path(paths))}
    submissions = load_jsonl(path)
    if len(submissions) != 50 or {item["brief_id"] for item in submissions} != set(briefs):
        raise RuntimeError(f"{annotator_id} must submit exactly one record for every Seed50 brief")
    unit_by_span, span_by_id = _evidence_maps(paths)
    grades: Counter[int] = Counter()
    missing_seeded_negatives: list[str] = []
    for submission in submissions:
        validate_record("AnnotationSubmission", submission, paths)
        brief_id = submission["brief_id"]
        if submission["annotator_id"] != annotator_id:
            raise RuntimeError(f"Annotator provenance mismatch: {brief_id}")
        local_nuggets = {nugget["local_id"]: nugget for nugget in submission["nuggets"]}
        if len(local_nuggets) != len(submission["nuggets"]):
            raise RuntimeError(f"Duplicate local nugget IDs: {brief_id}")
        judgments = submission["judgments"]
        judged_spans = [judgment["source_span_id"] for judgment in judgments]
        if len(judged_spans) != len(set(judged_spans)):
            raise RuntimeError(f"Duplicate judged source spans: {brief_id}")
        for judgment in judgments:
            span_id = judgment["source_span_id"]
            if span_id not in span_by_id or span_id not in unit_by_span:
                raise RuntimeError(f"Unknown or non-applications SourceSpan in {brief_id}: {span_id}")
            if not set(judgment["local_nugget_ids"]) <= set(local_nuggets):
                raise RuntimeError(f"Judgment references an unknown local nugget: {brief_id}")
            grades[judgment["grade"]] += 1
        nugget_spans = {span_id for nugget in submission["nuggets"] for span_id in nugget["source_span_ids"]}
        if not nugget_spans <= set(judged_spans) or any(span_id not in unit_by_span for span_id in nugget_spans):
            raise RuntimeError(f"Nugget evidence is not fully judged applications evidence: {brief_id}")
        positive = [judgment for judgment in judgments if judgment["grade"] >= 2]
        negative = [judgment for judgment in judgments if judgment["grade"] <= 1]
        if submission["answerability"] == "answerable":
            if submission["no_answer_reason"] is not None or submission["evidence_topology"] is None:
                raise RuntimeError(f"Answerable annotation sidecars are invalid: {brief_id}")
            if not submission["nuggets"] or not any(nugget["required"] for nugget in submission["nuggets"]) or not positive:
                raise RuntimeError(f"Answerable annotation lacks required positive evidence: {brief_id}")
            positive_docs = {unit_by_span[item["source_span_id"]]["source_document_id"] for item in positive}
            if submission["evidence_topology"] == "single_span" and len(positive) != 1:
                raise RuntimeError(f"single_span requires exactly one positive SourceSpan: {brief_id}")
            if submission["evidence_topology"] == "same_page_multi_span" and (len(positive) < 2 or len(positive_docs) != 1):
                raise RuntimeError(f"same_page_multi_span topology mismatch: {brief_id}")
            if submission["evidence_topology"] == "cross_page" and (len(positive) < 2 or len(positive_docs) < 2):
                raise RuntimeError(f"cross_page topology mismatch: {brief_id}")
        else:
            if submission["no_answer_reason"] not in NO_ANSWER_REASONS or submission["evidence_topology"] is not None:
                raise RuntimeError(f"No-answer sidecars are invalid: {brief_id}")
            if submission["nuggets"] or positive:
                raise RuntimeError(f"No-answer annotation contains positive evidence: {brief_id}")
        if not negative:
            missing_seeded_negatives.append(brief_id)
    return {
        "annotator_id": annotator_id,
        "topics": len(submissions),
        "grade_counts": {str(grade): grades[grade] for grade in range(4)},
        "topics_without_seeded_negative": missing_seeded_negatives,
    }


def _weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    size = 4
    observed = sum((left - right) ** 2 / 9 for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[left] * right_counts[right] * ((left - right) ** 2 / 9)
        for left in range(size)
        for right in range(size)
    ) / (len(pairs) ** 2)
    return 1.0 if expected == 0 and observed == 0 else (0.0 if expected == 0 else 1 - observed / expected)


def _agreement(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {item["brief_id"]: item for item in left}
    right_by_id = {item["brief_id"]: item for item in right}
    fields = ("intent", "answerability", "no_answer_reason", "evidence_topology", "source_fact_group")
    field_agreement = {
        field: sum(left_by_id[brief][field] == right_by_id[brief][field] for brief in left_by_id) / len(left_by_id)
        for field in fields
    }
    text_exact = sum(
        left_by_id[brief]["query_text"] == right_by_id[brief]["query_text"] for brief in left_by_id
    ) / len(left_by_id)
    jaccards: list[float] = []
    grade_pairs: list[tuple[int, int]] = []
    disagreement_ids: list[str] = []
    for brief in left_by_id:
        left_grades = {item["source_span_id"]: item["grade"] for item in left_by_id[brief]["judgments"]}
        right_grades = {item["source_span_id"]: item["grade"] for item in right_by_id[brief]["judgments"]}
        left_positive = {span for span, grade in left_grades.items() if grade >= 2}
        right_positive = {span for span, grade in right_grades.items() if grade >= 2}
        union = left_positive | right_positive
        jaccards.append(len(left_positive & right_positive) / len(union) if union else 1.0)
        common = set(left_grades) & set(right_grades)
        grade_pairs.extend((left_grades[span], right_grades[span]) for span in common)
        if (
            any(left_by_id[brief][field] != right_by_id[brief][field] for field in fields[:-1])
            or left_positive != right_positive
            or any(left_grades[span] != right_grades[span] for span in common)
        ):
            disagreement_ids.append(brief)
    return {
        "topic_count": len(left_by_id),
        "field_exact_agreement": field_agreement,
        "query_text_exact_agreement": text_exact,
        "mean_positive_span_jaccard": sum(jaccards) / len(jaccards),
        "common_judgment_count": len(grade_pairs),
        "quadratic_weighted_kappa_common_judgments": _weighted_kappa(grade_pairs),
        "substantive_disagreement_brief_ids": disagreement_ids,
    }


def _write_text(path: Path, text: str, paths: LabPaths) -> None:
    checked = paths.require_write_path(path)
    checked.parent.mkdir(parents=True, exist_ok=True)
    checked.write_text(text, encoding="utf-8", newline="\n")


def _project_qrels(
    paths: LabPaths,
    output_root: Path,
    queries: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    nuggets: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks"
    c1_chunks = load_jsonl(chunks_root / "C1-section-native" / "chunks.jsonl")
    unit_token_counts: dict[str, int] = {}
    for chunk in c1_chunks:
        for fragment in chunk["span_fragments"]:
            unit_id = fragment["evidence_unit_id"]
            unit_token_counts[unit_id] = max(unit_token_counts.get(unit_id, 0), fragment["unit_token_end"])
    answerable_ids = {query["id"] for query in queries if query["answerability"] == "answerable"}
    required_by_query: dict[str, set[str]] = defaultdict(set)
    for nugget in nuggets:
        if nugget["required"]:
            required_by_query[nugget["query_id"]].add(nugget["id"])
    judgment_by_span: dict[str, list[dict[str, Any]]] = {}
    for judgment in judgments:
        if judgment["query_id"] in answerable_ids:
            judgment_by_span.setdefault(judgment["source_span_id"], []).append(judgment)
    report: dict[str, Any] = {}
    query_file = output_root / "beir" / "queries.jsonl"
    for config_dir in sorted(path for path in chunks_root.glob("C*") if path.is_dir()):
        chunks = load_jsonl(config_dir / "chunks.jsonl")
        projected_state: dict[tuple[str, str], dict[str, Any]] = {}
        partial_positive_downgrades = 0
        for chunk in chunks:
            for fragment in chunk["span_fragments"]:
                full = (
                    fragment["unit_token_start"] == 0
                    and fragment["unit_token_end"] == unit_token_counts[fragment["evidence_unit_id"]]
                )
                for span_id in fragment["source_span_ids"]:
                    for judgment in judgment_by_span.get(span_id, []):
                        key = judgment["query_id"], chunk["id"]
                        state = projected_state.setdefault(
                            key, {"topic_relevance": False, "required_nugget_hits": set()}
                        )
                        state["topic_relevance"] |= judgment["topic_relevance"]
                        if judgment["required_nugget_hits"] and not full:
                            partial_positive_downgrades += 1
                        elif full:
                            state["required_nugget_hits"].update(
                                judgment["required_nugget_hits"]
                            )
        projected = {
            key: derive_retrieval_grade(
                topic_relevance=state["topic_relevance"],
                required_nugget_hits=state["required_nugget_hits"],
                required_nugget_ids=required_by_query[key[0]],
            )
            for key, state in projected_state.items()
        }
        derived = output_root / "derived" / config_dir.name
        qrels_dir = derived / "qrels"
        rows = sorted((query_id, chunk_id, grade) for (query_id, chunk_id), grade in projected.items())
        beir_text = "query-id\tcorpus-id\tscore\n" + "".join(
            f"{query_id}\t{chunk_id}\t{grade}\n" for query_id, chunk_id, grade in rows
        )
        trec_text = "".join(f"{query_id} 0 {chunk_id} {grade}\n" for query_id, chunk_id, grade in rows)
        _write_text(qrels_dir / "seed.tsv", beir_text, paths)
        _write_text(derived / "qrels.seed.trec", trec_text, paths)
        corpus, loaded_queries, loaded_qrels = GenericDataLoader(
            data_folder=str(paths.root),
            corpus_file=str(config_dir / "beir" / "corpus.jsonl"),
            query_file=str(query_file),
            qrels_folder=str(qrels_dir),
        ).load("seed")
        if (
            set(corpus) != {chunk["id"] for chunk in chunks}
            or set(loaded_qrels) != answerable_ids
            or set(loaded_queries) != answerable_ids
        ):
            raise RuntimeError(f"BEIR round-trip failed for projected {config_dir.name} qrels")
        manifest = json.loads((config_dir / "manifest.json").read_text(encoding="utf-8"))
        projection = {
            "config_id": config_dir.name,
            "chunk_config_hash": manifest["chunk_config_hash"],
            "qrel_rows": len(rows),
            "query_topics": len(queries),
            "evaluated_topics": len(loaded_queries),
            "answerable_topics": len(loaded_qrels),
            "partial_positive_fragments_downgraded_to_grade_1": partial_positive_downgrades,
            "policy": "grade is derived from topic relevance and the union of fully covered required atomic nuggets; partial positive fragments contribute topic relevance only",
            "beir_generic_data_loader_roundtrip": True,
            "beir_qrels_sha256": sha256_file(qrels_dir / "seed.tsv"),
            "trec_qrels_sha256": sha256_file(derived / "qrels.seed.trec"),
        }
        write_json(derived / "projection_report.json", projection, paths)
        report[config_dir.name] = projection
    return report


def build_seed50(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    validate_topic_briefs(paths)
    raw_root = paths.root / "benchmarks" / "seed50" / "annotations" / "raw"
    annotation_paths = {
        "annotator_a": raw_root / "annotator_a.jsonl",
        "annotator_b": raw_root / "annotator_b.jsonl",
        "adjudicator": raw_root / "adjudicator.jsonl",
    }
    validation = {
        annotator: validate_annotation_file(path, annotator, paths)
        for annotator, path in annotation_paths.items()
    }
    submissions = {name: load_jsonl(path) for name, path in annotation_paths.items()}
    briefs = {brief["brief_id"]: brief for brief in load_jsonl(topic_briefs_path(paths))}
    adjudicated = sorted(submissions["adjudicator"], key=lambda item: item["brief_id"])
    for item in adjudicated:
        brief = briefs[item["brief_id"]]
        for field in ("intent", "answerability", "no_answer_reason", "evidence_topology"):
            if item[field] != brief[field]:
                raise RuntimeError(f"Adjudication changed preregistered {field}: {item['brief_id']}")

    queries: list[dict[str, Any]] = []
    nuggets: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    for item in adjudicated:
        brief = briefs[item["brief_id"]]
        query_id = f"seed-{item['brief_id']}"
        query = {
            "id": query_id,
            "schema_version": 1,
            "text": item["query_text"],
            "intent": item["intent"],
            "answerability": item["answerability"],
            "no_answer_reason": item["no_answer_reason"],
            "evidence_topology": item["evidence_topology"],
            "source_fact_group": item["source_fact_group"],
            "smoke": brief["smoke"],
            "smoke_slice": brief["smoke_slice"],
            "annotation_version": 1,
            "status": "provisional",
        }
        validate_record("Query", query, paths)
        queries.append(query)
        nugget_ids: dict[str, str] = {}
        for source in item["nuggets"]:
            nugget_id = "nugget-" + stable_id(query_id, source["local_id"], source["text"], canonical_json({"spans": source["source_span_ids"]}))
            nugget_ids[source["local_id"]] = nugget_id
            nugget = {
                "id": nugget_id,
                "schema_version": 1,
                "query_id": query_id,
                "text": source["text"],
                "required": source["required"],
                "source_span_ids": source["source_span_ids"],
                "status": "provisional",
            }
            validate_record("Nugget", nugget, paths)
            nuggets.append(nugget)
        required_ids = {
            nugget_ids[source["local_id"]]
            for source in item["nuggets"]
            if source["required"]
        }
        for source in item["judgments"]:
            required_hits = [
                nugget_ids[local_id]
                for local_id in source["local_nugget_ids"]
                if local_id in nugget_ids and nugget_ids[local_id] in required_ids
            ]
            topic_relevance = source["grade"] >= 1
            judgment = {
                "id": "judgment-"
                + stable_id(
                    query_id,
                    source["source_span_id"],
                    canonical_json(
                        {
                            "topic_relevance": topic_relevance,
                            "required_nugget_hits": required_hits,
                            "answerability": item["answerability"],
                        }
                    ),
                ),
                "schema_version": 1,
                "query_id": query_id,
                "source_span_id": source["source_span_id"],
                "nugget_ids": required_hits,
                "topic_relevance": topic_relevance,
                "required_nugget_hits": required_hits,
                "answerability": item["answerability"],
                "grade": derive_retrieval_grade(
                    topic_relevance=topic_relevance,
                    required_nugget_hits=required_hits,
                    required_nugget_ids=required_ids,
                ),
                "rationale": source["rationale"],
                "status": "provisional",
            }
            validate_record("Judgment", judgment, paths)
            judgments.append(judgment)
            if source["grade"] <= 1:
                negative = {
                    "id": "negative-" + stable_id(query_id, source["source_span_id"]),
                    "schema_version": 1,
                    "query_id": query_id,
                    "source_span_id": source["source_span_id"],
                    "grade": source["grade"],
                    "provenance": "seeded_agent_candidate",
                    "status": "provisional",
                }
                validate_record("HardNegative", negative, paths)
                hard_negatives.append(negative)

    output_root = paths.root / "benchmarks" / "seed50" / "provisional"
    write_jsonl(output_root / "queries.jsonl", queries, paths=paths)
    write_jsonl(output_root / "nuggets.jsonl", nuggets, paths=paths)
    write_jsonl(output_root / "judgments.jsonl", judgments, paths=paths)
    write_jsonl(output_root / "hard_negatives.jsonl", hard_negatives, paths=paths)
    write_jsonl(
        output_root / "beir" / "queries.jsonl",
        ({"_id": query["id"], "text": query["text"]} for query in queries),
        sort_key="_id",
        paths=paths,
    )
    for annotator, records in submissions.items():
        write_jsonl(
            paths.root / "benchmarks" / "seed50" / "annotations" / f"{annotator}.jsonl",
            records,
            sort_key="brief_id",
            paths=paths,
        )
    agreement = _agreement(submissions["annotator_a"], submissions["annotator_b"])
    agreement["topic_contract_correction_brief_ids"] = sorted(TOPIC_CONTRACT_CORRECTIONS)
    write_json(output_root / "agreement_report.json", agreement, paths)
    projections = _project_qrels(paths, output_root, queries, judgments, nuggets)

    no_answer_ids = {item["brief_id"] for item in adjudicated if item["answerability"] == "no_answer"}
    disagreement_ids = set(agreement["substantive_disagreement_brief_ids"])
    queue_ids = no_answer_ids | disagreement_ids | TOPIC_CONTRACT_CORRECTIONS
    if len(queue_ids) < 15:
        for item in adjudicated:
            queue_ids.add(item["brief_id"])
            if len(queue_ids) >= 15:
                break
    review_queue = [
        {
            "id": brief_id,
            "brief_id": brief_id,
            "query_id": f"seed-{brief_id}",
            "reasons": sorted(
                reason
                for reason, applies in (
                    ("no_answer", brief_id in no_answer_ids),
                    ("agent_disagreement", brief_id in disagreement_ids),
                    ("topic_contract_correction", brief_id in TOPIC_CONTRACT_CORRECTIONS),
                    (
                        "stratified_minimum",
                        brief_id not in no_answer_ids | disagreement_ids | TOPIC_CONTRACT_CORRECTIONS,
                    ),
                )
                if applies
            ),
            "status": "pending_human_review",
        }
        for brief_id in sorted(queue_ids)
    ]
    write_jsonl(output_root / "human_review" / "queue.jsonl", review_queue, paths=paths)
    review_template = {
        "phase": "seed50-human-review",
        "decision": "PENDING",
        "reviewer": None,
        "annotation_version": 1,
        "reviewed_query_ids": [],
        "corrections": [],
        "required_scope": "all agent disagreements, all no-answer topics, and at least 15 total topics",
    }
    write_json(output_root / "human_review" / "receipt.template.json", review_template, paths)
    smoke_ids = [query["id"] for query in queries if query["smoke"]]
    _write_text(output_root / "smoke_query_ids.txt", "".join(f"{query_id}\n" for query_id in smoke_ids), paths)

    canonical_files = (
        "queries.jsonl",
        "nuggets.jsonl",
        "judgments.jsonl",
        "hard_negatives.jsonl",
        "beir/queries.jsonl",
        "agreement_report.json",
        "human_review/queue.jsonl",
        "human_review/receipt.template.json",
        "smoke_query_ids.txt",
    )
    output_hashes = {relative: sha256_file(output_root / relative) for relative in canonical_files}
    raw_annotation_hashes = {
        annotator: sha256_file(paths.root / "benchmarks" / "seed50" / "annotations" / "raw" / f"{annotator}.jsonl")
        for annotator in submissions
    }
    canonical_annotation_hashes = {
        annotator: sha256_file(paths.root / "benchmarks" / "seed50" / "annotations" / f"{annotator}.jsonl")
        for annotator in submissions
    }
    binding = {
        "output_hashes": output_hashes,
        "topic_briefs_sha256": sha256_file(topic_briefs_path(paths)),
        "raw_annotation_input_hashes": raw_annotation_hashes,
        "canonical_annotation_hashes": canonical_annotation_hashes,
        "projection_qrels_hashes": {
            config_id: {
                "beir": projection["beir_qrels_sha256"],
                "trec": projection["trec_qrels_sha256"],
            }
            for config_id, projection in projections.items()
        },
    }
    manifest = {
        "id": stable_id(SNAPSHOT_ID, canonical_json(binding)),
        "schema_version": 1,
        "benchmark": "Seed50",
        "benchmark_version": "1-provisional",
        "annotation_version": 1,
        "status": "provisional",
        "source_snapshot_id": SNAPSHOT_ID,
        "human_review_complete": False,
        "sota_claims_allowed": False,
        "counts": {
            "queries": len(queries),
            "answerable": sum(query["answerability"] == "answerable" for query in queries),
            "no_answer": sum(query["answerability"] == "no_answer" for query in queries),
            "nuggets": len(nuggets),
            "judgments": len(judgments),
            "seeded_hard_negatives": len(hard_negatives),
            "human_review_queue": len(review_queue),
        },
        "annotation_validation": validation,
        "topic_briefs_sha256": binding["topic_briefs_sha256"],
        "raw_annotation_input_hashes": raw_annotation_hashes,
        "canonical_annotation_hashes": canonical_annotation_hashes,
        "projection_reports": projections,
        "output_hashes": output_hashes,
    }
    write_json(output_root / "manifest.json", manifest, paths)
    return manifest
