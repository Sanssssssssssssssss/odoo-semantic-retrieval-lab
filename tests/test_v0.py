from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from osrlab.benchmark import load_jsonl
from osrlab.jsonio import canonical_json, sha256_file
from osrlab.paths import LabPaths
from osrlab.v0 import (
    PUBLIC_ROOT_RELATIVE,
    _load_config,
    build_topic_slots,
    canonicalize_review_queue,
    prepare_calibration_adjudication,
    validate_calibration_adjudication,
    validate_calibration_submission,
    verify_hidden_isolation,
)


def _pool_row(pool_id: str, config_id: str, spans: list[str]) -> dict:
    return {
        "id": "human-" + pool_id,
        "pool_item_id": pool_id,
        "depth": 20,
        "query": {"id": "seed-S001"},
        "candidate": {"chunk_config_id": config_id},
        "context": {
            "span_fragments": [
                {"source_span_ids": [span], "evidence_unit_id": "unit-" + span}
                for span in spans
            ]
        },
        "selection_reasons": ["agent_disagreement"],
    }


def test_v0_topic_slots_match_frozen_quotas() -> None:
    paths = LabPaths.discover()
    config = _load_config(paths)
    dev, hidden = build_topic_slots(config)
    assert len(dev) == 96 and len(hidden) == 64
    assert sum(row["calibration"] for row in dev) == 20
    assert not any(row["calibration"] for row in hidden)
    for split, rows in (("dev", dev), ("shadow_hidden", hidden)):
        topics = config["topics"]
        assert Counter(row["intent"] for row in rows) == topics["intent_by_split"][split]
        assert Counter(row["difficulty"] for row in rows) == topics["difficulty_by_split"][split]
        assert Counter(row["answerability"] for row in rows) == topics["answerability_by_split"][split]
        assert Counter(row["evidence_topology"] for row in rows if row["answerability"] == "answerable") == topics["evidence_topology_by_split"][split]
        assert Counter(row["no_answer_reason"] for row in rows if row["answerability"] == "no_answer") == topics["no_answer_reason_by_split"][split]


def test_canonical_review_deduplicates_chunk_aliases_only() -> None:
    rows, raw = canonicalize_review_queue(
        [
            _pool_row("pool-1", "C1-section-native", ["span-b", "span-a"]),
            _pool_row("pool-2", "C2-structure-bounded", ["span-a", "span-b"]),
            _pool_row("pool-3", "C2-structure-bounded", ["span-a"]),
        ]
    )
    assert raw == 3 and len(rows) == 2
    merged = next(row for row in rows if row["source_span_ids"] == ["span-a", "span-b"])
    assert merged["pool_item_ids"] == ["pool-1", "pool-2"]
    assert merged["chunk_config_ids"] == ["C1-section-native", "C2-structure-bounded"]


def test_v0_public_bootstrap_has_no_hidden_payload_and_is_hash_bound() -> None:
    paths = LabPaths.discover()
    root = paths.root / PUBLIC_ROOT_RELATIVE
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["output_hashes"].items():
        if relative.startswith("benchmarks/"):
            assert sha256_file(paths.root / relative) == digest
    dev = load_jsonl(root / "public" / "dev_topic_slots.jsonl")
    assert len(dev) == 96 and {row["split"] for row in dev} == {"dev"}
    assert all("query_text" not in row and "source_span_ids" not in row for row in dev)
    hidden = json.loads((root / "public" / "hidden_commitment.json").read_text(encoding="utf-8"))
    assert hidden["topic_count"] == 64 and hidden["contains_query_or_gold_content"] is False
    items = load_jsonl(root / "calibration" / "review_items.jsonl")
    assert len(items) == 20
    sensitive = {
        "selection_bucket", "selection_reasons", "intent",
        "no_answer_reason", "annotator_a", "annotator_b",
        "adjudicator", "retrieval_grade", "rationale",
    }
    assert all(not sensitive.intersection(row) for row in items)
    assert all("answerability" in row and "required_nuggets" in row for row in items)
    assert manifest["formal_gold_created"] is manifest["human_annotation_complete"] is False


def test_calibration_submission_validator_accepts_completed_blind_packet(tmp_path: Path) -> None:
    paths = LabPaths.discover()
    template = load_jsonl(
        paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / "annotator_a.template.jsonl"
    )
    items = {
        row["id"]: row
        for row in load_jsonl(paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / "review_items.jsonl")
    }
    for row in template:
        row.update(
            status="SUBMITTED",
            answerability=items[row["canonical_item_id"]]["answerability"],
            topic_relevance=False,
            retrieval_grade=0,
            rationale="Candidate evidence is nonrelevant.",
        )
    submission = tmp_path / "annotator_a.jsonl"
    submission.write_text("".join(canonical_json(row) + "\n" for row in template), encoding="utf-8", newline="\n")
    report = validate_calibration_submission(submission, "annotator_a", paths)
    assert report["items"] == 20 and report["grade_counts"]["0"] == 20
    assert report["formal_gold_created"] is False


def test_calibration_submission_validator_rejects_pending_packet(tmp_path: Path) -> None:
    paths = LabPaths.discover()
    template_path = paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / "annotator_a.template.jsonl"
    submission = tmp_path / "pending.jsonl"
    submission.write_bytes(template_path.read_bytes())
    with pytest.raises(RuntimeError, match="pending or bound to the wrong packet"):
        validate_calibration_submission(submission, "annotator_a", paths)


def test_hidden_store_is_gitignored_and_untracked() -> None:
    result = verify_hidden_isolation(LabPaths.discover())
    assert result["gitignored"] is True
    assert result["tracked_hidden_paths"] == 0


def _completed_submission(paths: LabPaths, annotator_id: str, grade: int = 0) -> list[dict]:
    rows = load_jsonl(
        paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / f"{annotator_id}.template.jsonl"
    )
    items = {
        row["id"]: row
        for row in load_jsonl(paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / "review_items.jsonl")
    }
    for row in rows:
        item = items[row["canonical_item_id"]]
        update = {
            "status": "SUBMITTED",
            "answerability": item["answerability"],
            "topic_relevance": grade >= 1,
            "retrieval_grade": grade,
            "rationale": "Independent human review.",
        }
        if grade >= 1:
            update["selected_source_span_ids"] = item["source_span_ids"][:1]
        if grade >= 2:
            hits = [nugget["id"] for nugget in item["required_nuggets"]]
            update["required_nugget_hits"] = hits if grade == 3 else hits[:1]
        row.update(update)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def test_adjudication_is_disagreement_only_and_validated(tmp_path: Path) -> None:
    paths = LabPaths.discover()
    a_rows = _completed_submission(paths, "annotator_a")
    b_rows = _completed_submission(paths, "annotator_b")
    first_item = {
        row["id"]: row
        for row in load_jsonl(
            paths.root / PUBLIC_ROOT_RELATIVE / "calibration" / "review_items.jsonl"
        )
    }[b_rows[0]["canonical_item_id"]]
    b_rows[0].update(
        topic_relevance=True,
        retrieval_grade=1,
        selected_source_span_ids=first_item["source_span_ids"][:1],
    )
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_rows(a_path, a_rows)
    _write_rows(b_path, b_rows)
    prepared = prepare_calibration_adjudication(a_path, b_path, paths)
    root = Path(prepared["root"])
    assert prepared["manifest"]["disagreements"] == 1
    context = load_jsonl(root / "context.jsonl")
    assert [row["id"] for row in context] == [b_rows[0]["canonical_item_id"]]
    adjudication = load_jsonl(root / "adjudicator.template.jsonl")
    adjudication[0].update(
        status="SUBMITTED",
        answerability=first_item["answerability"],
        topic_relevance=True,
        retrieval_grade=1,
        selected_source_span_ids=first_item["source_span_ids"][:1],
        rationale="Resolved from both reviews.",
    )
    submission = tmp_path / "adjudication.jsonl"
    _write_rows(submission, adjudication)
    report = validate_calibration_adjudication(root, submission, a_path, b_path, paths)
    assert report["disagreements"] == 1 and report["formal_gold_created"] is False


def test_adjudication_zero_disagreement_has_empty_private_packet(tmp_path: Path) -> None:
    paths = LabPaths.discover()
    a_rows = _completed_submission(paths, "annotator_a")
    b_rows = _completed_submission(paths, "annotator_b")
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_rows(a_path, a_rows)
    _write_rows(b_path, b_rows)
    prepared = prepare_calibration_adjudication(a_path, b_path, paths)
    root = Path(prepared["root"])
    assert prepared["manifest"]["status"] == "no_disagreements"
    assert load_jsonl(root / "context.jsonl") == []
    assert load_jsonl(root / "adjudicator.template.jsonl") == []
