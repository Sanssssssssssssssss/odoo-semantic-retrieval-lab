from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .benchmark import SNAPSHOT_ID, load_jsonl
from .contract import validate_record
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .verify import _git


CONFIG_RELATIVE = Path("configs/benchmark-v0.json")
PUBLIC_ROOT_RELATIVE = Path("benchmarks/v0/bootstrap")
SEED_REVIEW_RELATIVE = Path("benchmarks/seed50/pooling/provisional/human_review/queue.jsonl")
PUBLIC_FILE_ALLOWLIST = {
    "benchmarks/v0/bootstrap/manifest.json",
    "benchmarks/v0/bootstrap/calibration/annotator_a.template.jsonl",
    "benchmarks/v0/bootstrap/calibration/annotator_b.template.jsonl",
    "benchmarks/v0/bootstrap/calibration/review_items.jsonl",
    "benchmarks/v0/bootstrap/public/dev_topic_slots.jsonl",
    "benchmarks/v0/bootstrap/public/hidden_commitment.json",
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_config(paths: LabPaths) -> dict[str, Any]:
    config = json.loads((paths.root / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("source_snapshot_id") != SNAPSHOT_ID:
        raise RuntimeError("V0 config is not bound to the frozen Odoo 19 snapshot")
    topics = config["topics"]
    if sum(topics["split_counts"].values()) != topics["total"] or topics["total"] != 160:
        raise RuntimeError("V0 topic total must be exactly 160")
    for split, total in topics["split_counts"].items():
        answerability = topics["answerability_by_split"][split]
        if sum(topics["intent_by_split"][split].values()) != total:
            raise RuntimeError(f"V0 intent quota mismatch for {split}")
        if sum(topics["difficulty_by_split"][split].values()) != total:
            raise RuntimeError(f"V0 difficulty quota mismatch for {split}")
        if sum(answerability.values()) != total:
            raise RuntimeError(f"V0 answerability quota mismatch for {split}")
        if sum(topics["evidence_topology_by_split"][split].values()) != answerability["answerable"]:
            raise RuntimeError(f"V0 topology quota mismatch for {split}")
        if sum(topics["no_answer_reason_by_split"][split].values()) != answerability["no_answer"]:
            raise RuntimeError(f"V0 no-answer quota mismatch for {split}")
    calibration = config["tooling_calibration"]
    if sum(calibration["selection_reason_counts"].values()) != calibration["items"]:
        raise RuntimeError("Tooling calibration quotas do not match item count")
    if config["hidden"]["public_content_allowed"] is not False:
        raise RuntimeError("V0 hidden content must be forbidden from public outputs")
    return config


def _expanded_values(counts: dict[str, int], salt: str) -> list[str]:
    values = [(value, occurrence) for value, count in counts.items() for occurrence in range(count)]
    return [
        value
        for value, occurrence in sorted(
            values,
            key=lambda item: stable_id(salt, item[0], str(item[1]), length=64),
        )
    ]


def build_topic_slots(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topics = config["topics"]
    slots: dict[str, list[dict[str, Any]]] = {"dev": [], "shadow_hidden": []}
    next_id = 1
    for split in ("dev", "shadow_hidden"):
        total = topics["split_counts"][split]
        intents = _expanded_values(topics["intent_by_split"][split], f"{split}:intent")
        difficulties = _expanded_values(topics["difficulty_by_split"][split], f"{split}:difficulty")
        answerabilities = _expanded_values(
            topics["answerability_by_split"][split], f"{split}:answerability"
        )
        topologies = iter(
            _expanded_values(topics["evidence_topology_by_split"][split], f"{split}:topology")
        )
        reasons = iter(
            _expanded_values(topics["no_answer_reason_by_split"][split], f"{split}:reason")
        )
        for offset in range(total):
            answerability = answerabilities[offset]
            record = {
                "id": f"v0-{next_id:03d}",
                "schema_version": 1,
                "split": split,
                "calibration": split == "dev" and offset < topics["calibration_dev_topics"],
                "intent": intents[offset],
                "difficulty": difficulties[offset],
                "answerability": answerability,
                "no_answer_reason": next(reasons) if answerability == "no_answer" else None,
                "evidence_topology": next(topologies) if answerability == "answerable" else None,
                "status": "empty_human_authored_required",
            }
            validate_record("V0TopicSlot", record)
            slots[split].append(record)
            next_id += 1
    if next_id != topics["total"] + 1:
        raise RuntimeError("V0 slot generation did not produce exactly 160 records")
    return slots["dev"], slots["shadow_hidden"]


def canonicalize_review_queue(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    raw_count = 0
    for row in rows:
        raw_count += 1
        span_ids = sorted(
            {
                span_id
                for fragment in row["context"]["span_fragments"]
                for span_id in fragment["source_span_ids"]
            }
        )
        if not span_ids:
            raise RuntimeError(f"Review row has no canonical SourceSpan: {row['id']}")
        query_id = row["query"]["id"]
        item_id = "canonical-review-" + stable_id(
            query_id, canonical_json({"source_span_ids": span_ids}), length=32
        )
        group = groups.setdefault(
            item_id,
            {
                "id": item_id,
                "schema_version": 1,
                "query_id": query_id,
                "source_span_ids": span_ids,
                "selection_reasons": set(),
                "pool_item_ids": set(),
                "chunk_config_ids": set(),
                "minimum_depth": row["depth"],
            },
        )
        if group["query_id"] != query_id or group["source_span_ids"] != span_ids:
            raise RuntimeError(f"Canonical review ID collision: {item_id}")
        group["selection_reasons"].update(row["selection_reasons"])
        group["pool_item_ids"].add(row["pool_item_id"])
        group["chunk_config_ids"].add(row["candidate"]["chunk_config_id"])
        group["minimum_depth"] = min(group["minimum_depth"], row["depth"])
    result = []
    for group in groups.values():
        record = {
            **group,
            "selection_reasons": sorted(group["selection_reasons"]),
            "pool_item_ids": sorted(group["pool_item_ids"]),
            "chunk_config_ids": sorted(group["chunk_config_ids"]),
        }
        validate_record("CanonicalReviewIndex", record)
        result.append(record)
    return sorted(result, key=lambda row: row["id"]), raw_count


def _pick_diverse(
    candidates: list[dict[str, Any]],
    count: int,
    queries: dict[str, dict[str, Any]],
    used: set[str],
) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["id"] not in used:
            strata[queries[row["query_id"]]["intent"]].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: row["id"])
    selected = []
    while len(selected) < count and any(strata.values()):
        for intent in sorted(strata):
            if len(selected) == count:
                break
            if strata[intent]:
                row = strata[intent].pop(0)
                selected.append(row)
                used.add(row["id"])
    if len(selected) != count:
        raise RuntimeError(f"Insufficient canonical review items for calibration quota {count}")
    return selected


def select_calibration_items(
    canonical_rows: list[dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    quotas = config["tooling_calibration"]["selection_reason_counts"]
    used: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []
    for bucket in ("stratified_agreement_sample", "no_answer", "agent_disagreement"):
        candidates = [
            row
            for row in canonical_rows
            if bucket in row["selection_reasons"]
            and (
                bucket != "agent_disagreement"
                or queries[row["query_id"]]["answerability"] == "answerable"
            )
        ]
        selected.extend(
            (bucket, row)
            for row in _pick_diverse(candidates, quotas[bucket], queries, used)
        )
    return sorted(selected, key=lambda item: item[1]["id"])


def _evidence_by_span(paths: LabPaths) -> dict[str, dict[str, Any]]:
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    result: dict[str, dict[str, Any]] = {}
    for unit in _iter_jsonl(evidence_root / "evidence_units.jsonl"):
        for span_id in unit["source_span_ids"]:
            evidence = {
                "source_span_id": span_id,
                "evidence_unit_id": unit["id"],
                "source_uri": unit["source_uri"],
                "heading_path": unit["heading_path"],
                "node_type": unit["node_type"],
                "text": unit["rendered_text"],
            }
            if span_id in result and result[span_id] != evidence:
                raise RuntimeError(f"SourceSpan maps to multiple EvidenceUnits: {span_id}")
            result[span_id] = evidence
    return result


def _build_packet(
    selected: list[tuple[str, dict[str, Any]]],
    queries: dict[str, dict[str, Any]],
    evidence_by_span: dict[str, dict[str, Any]],
    packet_id: str,
) -> list[dict[str, Any]]:
    items = []
    for bucket, row in selected:
        query = queries[row["query_id"]]
        missing = [span_id for span_id in row["source_span_ids"] if span_id not in evidence_by_span]
        if missing:
            raise RuntimeError(f"Calibration item references unknown SourceSpan: {missing[0]}")
        item = {
            "id": row["id"],
            "schema_version": 1,
            "packet_id": packet_id,
            "query_id": query["id"],
            "query_text": query["text"],
            "source_span_ids": row["source_span_ids"],
            "evidence": [evidence_by_span[span_id] for span_id in row["source_span_ids"]],
            "alias_count": len(row["pool_item_ids"]),
            "chunk_config_ids": row["chunk_config_ids"],
            "status": "tooling_calibration_only_not_v0_gold",
        }
        validate_record("ReviewPacketItem", item)
        items.append(item)
    return items


def _submission_templates(
    items: list[dict[str, Any]], annotator_id: str
) -> list[dict[str, Any]]:
    result = []
    for item in items:
        record = {
            "id": "review-submission-" + stable_id(annotator_id, item["id"]),
            "schema_version": 1,
            "packet_id": item["packet_id"],
            "canonical_item_id": item["id"],
            "annotator_id": annotator_id,
            "status": "PENDING",
            "retrieval_grade": None,
            "selected_source_span_ids": [],
            "topic_issue": None,
            "rationale": "",
        }
        validate_record("ReviewSubmission", record)
        result.append(record)
    return result


def verify_hidden_isolation(paths: LabPaths, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or _load_config(paths)
    private_root = config["hidden"]["private_relative_root"].replace("\\", "/").rstrip("/")
    if not private_root.startswith(".private/"):
        raise RuntimeError("Hidden root must remain under the ignored .private directory")
    ignored = _git(paths.root, "check-ignore", f"{private_root}/probe.jsonl")
    if ignored.replace("\\", "/") != f"{private_root}/probe.jsonl":
        raise RuntimeError("Hidden root is not protected by .gitignore")
    tracked = [line.replace("\\", "/") for line in _git(paths.root, "ls-files").splitlines()]
    forbidden = [
        path
        for path in tracked
        if path.startswith(f"{private_root}/") or path.startswith("benchmarks/v0/hidden/")
    ]
    if forbidden:
        raise RuntimeError(f"Hidden benchmark content is tracked publicly: {forbidden[0]}")
    public_root = paths.root / "benchmarks" / "v0"
    public_paths = sorted(path for path in public_root.rglob("*") if path.is_file()) if public_root.is_dir() else []
    unexpected = [
        str(path.relative_to(paths.root)).replace("\\", "/")
        for path in public_paths
        if str(path.relative_to(paths.root)).replace("\\", "/") not in PUBLIC_FILE_ALLOWLIST
    ]
    if unexpected:
        raise RuntimeError(f"Unapproved public V0 file: {unexpected[0]}")
    hidden_ids = {f"v0-{index:03d}" for index in range(97, 161)}
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        if any(hidden_id in text for hidden_id in hidden_ids):
            raise RuntimeError(f"Hidden topic identifier leaked into public benchmark data: {path}")
    dev_slots = public_root / "bootstrap" / "public" / "dev_topic_slots.jsonl"
    if dev_slots.is_file() and any(row.get("split") != "dev" for row in _iter_jsonl(dev_slots)):
        raise RuntimeError("Public topic slots contain a non-dev split")
    review_items = public_root / "bootstrap" / "calibration" / "review_items.jsonl"
    if review_items.is_file() and any(
        not row.get("query_id", "").startswith("seed-") for row in _iter_jsonl(review_items)
    ):
        raise RuntimeError("Public tooling calibration contains a non-Seed query")
    return {
        "private_relative_root": private_root,
        "gitignored": True,
        "tracked_hidden_paths": 0,
        "public_hidden_content": False,
    }


def run_v0_bootstrap(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    require_approval("final-delivery", paths)
    config = _load_config(paths)
    isolation = verify_hidden_isolation(paths, config)
    config_path = paths.root / CONFIG_RELATIVE
    queue_path = paths.root / SEED_REVIEW_RELATIVE
    query_path = paths.root / "benchmarks" / "seed50" / "provisional" / "queries.jsonl"
    evidence_path = (
        paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence" / "evidence_units.jsonl"
    )
    canonical_rows, raw_count = canonicalize_review_queue(_iter_jsonl(queue_path))
    queries = {row["id"]: row for row in load_jsonl(query_path)}
    selected = select_calibration_items(canonical_rows, queries, config)
    dev_slots, hidden_slots = build_topic_slots(config)
    implementation_paths = [
        paths.root / "lab.ps1",
        paths.root / "src/osrlab/cli.py",
        paths.root / "src/osrlab/v0.py",
        paths.root / "src/osrlab/benchmark.py",
        paths.root / "src/osrlab/contract.py",
        paths.root / "src/osrlab/gates.py",
        paths.root / "src/osrlab/jsonio.py",
        paths.root / "src/osrlab/paths.py",
        paths.root / "src/osrlab/verify.py",
        paths.root / "schemas/osrlab.schema.json",
    ]
    implementation_hashes = {
        str(path.relative_to(paths.root)).replace("\\", "/"): sha256_file(path)
        for path in implementation_paths
    }
    input_hashes = {
        str(CONFIG_RELATIVE).replace("\\", "/"): sha256_file(config_path),
        str(SEED_REVIEW_RELATIVE).replace("\\", "/"): sha256_file(queue_path),
        "benchmarks/seed50/provisional/queries.jsonl": sha256_file(query_path),
        f"corpus/derived/{SNAPSHOT_ID}/evidence/evidence_units.jsonl": sha256_file(evidence_path),
    }
    content_binding_sha256 = stable_id(
        canonical_json(
            {
                "canonical_ids": [row["id"] for row in canonical_rows],
                "selected": [(bucket, row["id"]) for bucket, row in selected],
                "dev_slots": dev_slots,
                "hidden_slots": hidden_slots,
            }
        ),
        length=64,
    )
    run_id = stable_id(
        canonical_json(input_hashes),
        canonical_json(implementation_hashes),
        content_binding_sha256,
        length=40,
    )
    packet_id = stable_id(
        run_id,
        canonical_json({"selected": [row["id"] for _, row in selected]}),
        length=40,
    )
    packet = _build_packet(
        selected,
        queries,
        _evidence_by_span(paths),
        packet_id,
    )

    public_root = paths.root / PUBLIC_ROOT_RELATIVE
    public_topic_root = public_root / "public"
    calibration_root = public_root / "calibration"
    write_jsonl(public_topic_root / "dev_topic_slots.jsonl", dev_slots, paths=paths)
    private_root = paths.require_write_path(
        paths.root / config["hidden"]["private_relative_root"] / "bootstrap"
    )
    hidden_slots_path = private_root / "shadow_hidden_topic_slots.jsonl"
    write_jsonl(hidden_slots_path, hidden_slots, paths=paths)
    hidden_commitment = {
        "benchmark_id": config["benchmark_id"],
        "schema_version": 1,
        "split": "shadow_hidden",
        "topic_count": len(hidden_slots),
        "topic_slots_sha256": sha256_file(hidden_slots_path),
        "contains_query_or_gold_content": False,
        "status": "empty_private_template_not_frozen_hidden_gold",
    }
    write_json(public_topic_root / "hidden_commitment.json", hidden_commitment, paths)
    write_jsonl(calibration_root / "review_items.jsonl", packet, paths=paths)
    for annotator_id in ("annotator_a", "annotator_b"):
        write_jsonl(
            calibration_root / f"{annotator_id}.template.jsonl",
            _submission_templates(packet, annotator_id),
            paths=paths,
        )

    artifact_root = paths.root / "artifacts" / "v0" / "bootstrap" / run_id
    write_jsonl(artifact_root / "canonical_review_index.jsonl", canonical_rows, paths=paths)
    write_jsonl(
        artifact_root / "calibration_sampling.jsonl",
        (
            {
                "id": row["id"],
                "selection_bucket": bucket,
                "selection_reasons": row["selection_reasons"],
            }
            for bucket, row in selected
        ),
        paths=paths,
    )
    report = (
        "# V0 benchmark bootstrap\n\n"
        f"- Chunk-level Seed50 review rows: {raw_count}\n"
        f"- Exact canonical query/source-span groups: {len(canonical_rows)}\n"
        f"- Tooling calibration items: {len(packet)} (not V0 gold)\n"
        "- Formal V0 topics: 160 empty human-authored slots; 96 public dev and 64 private shadow-hidden.\n"
        "- Agent-created final gold: forbidden.\n"
    )
    report_path = paths.require_write_path(artifact_root / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    output_paths = {
        "benchmarks/v0/bootstrap/public/dev_topic_slots.jsonl": public_topic_root / "dev_topic_slots.jsonl",
        "benchmarks/v0/bootstrap/public/hidden_commitment.json": public_topic_root / "hidden_commitment.json",
        "benchmarks/v0/bootstrap/calibration/review_items.jsonl": calibration_root / "review_items.jsonl",
        "benchmarks/v0/bootstrap/calibration/annotator_a.template.jsonl": calibration_root / "annotator_a.template.jsonl",
        "benchmarks/v0/bootstrap/calibration/annotator_b.template.jsonl": calibration_root / "annotator_b.template.jsonl",
        "artifacts/v0/bootstrap/canonical_review_index.jsonl": artifact_root / "canonical_review_index.jsonl",
        "artifacts/v0/bootstrap/calibration_sampling.jsonl": artifact_root / "calibration_sampling.jsonl",
        "artifacts/v0/bootstrap/report.md": report_path,
    }
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "phase": "v0-bootstrap",
        "status": "bootstrap_ready_human_staffing_required",
        "benchmark_id": config["benchmark_id"],
        "source_snapshot_id": SNAPSHOT_ID,
        "input_hashes": input_hashes,
        "implementation_hashes": implementation_hashes,
        "content_binding_sha256": content_binding_sha256,
        "packet_id": packet_id,
        "output_hashes": {name: sha256_file(path) for name, path in output_paths.items()},
        "hidden_template_sha256": hidden_commitment["topic_slots_sha256"],
        "hidden_isolation": isolation,
        "counts": {
            "raw_chunk_review_rows": raw_count,
            "canonical_review_groups": len(canonical_rows),
            "tooling_calibration_items": len(packet),
            "dev_topic_slots": len(dev_slots),
            "shadow_hidden_topic_slots": len(hidden_slots),
        },
        "formal_gold_created": False,
        "human_annotation_complete": False,
        "sota_claims_allowed": False,
    }
    write_json(public_root / "manifest.json", manifest, paths)
    write_json(artifact_root / "manifest.json", manifest, paths)
    write_json(
        paths.root / "artifacts" / "v0" / "bootstrap" / "active.json",
        {"run_id": run_id, "root": str(artifact_root)},
        paths,
    )
    verify_hidden_isolation(paths, config)
    return {"root": str(artifact_root), "manifest": manifest}


def validate_calibration_submission(
    submission_path: Path,
    annotator_id: str,
    paths: LabPaths | None = None,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    if annotator_id not in {"annotator_a", "annotator_b"}:
        raise RuntimeError(f"Unknown annotator: {annotator_id}")
    root = paths.root / PUBLIC_ROOT_RELATIVE
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    item_path = root / "calibration" / "review_items.jsonl"
    item_hash_key = "benchmarks/v0/bootstrap/calibration/review_items.jsonl"
    if sha256_file(item_path) != manifest["output_hashes"][item_hash_key]:
        raise RuntimeError("Calibration packet hash does not match its manifest")
    items = {row["id"]: row for row in load_jsonl(item_path)}
    if any(row["packet_id"] != manifest["packet_id"] for row in items.values()):
        raise RuntimeError("Calibration item packet binding mismatch")
    submissions = load_jsonl(submission_path)
    by_item = {row["canonical_item_id"]: row for row in submissions}
    if len(by_item) != len(submissions) or set(by_item) != set(items):
        raise RuntimeError("Submission must contain exactly one record for every calibration item")
    grade_counts: Counter[int] = Counter()
    topic_issues = 0
    for item_id, submission in by_item.items():
        validate_record("ReviewSubmission", submission)
        item = items[item_id]
        if submission["annotator_id"] != annotator_id:
            raise RuntimeError(f"Annotator provenance mismatch: {item_id}")
        if submission["id"] != "review-submission-" + stable_id(annotator_id, item_id):
            raise RuntimeError(f"Submission ID binding mismatch: {item_id}")
        if submission["packet_id"] != item["packet_id"] or submission["status"] != "SUBMITTED":
            raise RuntimeError(f"Submission is pending or bound to the wrong packet: {item_id}")
        if not submission["rationale"].strip():
            raise RuntimeError(f"Submission rationale is required: {item_id}")
        if not set(submission["selected_source_span_ids"]) <= set(item["source_span_ids"]):
            raise RuntimeError(f"Submission selects a SourceSpan outside the item: {item_id}")
        grade = submission["retrieval_grade"]
        if submission["topic_issue"] is not None:
            if grade is not None:
                raise RuntimeError(f"Topic issues must not be mixed with a relevance grade: {item_id}")
            topic_issues += 1
            continue
        if grade is None:
            raise RuntimeError(f"Completed submission lacks a relevance grade: {item_id}")
        if grade >= 2 and not submission["selected_source_span_ids"]:
            raise RuntimeError(f"Positive judgment lacks SourceSpan evidence: {item_id}")
        grade_counts[grade] += 1
    return {
        "phase": "v0-tooling-calibration",
        "annotator_id": annotator_id,
        "packet_manifest_id": manifest["id"],
        "submission_sha256": sha256_file(submission_path),
        "items": len(submissions),
        "grade_counts": {str(grade): grade_counts[grade] for grade in range(4)},
        "topic_issues": topic_issues,
        "formal_gold_created": False,
    }


def prepare_calibration_adjudication(
    annotator_a_path: Path,
    annotator_b_path: Path,
    paths: LabPaths | None = None,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    a_report = validate_calibration_submission(annotator_a_path, "annotator_a", paths)
    b_report = validate_calibration_submission(annotator_b_path, "annotator_b", paths)
    root = paths.root / PUBLIC_ROOT_RELATIVE
    items = {row["id"]: row for row in load_jsonl(root / "calibration" / "review_items.jsonl")}
    a_rows = {row["canonical_item_id"]: row for row in load_jsonl(annotator_a_path)}
    b_rows = {row["canonical_item_id"]: row for row in load_jsonl(annotator_b_path)}

    def label(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["retrieval_grade"],
            tuple(row["selected_source_span_ids"]),
            row["topic_issue"],
        )

    disagreements = [item_id for item_id in sorted(items) if label(a_rows[item_id]) != label(b_rows[item_id])]
    adjudication_id = stable_id(
        a_report["submission_sha256"],
        b_report["submission_sha256"],
        canonical_json({"disagreements": disagreements}),
        length=40,
    )
    output = paths.require_write_path(
        paths.root / ".private" / "v0" / "adjudication" / adjudication_id
    )
    context = [
        {
            "id": item_id,
            "packet_item": items[item_id],
            "annotator_a": a_rows[item_id],
            "annotator_b": b_rows[item_id],
        }
        for item_id in disagreements
    ]
    template = _submission_templates([items[item_id] for item_id in disagreements], "adjudicator")
    write_jsonl(output / "context.jsonl", context, paths=paths)
    write_jsonl(output / "adjudicator.template.jsonl", template, paths=paths)
    manifest = {
        "id": adjudication_id,
        "schema_version": 1,
        "phase": "v0-tooling-calibration-adjudication",
        "status": "pending_human_adjudication" if disagreements else "no_disagreements",
        "packet_id": next(iter(items.values()))["packet_id"],
        "annotator_input_hashes": {
            "annotator_a": a_report["submission_sha256"],
            "annotator_b": b_report["submission_sha256"],
        },
        "items": len(items),
        "disagreements": len(disagreements),
        "output_hashes": {
            "context.jsonl": sha256_file(output / "context.jsonl"),
            "adjudicator.template.jsonl": sha256_file(output / "adjudicator.template.jsonl"),
        },
        "formal_gold_created": False,
    }
    write_json(output / "manifest.json", manifest, paths)
    return {"root": str(output), "manifest": manifest}


def validate_calibration_adjudication(
    adjudication_root: Path,
    submission_path: Path,
    annotator_a_path: Path,
    annotator_b_path: Path,
    paths: LabPaths | None = None,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    private_root = (paths.root / ".private" / "v0" / "adjudication").resolve()
    adjudication_root = adjudication_root.resolve()
    if not adjudication_root.is_relative_to(private_root):
        raise RuntimeError("Adjudication root must remain under .private/v0/adjudication")
    manifest = json.loads((adjudication_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("id") != adjudication_root.name:
        raise RuntimeError("Adjudication directory is not bound to its manifest ID")
    for name in ("context.jsonl", "adjudicator.template.jsonl"):
        if sha256_file(adjudication_root / name) != manifest["output_hashes"][name]:
            raise RuntimeError(f"Adjudication output hash mismatch: {name}")

    a_report = validate_calibration_submission(annotator_a_path, "annotator_a", paths)
    b_report = validate_calibration_submission(annotator_b_path, "annotator_b", paths)
    expected_hashes = {
        "annotator_a": a_report["submission_sha256"],
        "annotator_b": b_report["submission_sha256"],
    }
    if manifest.get("annotator_input_hashes") != expected_hashes:
        raise RuntimeError("Adjudication manifest is not bound to the supplied A/B submissions")

    a_rows = {row["canonical_item_id"]: row for row in load_jsonl(annotator_a_path)}
    b_rows = {row["canonical_item_id"]: row for row in load_jsonl(annotator_b_path)}

    def label(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["retrieval_grade"], tuple(row["selected_source_span_ids"]), row["topic_issue"]

    disagreements = sorted(item_id for item_id in a_rows if label(a_rows[item_id]) != label(b_rows[item_id]))
    expected_id = stable_id(
        expected_hashes["annotator_a"],
        expected_hashes["annotator_b"],
        canonical_json({"disagreements": disagreements}),
        length=40,
    )
    context = load_jsonl(adjudication_root / "context.jsonl")
    context_by_item = {row["id"]: row for row in context}
    if (
        expected_id != manifest["id"]
        or manifest["disagreements"] != len(disagreements)
        or len(context_by_item) != len(context)
        or sorted(context_by_item) != disagreements
    ):
        raise RuntimeError("Adjudication disagreement set does not match the supplied A/B submissions")
    for item_id, row in context_by_item.items():
        if row["annotator_a"] != a_rows[item_id] or row["annotator_b"] != b_rows[item_id]:
            raise RuntimeError(f"Adjudication context does not match A/B submissions: {item_id}")

    submissions = load_jsonl(submission_path)
    by_item = {row["canonical_item_id"]: row for row in submissions}
    if len(by_item) != len(submissions) or sorted(by_item) != disagreements:
        raise RuntimeError("Adjudicator submission must contain exactly the disagreement set")
    for item_id, submission in by_item.items():
        validate_record("ReviewSubmission", submission)
        item = context_by_item[item_id]["packet_item"]
        if submission["annotator_id"] != "adjudicator":
            raise RuntimeError(f"Adjudicator provenance mismatch: {item_id}")
        if submission["id"] != "review-submission-" + stable_id("adjudicator", item_id):
            raise RuntimeError(f"Adjudicator submission ID binding mismatch: {item_id}")
        if submission["packet_id"] != manifest["packet_id"] or submission["status"] != "SUBMITTED":
            raise RuntimeError(f"Adjudication is pending or bound to the wrong packet: {item_id}")
        if not submission["rationale"].strip():
            raise RuntimeError(f"Adjudication rationale is required: {item_id}")
        if not set(submission["selected_source_span_ids"]) <= set(item["source_span_ids"]):
            raise RuntimeError(f"Adjudication selects a SourceSpan outside the item: {item_id}")
        grade = submission["retrieval_grade"]
        if submission["topic_issue"] is not None:
            if grade is not None:
                raise RuntimeError(f"Topic issues must not be mixed with a relevance grade: {item_id}")
        elif grade is None:
            raise RuntimeError(f"Completed adjudication lacks a relevance grade: {item_id}")
        elif grade >= 2 and not submission["selected_source_span_ids"]:
            raise RuntimeError(f"Positive adjudication lacks SourceSpan evidence: {item_id}")
    return {
        "phase": "v0-tooling-calibration-adjudication",
        "adjudication_manifest_id": manifest["id"],
        "annotator_input_hashes": expected_hashes,
        "submission_sha256": sha256_file(submission_path),
        "disagreements": len(disagreements),
        "formal_gold_created": False,
    }
