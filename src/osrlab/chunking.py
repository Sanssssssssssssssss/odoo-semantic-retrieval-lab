from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .contract import validate_record
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths


SNAPSHOT_ID = "32a8b8d77833f22b4bc74ed4ea78b6a82b5338fd"
EXPECTED_EXTRACTION_MANIFEST_SHA256 = "4c56e53b3035a8ce2988ccfee64adc6580d510d5a822ce7816e78f6b6d855220"
EXPECTED_EVIDENCE_OUTPUTS = {
    "boundary_evidence_units.jsonl",
    "boundary_source_documents.jsonl",
    "evidence_units.jsonl",
    "exclusions.jsonl",
    "source_documents.jsonl",
    "source_spans.jsonl",
    "warnings.jsonl",
}
MODEL_KEY = "bge_small_en_v1_5"
CANONICAL_FILES = (
    "chunk_config.json",
    "chunks.jsonl",
    "beir/corpus.jsonl",
    "statistics.json",
    "integrity.json",
    "manifest.json",
)
CONFIG_PARAMETERS: dict[str, dict[str, Any]] = {
    "C0-fixed": {"window_tokens": 512, "overlap_tokens": 100, "document_boundary": True},
    "C1-section-native": {
        "logical_unit": "sphinx_section",
        "truncate": False,
        "scoring_window_tokens": 480,
        "scoring_overlap_tokens": 64,
        "logical_score": "max_subwindow",
    },
    "C2-structure-bounded": {
        "soft_tokens": 384,
        "hard_tokens": 512,
        "forced_overlap_tokens": 64,
        "atom_types": ["heading", "paragraph", "list_item", "table_row", "code", "admonition", "image_alt"],
    },
    "C3-structure-merged": {
        "basis": "C2-structure-bounded",
        "min_tokens": 128,
        "target_tokens": 384,
        "hard_tokens": 512,
        "parent_section_boundary": True,
    },
}


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_evidence_snapshot(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    manifest_path = evidence_root / "extraction_manifest.json"
    if sha256_file(manifest_path) != EXPECTED_EXTRACTION_MANIFEST_SHA256:
        raise RuntimeError("Frozen extraction manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_snapshot_id") != SNAPSHOT_ID or manifest.get("deterministic_double_run") is not True:
        raise RuntimeError("Evidence extraction manifest is not the frozen deterministic snapshot")
    if set(manifest.get("output_hashes", {})) != EXPECTED_EVIDENCE_OUTPUTS:
        raise RuntimeError("Frozen extraction manifest has an incomplete or unexpected output set")
    for relative, expected_hash in manifest["output_hashes"].items():
        path = evidence_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen EvidenceUnit snapshot mismatch: {relative}")
    return manifest


def _model_spec(paths: LabPaths) -> dict[str, Any]:
    return json.loads((paths.root / "configs" / "models.json").read_text(encoding="utf-8"))[MODEL_KEY]


def _snapshot_dir(paths: LabPaths, spec: dict[str, Any]) -> Path:
    return (
        paths.root
        / ".cache"
        / "huggingface"
        / f"models--{spec['repo_id'].replace('/', '--')}"
        / "snapshots"
        / spec["revision"]
    )


def verify_model_snapshot(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    spec = _model_spec(paths)
    snapshot = _snapshot_dir(paths, spec)
    if not snapshot.is_dir():
        raise RuntimeError(
            "Pinned BGE snapshot is absent. Run: "
            f".\\.venv\\Scripts\\hf.exe download {spec['repo_id']} --revision {spec['revision']} "
            "--cache-dir .cache\\huggingface"
        )
    actual: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(spec["files"].items()):
        path = snapshot / relative
        if not path.is_file():
            raise RuntimeError(f"Pinned model file is absent: {relative}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed != expected:
            raise RuntimeError(f"Pinned model file mismatch: {relative}: {observed}")
        actual[relative] = observed
    receipt = {
        "id": stable_id(spec["repo_id"], spec["revision"], canonical_json(actual)),
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "snapshot_path": str(snapshot.relative_to(paths.root)).replace("\\", "/"),
        "files": actual,
    }
    write_json(paths.root / "artifacts" / "models" / f"{MODEL_KEY}.json", receipt, paths)
    return receipt


def _chunk_configs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for config_id, parameters in CONFIG_PARAMETERS.items():
        unsigned = {
            "id": config_id,
            "schema_version": 1,
            "tokenizer_id": spec["repo_id"],
            "tokenizer_revision": spec["revision"],
            "parameters": parameters,
        }
        record = {**unsigned, "config_sha256": _digest_text(canonical_json(unsigned))}
        validate_record("ChunkConfig", record)
        output[config_id] = record
    return output


def _tokenizer(paths: LabPaths, spec: dict[str, Any]) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        _snapshot_dir(paths, spec),
        revision=spec["revision"],
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Chunk fragment mapping requires a fast tokenizer with character offsets")
    return tokenizer


def _tokenize(tokenizer: PreTrainedTokenizerFast, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    return list(encoded["input_ids"]), [tuple(pair) for pair in encoded["offset_mapping"]]


def _common_heading(units: list[dict[str, Any]]) -> list[str]:
    paths = [unit["heading_path"] for unit in units]
    common: list[str] = []
    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return common or list(paths[0])


def _ordered_unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value is not None))


def _sequence(
    tokenizer: PreTrainedTokenizerFast, units: list[dict[str, Any]]
) -> tuple[str, list[int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int, int]]]:
    parts: list[str] = []
    unit_chars: list[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        if parts:
            cursor += 2
        text = unit["rendered_text"]
        parts.append(text)
        unit_chars.append((cursor, cursor + len(text)))
        cursor += len(text)
    text = "\n\n".join(parts)
    token_ids, offsets = _tokenize(tokenizer, text)
    ownership: list[tuple[int, int, int]] = []
    local_counts = [0] * len(units)
    unit_index = 0
    for start, end in offsets:
        while unit_index + 1 < len(unit_chars) and start >= unit_chars[unit_index][1]:
            unit_index += 1
        unit_start, unit_end = unit_chars[unit_index]
        if start < unit_start or end > unit_end:
            raise RuntimeError(f"Tokenizer offset crosses an EvidenceUnit boundary: {(start, end)}")
        ownership.append((unit_index, local_counts[unit_index], start - unit_start))
        local_counts[unit_index] += 1
    return text, token_ids, offsets, unit_chars, ownership


def _fragments(
    units: list[dict[str, Any]],
    offsets: list[tuple[int, int]],
    unit_chars: list[tuple[int, int]],
    ownership: list[tuple[int, int, int]],
    token_start: int,
    token_end: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for absolute_index in range(token_start, token_end):
        unit_index, local_token, local_char = ownership[absolute_index]
        unit = units[unit_index]
        char_end = offsets[absolute_index][1] - unit_chars[unit_index][0]
        if (
            output
            and output[-1]["evidence_unit_id"] == unit["id"]
            and output[-1]["unit_token_end"] == local_token
            and output[-1]["chunk_token_end"] == absolute_index - token_start
        ):
            output[-1]["unit_token_end"] = local_token + 1
            output[-1]["chunk_token_end"] = absolute_index - token_start + 1
            output[-1]["unit_char_end"] = char_end
        else:
            output.append(
                {
                    "evidence_unit_id": unit["id"],
                    "source_span_ids": unit["source_span_ids"],
                    "source_uri": unit["source_uri"],
                    "anchor": unit.get("anchor"),
                    "unit_token_start": local_token,
                    "unit_token_end": local_token + 1,
                    "chunk_token_start": absolute_index - token_start,
                    "chunk_token_end": absolute_index - token_start + 1,
                    "unit_char_start": local_char,
                    "unit_char_end": char_end,
                }
            )
    return output


def _windows(length: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if length <= size:
        return [(0, length)]
    step = size - overlap
    output: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + size, length)
        output.append((start, end))
        if end == length:
            break
        start += step
    return output


def _c1_scoring_windows(length: int) -> list[tuple[int, int]]:
    return [(0, length)] if length <= 512 else _windows(length, 480, 64)


def _make_chunk(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    units: list[dict[str, Any]],
    token_start: int = 0,
    token_end: int | None = None,
    *,
    scoring_windows: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    full_text, token_ids, offsets, unit_chars, ownership = _sequence(tokenizer, units)
    token_end = len(token_ids) if token_end is None else token_end
    if not token_ids or token_start >= token_end:
        raise RuntimeError("Empty chunk is forbidden")
    char_start, char_end = offsets[token_start][0], offsets[token_end - 1][1]
    text = full_text[char_start:char_end]
    fragments = _fragments(units, offsets, unit_chars, ownership, token_start, token_end)
    heading = _common_heading(units)
    token_count = token_end - token_start
    windows = scoring_windows or [(0, token_count)]
    unsigned = {
        "schema_version": 1,
        "source_snapshot_id": SNAPSHOT_ID,
        "chunk_config_id": config["id"],
        "chunk_config_hash": config["config_sha256"],
        "source_document_id": units[0]["source_document_id"],
        "source_uri": units[0]["source_uri"],
        "section_ids": _ordered_unique(unit["section_id"] for unit in units),
        "parent_section_ids": _ordered_unique(unit["parent_section_id"] for unit in units),
        "heading_path": heading,
        "title": " > ".join(heading),
        "text": text,
        "text_sha256": _digest_text(text),
        "token_count": token_count,
        "span_fragments": fragments,
        "scoring_windows": [{"token_start": start, "token_end": end} for start, end in windows],
        "component_chunk_ids": [],
    }
    record = {
        "id": stable_id(
            SNAPSHOT_ID,
            config["config_sha256"],
            unsigned["source_document_id"],
            unsigned["text_sha256"],
            canonical_json({"fragments": fragments}),
        ),
        **unsigned,
    }
    validate_record("Chunk", record)
    return record


def _group(units: list[dict[str, Any]], key: str) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    for unit in units:
        candidate = (unit["source_document_id"], unit[key])
        if current and candidate != current_key:
            groups.append(current)
            current = []
        current.append(unit)
        current_key = candidate
    if current:
        groups.append(current)
    return groups


def _c0(config: dict[str, Any], tokenizer: PreTrainedTokenizerFast, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for document in _group(units, "source_document_id"):
        _, ids, _, _, _ = _sequence(tokenizer, document)
        for start, end in _windows(len(ids), 512, 100):
            output.append(_make_chunk(config, tokenizer, document, start, end))
    return output


def _c1(config: dict[str, Any], tokenizer: PreTrainedTokenizerFast, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for section in _group(units, "section_id"):
        _, ids, _, _, _ = _sequence(tokenizer, section)
        scoring = _c1_scoring_windows(len(ids))
        output.append(_make_chunk(config, tokenizer, section, scoring_windows=scoring))
    return output


def _c2(config: dict[str, Any], tokenizer: PreTrainedTokenizerFast, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for section in _group(units, "section_id"):
        pending: list[dict[str, Any]] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending, pending_tokens
            if pending:
                chunk = _make_chunk(config, tokenizer, pending)
                if chunk["token_count"] > 512:
                    raise RuntimeError("C2 packed chunk exceeded hard limit")
                output.append(chunk)
                pending, pending_tokens = [], 0

        for unit in section:
            ids, _ = _tokenize(tokenizer, unit["rendered_text"])
            count = len(ids)
            if count > 512:
                flush()
                for start, end in _windows(count, 512, 64):
                    output.append(_make_chunk(config, tokenizer, [unit], start, end))
                continue
            if pending and (pending_tokens >= 384 or pending_tokens + count > 512):
                flush()
            pending.append(unit)
            pending_tokens += count
        flush()
    return output


def _merge_components(
    config: dict[str, Any], tokenizer: PreTrainedTokenizerFast, components: list[dict[str, Any]]
) -> dict[str, Any]:
    text = "\n\n".join(component["text"] for component in components)
    token_ids, offsets = _tokenize(tokenizer, text)
    component_chars: list[tuple[int, int]] = []
    cursor = 0
    for component in components:
        if component_chars:
            cursor += 2
        component_chars.append((cursor, cursor + len(component["text"])))
        cursor += len(component["text"])
    token_starts: list[int] = []
    for char_start, char_end in component_chars:
        indices = [i for i, (start, end) in enumerate(offsets) if start >= char_start and end <= char_end]
        if not indices:
            raise RuntimeError("Merged C3 component has no tokens")
        token_starts.append(indices[0])
    fragments: list[dict[str, Any]] = []
    for component, shift in zip(components, token_starts):
        for fragment in component["span_fragments"]:
            fragments.append(
                {
                    **fragment,
                    "chunk_token_start": fragment["chunk_token_start"] + shift,
                    "chunk_token_end": fragment["chunk_token_end"] + shift,
                }
            )
    heading_paths = [component["heading_path"] for component in components]
    heading: list[str] = []
    for parts in zip(*heading_paths):
        if len(set(parts)) != 1:
            break
        heading.append(parts[0])
    heading = heading or list(heading_paths[0])
    unsigned = {
        "schema_version": 1,
        "source_snapshot_id": SNAPSHOT_ID,
        "chunk_config_id": config["id"],
        "chunk_config_hash": config["config_sha256"],
        "source_document_id": components[0]["source_document_id"],
        "source_uri": components[0]["source_uri"],
        "section_ids": _ordered_unique(section for item in components for section in item["section_ids"]),
        "parent_section_ids": _ordered_unique(section for item in components for section in item["parent_section_ids"]),
        "heading_path": heading,
        "title": " > ".join(heading),
        "text": text,
        "text_sha256": _digest_text(text),
        "token_count": len(token_ids),
        "span_fragments": fragments,
        "scoring_windows": [{"token_start": 0, "token_end": len(token_ids)}],
        "component_chunk_ids": [component["id"] for component in components],
    }
    if len(token_ids) > 512:
        raise RuntimeError("C3 merged chunk exceeded hard limit")
    record = {
        "id": stable_id(
            SNAPSHOT_ID,
            config["config_sha256"],
            unsigned["source_document_id"],
            unsigned["text_sha256"],
            canonical_json({"fragments": fragments}),
        ),
        **unsigned,
    }
    validate_record("Chunk", record)
    return record


def _component_units(component: dict[str, Any]) -> set[str]:
    return {fragment["evidence_unit_id"] for fragment in component["span_fragments"]}


def _component_parent(component: dict[str, Any]) -> tuple[str, str | None]:
    parent = component["parent_section_ids"][0] if component["parent_section_ids"] else None
    return component["source_document_id"], parent


def _c3_segments(c2_chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent_groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for component in c2_chunks:
        parent_groups[_component_parent(component)].append(component)
    segments: list[list[dict[str, Any]]] = []
    for components in parent_groups.values():
        current: list[dict[str, Any]] = []
        for component in components:
            if current and _component_units(component) & _component_units(current[-1]):
                segments.append(current)
                current = []
            current.append(component)
        if current:
            segments.append(current)
    return segments


def _optimal_c3_partition(segment: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    # Objective: minimum under-min bins, then minimum total distance from target,
    # then minimum bin count and lexicographically earliest stable cut positions.
    count = len(segment)
    prefix = [0]
    for component in segment:
        prefix.append(prefix[-1] + component["token_count"])
    best: list[tuple[tuple[int, int, int, tuple[int, ...]], list[list[dict[str, Any]]]] | None] = [None] * (count + 1)
    best[0] = ((0, 0, 0, ()), [])
    for end in range(1, count + 1):
        candidates: list[tuple[tuple[int, int, int, tuple[int, ...]], list[list[dict[str, Any]]]]] = []
        for start in range(end - 1, -1, -1):
            tokens = prefix[end] - prefix[start]
            if tokens > 512:
                break
            previous = best[start]
            if previous is None:
                continue
            objective, bins = previous
            cuts = objective[3] + (end,)
            candidate_objective = (
                objective[0] + int(tokens < 128),
                objective[1] + abs(tokens - 384),
                objective[2] + 1,
                cuts,
            )
            candidates.append((candidate_objective, bins + [segment[start:end]]))
        if not candidates:
            raise RuntimeError("C3 component cannot fit within the hard token limit")
        best[end] = min(candidates, key=lambda item: item[0])
    return best[count][1]  # type: ignore[index]


def _optimal_c3_bins(c2_chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [chunk_bin for segment in _c3_segments(c2_chunks) for chunk_bin in _optimal_c3_partition(segment)]


def _c3(
    config: dict[str, Any], tokenizer: PreTrainedTokenizerFast, c2_chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [_merge_components(config, tokenizer, components) for components in _optimal_c3_bins(c2_chunks)]


def _percentile(values: list[int], probability: float) -> int:
    return sorted(values)[max(0, math.ceil(probability * len(values)) - 1)]


def _integrity(
    config_id: str,
    chunks: list[dict[str, Any]],
    units: list[dict[str, Any]],
    source_span_ids: set[str],
    document_ids: set[str],
    c2_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chunk_ids = [chunk["id"] for chunk in chunks]
    covered_units = {fragment["evidence_unit_id"] for chunk in chunks for fragment in chunk["span_fragments"]}
    referenced_spans = {span for chunk in chunks for fragment in chunk["span_fragments"] for span in fragment["source_span_ids"]}
    observed_docs = {chunk["source_document_id"] for chunk in chunks}
    fragment_bounds = all(
        0 <= fragment["chunk_token_start"] < fragment["chunk_token_end"] <= chunk["token_count"]
        and 0 <= fragment["unit_token_start"] < fragment["unit_token_end"]
        and 0 <= fragment["unit_char_start"] < fragment["unit_char_end"]
        for chunk in chunks
        for fragment in chunk["span_fragments"]
    )
    hard_limit = config_id == "C1-section-native" or all(chunk["token_count"] <= 512 for chunk in chunks)
    c1_scoring_policy = config_id != "C1-section-native" or all(
        (
            chunk["scoring_windows"] == [{"token_start": 0, "token_end": chunk["token_count"]}]
            if chunk["token_count"] <= 512
            else (
                chunk["scoring_windows"][0]["token_start"] == 0
                and chunk["scoring_windows"][-1]["token_end"] == chunk["token_count"]
                and all(window["token_end"] - window["token_start"] <= 480 for window in chunk["scoring_windows"])
                and all(
                    left["token_end"] - right["token_start"] == 64
                    for left, right in zip(chunk["scoring_windows"], chunk["scoring_windows"][1:])
                )
            )
        )
        for chunk in chunks
    )
    c3_minimum_policy = True
    c3_under_min = 0
    c3_optimal_under_min = 0
    c3_single_parent_identity = True
    if config_id == "C3-structure-merged":
        if c2_chunks is None:
            raise RuntimeError("C3 integrity requires its ordered C2 components")
        c3_under_min = sum(chunk["token_count"] < 128 for chunk in chunks)
        c3_optimal_under_min = sum(
            sum(item["token_count"] for item in chunk_bin) < 128 for chunk_bin in _optimal_c3_bins(c2_chunks)
        )
        c3_minimum_policy = c3_under_min == c3_optimal_under_min
        c2_by_id = {chunk["id"]: chunk for chunk in c2_chunks}
        c3_single_parent_identity = all(
            len({_component_parent(c2_by_id[component_id]) for component_id in chunk["component_chunk_ids"]}) == 1
            for chunk in chunks
        )
    checks = {
        "chunk_ids_unique": len(chunk_ids) == len(set(chunk_ids)),
        "all_evidence_units_covered": covered_units == {unit["id"] for unit in units},
        "all_source_spans_known": referenced_spans <= source_span_ids,
        "all_application_documents_covered": observed_docs == document_ids,
        "fragment_bounds_valid": fragment_bounds,
        "hard_limit_valid": hard_limit,
        "text_hashes_valid": all(_digest_text(chunk["text"]) == chunk["text_sha256"] for chunk in chunks),
        "c3_minimum_policy_valid": c3_minimum_policy,
        "c1_scoring_window_policy_valid": c1_scoring_policy,
        "c3_single_parent_identity": c3_single_parent_identity,
    }
    return {
        "config_id": config_id,
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "chunks": len(chunks),
            "covered_evidence_units": len(covered_units),
            "referenced_source_spans": len(referenced_spans),
            "covered_documents": len(observed_docs),
            "c3_unavoidable_chunks_below_min": c3_under_min,
            "c3_theoretical_minimum_chunks_below_min": c3_optimal_under_min,
        },
    }


def _write_output(
    paths: LabPaths,
    output_root: Path,
    configs: dict[str, dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    units: list[dict[str, Any]],
    source_span_ids: set[str],
    document_ids: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config_id in CONFIG_PARAMETERS:
        chunks = records[config_id]
        config_dir = output_root / config_id
        write_json(config_dir / "chunk_config.json", configs[config_id], paths)
        write_jsonl(config_dir / "chunks.jsonl", chunks, paths=paths)
        write_jsonl(
            config_dir / "beir" / "corpus.jsonl",
            ({"_id": chunk["id"], "title": chunk["title"], "text": chunk["text"]} for chunk in chunks),
            sort_key="_id",
            paths=paths,
        )
        loaded_corpus = GenericDataLoader(data_folder=str(config_dir / "beir")).load_corpus()
        beir_roundtrip = set(loaded_corpus) == {chunk["id"] for chunk in chunks}
        token_counts = [chunk["token_count"] for chunk in chunks]
        statistics = {
            "config_id": config_id,
            "chunk_count": len(chunks),
            "document_count": len({chunk["source_document_id"] for chunk in chunks}),
            "token_count": {
                "min": min(token_counts),
                "mean": sum(token_counts) / len(token_counts),
                "p50": _percentile(token_counts, 0.50),
                "p95": _percentile(token_counts, 0.95),
                "max": max(token_counts),
            },
            "logical_chunks_over_512": sum(count > 512 for count in token_counts),
            "span_fragment_count": sum(len(chunk["span_fragments"]) for chunk in chunks),
        }
        integrity = _integrity(
            config_id,
            chunks,
            units,
            source_span_ids,
            document_ids,
            records["C2-structure-bounded"] if config_id == "C3-structure-merged" else None,
        )
        integrity["checks"]["beir_generic_data_loader_roundtrip"] = beir_roundtrip
        integrity["passed"] = all(integrity["checks"].values())
        integrity["counts"]["beir_roundtrip_documents"] = len(loaded_corpus)
        if not integrity["passed"]:
            raise RuntimeError(f"Chunk integrity failed for {config_id}: {integrity}")
        write_json(config_dir / "statistics.json", statistics, paths)
        write_json(config_dir / "integrity.json", integrity, paths)
        output_hashes = {
            relative: sha256_file(config_dir / relative)
            for relative in CANONICAL_FILES
            if relative != "manifest.json"
        }
        manifest = {
            "id": stable_id(SNAPSHOT_ID, configs[config_id]["config_sha256"], canonical_json(output_hashes)),
            "schema_version": 1,
            "source_snapshot_id": SNAPSHOT_ID,
            "extraction_manifest_sha256": sha256_file(
                paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence" / "extraction_manifest.json"
            ),
            "chunk_config_hash": configs[config_id]["config_sha256"],
            "output_hashes": output_hashes,
        }
        write_json(config_dir / "manifest.json", manifest, paths)
        result[config_id] = {"statistics": statistics, "integrity": integrity, "manifest": manifest}
    return result


def _generate(paths: LabPaths, tokenizer: PreTrainedTokenizerFast, configs: dict[str, dict[str, Any]]) -> tuple[
    dict[str, list[dict[str, Any]]], list[dict[str, Any]], set[str], set[str]
]:
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = sorted(_load_jsonl(evidence_root / "evidence_units.jsonl"), key=lambda item: (item["source_document_id"], item["ordinal"], item["id"]))
    source_span_ids = {record["id"] for record in _load_jsonl(evidence_root / "source_spans.jsonl")}
    document_ids = {record["id"] for record in _load_jsonl(evidence_root / "source_documents.jsonl")}
    records: dict[str, list[dict[str, Any]]] = {}
    records["C0-fixed"] = _c0(configs["C0-fixed"], tokenizer, units)
    records["C1-section-native"] = _c1(configs["C1-section-native"], tokenizer, units)
    records["C2-structure-bounded"] = _c2(configs["C2-structure-bounded"], tokenizer, units)
    records["C3-structure-merged"] = _c3(configs["C3-structure-merged"], tokenizer, records["C2-structure-bounded"])
    return records, units, source_span_ids, document_ids


def chunk_twice(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    verify_evidence_snapshot(paths)
    model_receipt = verify_model_snapshot(paths)
    spec = _model_spec(paths)
    configs = _chunk_configs(spec)
    tokenizer = _tokenizer(paths, spec)
    temp_root = paths.require_write_path(paths.root / ".cache" / "chunk-determinism" / SNAPSHOT_ID)
    run_roots = [temp_root / "run-a", temp_root / "run-b"]
    for run_root in run_roots:
        checked = paths.require_write_path(run_root)
        if checked.exists():
            shutil.rmtree(checked)
        records, units, source_span_ids, document_ids = _generate(paths, tokenizer, configs)
        _write_output(paths, checked, configs, records, units, source_span_ids, document_ids)
    hashes: dict[str, str] = {}
    for config_id in CONFIG_PARAMETERS:
        for relative in CANONICAL_FILES:
            left = run_roots[0] / config_id / relative
            right = run_roots[1] / config_id / relative
            left_hash, right_hash = sha256_file(left), sha256_file(right)
            if left_hash != right_hash:
                raise RuntimeError(f"Chunk generation is nondeterministic: {config_id}/{relative}")
            hashes[f"{config_id}/{relative}"] = left_hash
    final_root = paths.require_write_path(paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks")
    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.copytree(run_roots[0], final_root)
    summary = {
        "source_snapshot_id": SNAPSHOT_ID,
        "tokenizer": {key: model_receipt[key] for key in ("repo_id", "revision", "license", "files")},
        "deterministic_double_run": True,
        "canonical_hashes": hashes,
        "configs": {
            config_id: json.loads((final_root / config_id / "statistics.json").read_text(encoding="utf-8"))
            for config_id in CONFIG_PARAMETERS
        },
    }
    write_json(final_root / "chunking_summary.json", summary, paths)
    return summary
