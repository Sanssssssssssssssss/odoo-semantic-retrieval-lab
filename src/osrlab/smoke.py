from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, TypeVar

import bm25s
import ir_measures
import numpy as np
import psutil

from .benchmark import SNAPSHOT_ID, load_jsonl
from .contract import validate_record
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .verify import verify_source


T = TypeVar("T")
RUN_TAG = "E0-C2-seed-smoke"
EXPECTED_C2_MANIFEST_SHA256 = "e13c1a7a86cd471eaa2bba38b4bfac26b366aea68e1ee3151330d220b7c52f4e"
EXPECTED_SEED50_MANIFEST_SHA256 = "4e6dd6790e3a464939a441fe8252b4704c1cc5b678b2ea66ba1f462f147e8183"
E0_CONFIG = {
    "system": "E0-BM25",
    "implementation": "bm25s",
    "implementation_version": version("bm25s"),
    "method": "lucene",
    "k1": 1.5,
    "b": 0.75,
    "lower": True,
    "stopwords": "english",
    "stemming": False,
    "indexed_fields": ["heading_path", "EvidenceUnit.lexical_text"],
    "partial_fragment_policy": "heading_path+bounded_rendered_text",
    "top_k": 100,
    "tie_break": "chunk_id_ascending",
}


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, _, chunk_id, grade = line.split()
            qrels[query_id][chunk_id] = int(grade)
    return dict(qrels)


def _verify_manifest_outputs(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["output_hashes"].items():
        if sha256_file(directory / relative) != expected:
            raise RuntimeError(f"Frozen input mismatch: {directory.name}/{relative}")
    return manifest


def _rank(scores: np.ndarray, chunk_ids: list[str], top_k: int) -> list[int]:
    if scores.shape != (len(chunk_ids),):
        raise RuntimeError("BM25 returned an unexpected score vector")
    return np.lexsort((np.asarray(chunk_ids), -scores))[:top_k].tolist()


def _lexical_chunk_text(
    chunk: dict[str, Any], units: dict[str, dict[str, Any]], unit_token_counts: dict[str, int]
) -> str:
    parts = [" > ".join(chunk["heading_path"])]
    seen: set[tuple[str, int, int]] = set()
    for fragment in chunk["span_fragments"]:
        key = (
            fragment["evidence_unit_id"],
            fragment["unit_token_start"],
            fragment["unit_token_end"],
        )
        if key in seen:
            continue
        seen.add(key)
        unit = units[fragment["evidence_unit_id"]]
        if fragment["unit_token_start"] == 0 and fragment["unit_token_end"] == unit_token_counts[unit["id"]]:
            parts.append(unit["lexical_text"])
        else:
            fragment_text = unit["rendered_text"][fragment["unit_char_start"] : fragment["unit_char_end"]]
            parts.append(" > ".join(unit["heading_path"]) + " " + fragment_text)
    return "\n".join(parts)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _interval_length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _overlap_length(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    total = 0
    for left_start, left_end in _merge_intervals(left):
        for right_start, right_end in _merge_intervals(right):
            total += max(0, min(left_end, right_end) - max(left_start, right_start))
    return total


def assemble_evidence_cards(
    query_id: str,
    ranking: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
    *,
    max_cards: int = 4,
    token_budget: int = 2_048,
    tokenizer: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    cards: list[dict[str, Any]] = []
    selected: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_fragments: set[tuple[str, int, int]] = set()
    cumulative = 0
    overlap_tokens = 0
    raw_tokens = 0
    for result in ranking:
        if len(cards) == max_cards:
            break
        chunk = chunks[result["chunk_id"]]
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fragment in chunk["span_fragments"]:
            intervals[fragment["evidence_unit_id"]].append(
                (fragment["unit_token_start"], fragment["unit_token_end"])
            )
        candidate_tokens = sum(_interval_length(value) for value in intervals.values())
        overlap = sum(_overlap_length(value, selected[unit_id]) for unit_id, value in intervals.items())
        if candidate_tokens and overlap / candidate_tokens > 0.8:
            continue
        novel = [
            fragment
            for fragment in chunk["span_fragments"]
            if (
                fragment["evidence_unit_id"],
                fragment["unit_token_start"],
                fragment["unit_token_end"],
            )
            not in seen_fragments
        ]
        if not novel:
            continue
        remaining = token_budget - cumulative
        chosen: list[dict[str, Any]] = []
        cut: dict[str, Any] | None = None
        for fragment in novel:
            length = fragment["unit_token_end"] - fragment["unit_token_start"]
            if length <= remaining:
                chosen.append(fragment)
                remaining -= length
                continue
            if chosen:
                last = chosen[-1]
                cut = {
                    "kind": "evidence_unit_boundary",
                    "evidence_unit_id": last["evidence_unit_id"],
                    "unit_token_end": last["unit_token_end"],
                    "unit_char_end": last["unit_char_end"],
                }
                break
            if tokenizer is not None and remaining > 0:
                unit = units[fragment["evidence_unit_id"]]
                offsets = tokenizer(
                    unit["rendered_text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    truncation=False,
                )["offset_mapping"]
                limit = min(fragment["unit_token_start"] + remaining, fragment["unit_token_end"])
                max_char = offsets[limit - 1][1]
                boundaries = [
                    match.end()
                    for match in re.finditer(r"(?:[.!?](?=\s|$)|\n+)", unit["rendered_text"])
                    if fragment["unit_char_start"] < match.end() <= max_char
                ]
                if boundaries:
                    char_end = boundaries[-1]
                    token_end = max(
                        index + 1
                        for index, (_, end) in enumerate(offsets)
                        if fragment["unit_token_start"] <= index < fragment["unit_token_end"] and end <= char_end
                    )
                    partial = dict(fragment)
                    partial["unit_token_end"] = token_end
                    partial["unit_char_end"] = char_end
                    partial["chunk_token_end"] = partial["chunk_token_start"] + token_end - partial["unit_token_start"]
                    chosen.append(partial)
                    remaining -= token_end - partial["unit_token_start"]
                    cut = {
                        "kind": "sentence_boundary",
                        "evidence_unit_id": partial["evidence_unit_id"],
                        "unit_token_end": token_end,
                        "unit_char_end": char_end,
                    }
            break
        if not chosen:
            continue
        novel_tokens = sum(fragment["unit_token_end"] - fragment["unit_token_start"] for fragment in chosen)
        excerpt = "\n\n".join(
            units[fragment["evidence_unit_id"]]["rendered_text"][
                fragment["unit_char_start"] : fragment["unit_char_end"]
            ]
            for fragment in chosen
        )
        cumulative += novel_tokens
        card = {
            "id": "card-" + stable_id(query_id, chunk["id"], str(result["rank"])),
            "schema_version": 1,
            "query_id": query_id,
            "rank": result["rank"],
            "chunk_id": chunk["id"],
            "retrieval_score": float(result["score"]),
            "source_uri": chosen[0]["source_uri"],
            "anchor": chosen[0]["anchor"],
            "excerpt": excerpt,
            "token_count": novel_tokens,
            "cumulative_token_count": cumulative,
            "source_span_ids": sorted({span for fragment in chosen for span in fragment["source_span_ids"]}),
            "span_fragments": chosen,
            "truncated": cut is not None,
            "cut": cut,
        }
        validate_record("EvidenceCard", card)
        cards.append(card)
        raw_tokens += candidate_tokens
        overlap_tokens += overlap
        chosen_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fragment in chosen:
            chosen_intervals[fragment["evidence_unit_id"]].append(
                (fragment["unit_token_start"], fragment["unit_token_end"])
            )
        for unit_id, value in chosen_intervals.items():
            selected[unit_id].extend(value)
            selected[unit_id] = _merge_intervals(selected[unit_id])
        seen_fragments.update(
            (fragment["evidence_unit_id"], fragment["unit_token_start"], fragment["unit_token_end"])
            for fragment in chosen
        )
        if cut is not None:
            break
    return cards, {
        "selected_cards": len(cards),
        "selected_tokens": cumulative,
        "overlap_tokens": overlap_tokens,
        "raw_selected_tokens": raw_tokens,
        "duplicate_evidence_rate": overlap_tokens / raw_tokens if raw_tokens else 0.0,
    }


def evaluate_ranking(
    qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]
) -> dict[str, float]:
    measures = {
        "ndcg_at_10": ir_measures.nDCG @ 10,
        "ndcg_at_3": ir_measures.nDCG @ 3,
        "p_at_3": ir_measures.P(rel=2) @ 3,
        "recall_at_3": ir_measures.Recall(rel=2) @ 3,
        "recall_at_20": ir_measures.Recall(rel=2) @ 20,
        "recall_at_100": ir_measures.Recall(rel=2) @ 100,
        "mrr_at_10": ir_measures.RR(rel=2) @ 10,
        "map": ir_measures.AP(rel=2),
        "judged_at_10": ir_measures.Judged @ 10,
        "bpref": ir_measures.BPref(rel=2),
    }
    calculated = ir_measures.calc_aggregate(measures.values(), qrels, run)
    return {name: float(calculated[measure]) for name, measure in measures.items()}


def _answerability_diagnostics(labels_and_scores: list[tuple[str, int, float]]) -> dict[str, Any]:
    positives = [score for _, label, score in labels_and_scores if label == 1]
    negatives = [score for _, label, score in labels_and_scores if label == 0]
    wins = sum((left > right) + 0.5 * (left == right) for left in positives for right in negatives)
    auroc = wins / (len(positives) * len(negatives))
    ranked = sorted(labels_and_scores, key=lambda item: (-item[2], item[0]))
    hits = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ranked, 1):
        hits += label
        if label:
            precision_sum += hits / rank
    return {
        "score": "maximum_BM25_score",
        "answerable_positive": True,
        "auroc": auroc,
        "auprc_average_precision": precision_sum / len(positives),
        "topics": [
            {"query_id": query_id, "answerable": bool(label), "score": score}
            for query_id, label, score in ranked
        ],
        "threshold_metrics": "deferred_to_Seed50_source_fact_group_cross_validation",
    }


def _source_document_ndcg(
    answerable_ids: set[str],
    rankings: dict[str, list[dict[str, Any]]],
    chunks: dict[str, dict[str, Any]],
    judgments: list[dict[str, Any]],
    span_to_document: dict[str, str],
) -> float:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgments:
        if judgment["query_id"] in answerable_ids:
            document_id = span_to_document[judgment["source_span_id"]]
            qrels[judgment["query_id"]][document_id] = max(
                judgment["grade"], qrels[judgment["query_id"]].get(document_id, 0)
            )
    run: dict[str, dict[str, float]] = defaultdict(dict)
    for query_id in sorted(answerable_ids):
        for result in rankings[query_id]:
            document_id = chunks[result["chunk_id"]]["source_document_id"]
            run[query_id][document_id] = max(result["score"], run[query_id].get(document_id, -math.inf))
    return float(ir_measures.calc_aggregate([ir_measures.nDCG @ 10], dict(qrels), dict(run))[ir_measures.nDCG @ 10])


def _evidence_metrics(
    answerable_ids: set[str],
    cards_by_query: dict[str, list[dict[str, Any]]],
    card_diagnostics: dict[str, dict[str, float | int]],
    judgments: list[dict[str, Any]],
    nuggets: list[dict[str, Any]],
    unit_token_counts: dict[str, int],
) -> dict[str, float]:
    gold_spans: dict[str, set[str]] = defaultdict(set)
    for judgment in judgments:
        if judgment["query_id"] in answerable_ids and judgment["grade"] >= 2:
            gold_spans[judgment["query_id"]].add(judgment["source_span_id"])
    required: dict[str, list[set[str]]] = defaultdict(list)
    for nugget in nuggets:
        if nugget["query_id"] in answerable_ids and nugget["required"]:
            required[nugget["query_id"]].append(set(nugget["source_span_ids"]))
    evidence_recall: list[float] = []
    nugget_recall: list[float] = []
    completeness: list[float] = []
    irrelevant_ratios: list[float] = []
    for query_id in sorted(answerable_ids):
        retrieved: set[str] = set()
        relevant_tokens = 0
        total_tokens = 0
        for card in cards_by_query[query_id]:
            for fragment in card["span_fragments"]:
                length = fragment["unit_token_end"] - fragment["unit_token_start"]
                total_tokens += length
                full = (
                    fragment["unit_token_start"] == 0
                    and fragment["unit_token_end"] == unit_token_counts[fragment["evidence_unit_id"]]
                )
                relevant = full and bool(set(fragment["source_span_ids"]) & gold_spans[query_id])
                if relevant:
                    relevant_tokens += length
                    retrieved.update(set(fragment["source_span_ids"]) & gold_spans[query_id])
        evidence_recall.append(len(retrieved) / len(gold_spans[query_id]))
        covered = [span_ids <= retrieved for span_ids in required[query_id]]
        nugget_recall.append(sum(covered) / len(covered))
        completeness.append(float(all(covered)))
        irrelevant_ratios.append(1 - relevant_tokens / total_tokens if total_tokens else 1.0)
    return {
        "evidence_recall_at_2048_tokens": float(np.mean(evidence_recall)),
        "required_nugget_recall_at_2048_tokens": float(np.mean(nugget_recall)),
        "required_nugget_completeness_at_2048_tokens": float(np.mean(completeness)),
        "irrelevant_token_ratio_at_2048_tokens": float(np.mean(irrelevant_ratios)),
        "duplicate_evidence_rate_at_2048_tokens": float(
            np.mean(
                [
                    card_diagnostics[query_id]["duplicate_evidence_rate"]
                    for query_id in sorted(answerable_ids)
                ]
            )
        ),
    }


def _rss() -> tuple[int, int, int]:
    process = psutil.Process()
    children = process.children(recursive=True)
    own = process.memory_info().rss
    return own, own + sum(child.memory_info().rss for child in children if child.is_running()), len(children)


def _percentiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64) / 1_000_000
    return {f"p{percentile}_ms": float(np.percentile(array, percentile)) for percentile in (50, 90, 95, 99)}


def run_smoke(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    require_approval("p2a", paths)
    verify_source(paths)
    chunk_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / "C2-structure-bounded"
    if sha256_file(chunk_root / "manifest.json") != EXPECTED_C2_MANIFEST_SHA256:
        raise RuntimeError("Frozen C2 manifest hash mismatch")
    chunk_manifest = _verify_manifest_outputs(chunk_root)
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    benchmark_manifest_path = benchmark_root / "manifest.json"
    if sha256_file(benchmark_manifest_path) != EXPECTED_SEED50_MANIFEST_SHA256:
        raise RuntimeError("Approved provisional Seed50 manifest hash mismatch")
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if benchmark_manifest["status"] != "provisional" or benchmark_manifest["human_review_complete"]:
        raise RuntimeError("P3a smoke expects the approved provisional Seed50 contract")
    config_hash = stable_id(canonical_json(E0_CONFIG), length=64)
    run_id = stable_id(
        SNAPSHOT_ID,
        chunk_manifest["id"],
        sha256_file(benchmark_manifest_path),
        config_hash,
        length=40,
    )
    output = paths.require_write_path(paths.root / "artifacts" / "runs" / "p3a-smoke" / run_id)
    timings: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []

    def sample(stage: str, query_id: str | None) -> None:
        own, tree, children = _rss()
        record = {
            "run_id": run_id,
            "query_id": query_id,
            "stage": stage,
            "monotonic_ns": time.perf_counter_ns(),
            "rss_bytes": own,
            "process_tree_rss_bytes": tree,
            "child_processes": children,
        }
        validate_record("ResourceSample", record)
        resources.append(record)

    def timed(stage: str, query_id: str | None, function: Callable[[], T]) -> T:
        sample(stage + ":start", query_id)
        started = time.perf_counter_ns()
        value = function()
        record = {
            "run_id": run_id,
            "query_id": query_id,
            "stage": stage,
            "duration_ns": time.perf_counter_ns() - started,
        }
        validate_record("StageTiming", record)
        timings.append(record)
        sample(stage + ":end", query_id)
        return value

    chunks_list = timed("corpus_load", None, lambda: load_jsonl(chunk_root / "chunks.jsonl"))
    chunks = {chunk["id"]: chunk for chunk in chunks_list}
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    c1_chunks_path = (
        paths.root
        / "corpus"
        / "derived"
        / SNAPSHOT_ID
        / "chunks"
        / "C1-section-native"
        / "chunks.jsonl"
    )

    def load_evidence_metadata() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        loaded_units = {
            unit["id"]: unit for unit in load_jsonl(evidence_root / "evidence_units.jsonl")
        }
        loaded_spans = load_jsonl(evidence_root / "source_spans.jsonl")
        counts: dict[str, int] = {}
        for section_chunk in load_jsonl(c1_chunks_path):
            for fragment in section_chunk["span_fragments"]:
                unit_id = fragment["evidence_unit_id"]
                counts[unit_id] = max(counts.get(unit_id, 0), fragment["unit_token_end"])
        return loaded_units, loaded_spans, counts

    units, spans, unit_token_counts = timed("evidence_metadata_load", None, load_evidence_metadata)
    span_to_document = {span["id"]: span["source_document_id"] for span in spans}
    queries = timed("benchmark_load", None, lambda: load_jsonl(benchmark_root / "queries.jsonl"))
    smoke_queries = [query for query in queries if query["smoke"]]
    if len(smoke_queries) != 12:
        raise RuntimeError("P3a requires the preregistered Smoke12")
    query_by_id = {query["id"]: query for query in smoke_queries}
    corpus_text = [_lexical_chunk_text(chunk, units, unit_token_counts) for chunk in chunks_list]
    corpus_tokens = timed(
        "bm25_corpus_tokenize",
        None,
        lambda: bm25s.tokenize(
            corpus_text, lower=True, stopwords="english", stemmer=None, show_progress=False
        ),
    )
    retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    timed("bm25_index_build", None, lambda: retriever.index(corpus_tokens, show_progress=False))
    chunk_ids = [chunk["id"] for chunk in chunks_list]
    rankings: dict[str, list[dict[str, Any]]] = {}
    cards_by_query: dict[str, list[dict[str, Any]]] = {}
    card_diagnostics: dict[str, dict[str, float | int]] = {}
    for query in smoke_queries:
        query_id = query["id"]
        query_tokens = timed(
            "query_tokenize",
            query_id,
            lambda text=query["text"]: bm25s.tokenize(
                text,
                lower=True,
                stopwords="english",
                stemmer=None,
                return_ids=False,
                show_progress=False,
            )[0],
        )

        def search(tokens: list[str] = query_tokens) -> list[dict[str, Any]]:
            scores = np.asarray(retriever.get_scores(tokens), dtype=np.float32)
            indices = _rank(scores, chunk_ids, E0_CONFIG["top_k"])
            return [
                {"query_id": query_id, "rank": rank, "chunk_id": chunk_ids[index], "score": float(scores[index])}
                for rank, index in enumerate(indices, 1)
            ]

        ranking = timed("sparse_search", query_id, search)
        rankings[query_id] = ranking
        cards, diagnostics = timed(
            "evidence_card_assembly",
            query_id,
            lambda: assemble_evidence_cards(query_id, ranking, chunks, units),
        )
        cards_by_query[query_id] = cards
        card_diagnostics[query_id] = diagnostics

    qrels = _read_qrels(benchmark_root / "derived" / "C2-structure-bounded" / "qrels.seed.trec")
    answerable_ids = {query["id"] for query in smoke_queries if query["answerability"] == "answerable"}
    no_answer_ids = set(query_by_id) - answerable_ids
    if set(qrels) & no_answer_ids:
        raise RuntimeError("No-answer topics must not enter ordinary ranking qrels")
    smoke_qrels = {query_id: qrels[query_id] for query_id in sorted(answerable_ids)}
    smoke_run = {
        query_id: {result["chunk_id"]: result["score"] for result in rankings[query_id]}
        for query_id in sorted(answerable_ids)
    }
    judgments = load_jsonl(benchmark_root / "judgments.jsonl")
    nuggets = load_jsonl(benchmark_root / "nuggets.jsonl")
    def calculate_metrics() -> dict[str, Any]:
        fixed_chunk = evaluate_ranking(smoke_qrels, smoke_run)
        fixed_chunk["source_document_ndcg_at_10"] = _source_document_ndcg(
            answerable_ids, rankings, chunks, judgments, span_to_document
        )
        fixed_chunk.update(
            _evidence_metrics(
                answerable_ids, cards_by_query, card_diagnostics, judgments, nuggets, unit_token_counts
            )
        )
        labels_and_scores = [
            (
                query["id"],
                int(query["answerability"] == "answerable"),
                rankings[query["id"]][0]["score"],
            )
            for query in smoke_queries
        ]
        return {
            "status": "provisional_smoke_diagnostic",
            "ranking_scope": {
                "answerable_only": True,
                "query_count": len(answerable_ids),
                "query_ids": sorted(answerable_ids),
                "excluded_no_answer_query_ids": sorted(no_answer_ids),
                "binary_relevance_threshold": 2,
            },
            "fixed_chunk_metrics": fixed_chunk,
            "no_answer_diagnostics": _answerability_diagnostics(labels_and_scores),
            "sota_claims_allowed": False,
        }

    metrics = timed("metrics", None, calculate_metrics)
    trec = "".join(
        f"{query_id} Q0 {result['chunk_id']} {result['rank']} {result['score']:.9g} {RUN_TAG}\n"
        for query_id in sorted(rankings)
        for result in rankings[query_id]
    )
    top10 = [
        {
            **result,
            "query_text": query_by_id[query_id]["text"],
            "answerability": query_by_id[query_id]["answerability"],
            "title": chunks[result["chunk_id"]]["title"],
            "source_uri": chunks[result["chunk_id"]]["source_uri"],
            "qrel_grade": qrels.get(query_id, {}).get(result["chunk_id"]),
        }
        for query_id in sorted(rankings)
        for result in rankings[query_id][:10]
    ]
    cards = [card for query_id in sorted(cards_by_query) for card in cards_by_query[query_id]]
    _write_text = lambda path, text: paths.require_write_path(path).write_text(
        text, encoding="utf-8", newline="\n"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", E0_CONFIG, paths)
    _write_text(output / "run.trec", trec)
    write_jsonl(output / "top10.jsonl", top10, sort_key="query_id", paths=paths)
    write_jsonl(output / "evidence_cards.jsonl", cards, sort_key="query_id", paths=paths)
    write_json(output / "metrics.json", metrics, paths)
    write_jsonl(output / "stage_timings.jsonl", timings, sort_key="query_id", paths=paths)
    write_jsonl(output / "resource_samples.jsonl", resources, sort_key="monotonic_ns", paths=paths)
    query_latency = [
        sum(
            item["duration_ns"]
            for item in timings
            if item["query_id"] == query_id
            and item["stage"] in {"query_tokenize", "sparse_search", "evidence_card_assembly"}
        )
        for query_id in sorted(rankings)
    ]
    latency = {
        "scope": "query_tokenize+sparse_search+evidence_card_assembly",
        "downloads_included": False,
        "corpus_load_included": False,
        "index_build_included": False,
        "query_count": len(query_latency),
        **_percentiles(query_latency),
        "sampled_peak_process_tree_rss_bytes": max(item["process_tree_rss_bytes"] for item in resources),
        "build_stages_ns": {
            item["stage"]: item["duration_ns"]
            for item in timings
            if item["query_id"] is None and item["stage"] != "metrics"
        },
    }
    write_json(output / "latency.json", latency, paths)
    report = (
        "# P3a Smoke12 — C2 + E0\n\n"
        f"- Run: `{run_id}`\n"
        f"- Answerable ranking topics: {len(answerable_ids)}; no-answer diagnostics: {len(no_answer_ids)}\n"
        f"- nDCG@10: {metrics['fixed_chunk_metrics']['ndcg_at_10']:.6f}\n"
        f"- Recall@20: {metrics['fixed_chunk_metrics']['recall_at_20']:.6f}\n"
        f"- Evidence Recall@2,048 tokens: {metrics['fixed_chunk_metrics']['evidence_recall_at_2048_tokens']:.6f}\n"
        f"- Query p50/p95: {latency['p50_ms']:.3f}/{latency['p95_ms']:.3f} ms\n"
        "- Downloads, corpus loading, corpus tokenization, and index build are excluded from query latency.\n"
        "- Seed and all metrics remain provisional; this report cannot support SOTA claims.\n"
    )
    _write_text(output / "report.md", report)
    output_hashes = {
        relative: sha256_file(output / relative)
        for relative in (
            "config.json",
            "run.trec",
            "top10.jsonl",
            "evidence_cards.jsonl",
            "metrics.json",
            "latency.json",
            "stage_timings.jsonl",
            "resource_samples.jsonl",
            "report.md",
        )
    }
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "system": "E0-BM25",
        "source_snapshot_id": SNAPSHOT_ID,
        "chunk_config_id": "C2-structure-bounded",
        "chunk_config_hash": chunk_manifest["chunk_config_hash"],
        "corpus_hash": sha256_file(chunk_root / "manifest.json"),
        "benchmark_hash": sha256_file(benchmark_manifest_path),
        "config_hash": config_hash,
        "query_set": "Smoke12",
        "status": "provisional_smoke_diagnostic",
        "run_tag": RUN_TAG,
        "acquisition_included": False,
        "output_hashes": output_hashes,
    }
    validate_record("RunManifest", manifest)
    write_json(output / "manifest.json", manifest, paths)
    return {"run_id": run_id, "output": str(output), "metrics": metrics, "latency": latency}
