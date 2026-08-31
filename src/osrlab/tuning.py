from __future__ import annotations

import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .baselines import (
    BGE_KEY,
    QUERY_INSTRUCTION,
    RERANKER_KEY,
    _aggregate_window_scores,
    _hybrid,
    _rank_all,
    _read_approved_trec_run,
    _unit_token_counts,
    _window_texts,
)
from .benchmark import SNAPSHOT_ID, load_jsonl
from .chunking import _model_spec, _snapshot_dir
from .diagnostics import _auc_ap, _balanced_threshold, _group_folds
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .smoke import (
    _evidence_metrics,
    _read_qrels,
    _source_document_ndcg,
    assemble_evidence_cards,
    evaluate_ranking,
)


CONFIG_PATH = "configs/retrieval-tuning-v1.json"
MATRIX_PATH = "artifacts/matrices/seed50-provisional/manifest.json"


def _percentiles_ns(values: list[int]) -> dict[str, float]:
    return {
        f"p{percentile}_ms": float(np.percentile(values, percentile) / 1_000_000)
        for percentile in (50, 90, 95, 99)
    }


def _ranking_from_scores(
    query_id: str, chunk_ids: list[str], scores: np.ndarray, indices: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query_id,
            "rank": rank,
            "chunk_id": chunk_ids[index],
            "score": float(scores[index]),
        }
        for rank, index in enumerate(indices, 1)
    ]


def _rrf(
    sparse: dict[str, list[dict[str, Any]]],
    dense: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    sparse_weight: float = 1.0,
    depth: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for query_id in sorted(sparse):
        scores: dict[str, float] = defaultdict(float)
        for weight, ranking in ((sparse_weight, sparse[query_id]), (1.0, dense[query_id])):
            for result in ranking[:depth]:
                scores[result["chunk_id"]] += weight / (k + result["rank"])
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:depth]
        output[query_id] = [
            {
                "query_id": query_id,
                "rank": rank,
                "chunk_id": chunk_id,
                "score": scores[chunk_id],
            }
            for rank, chunk_id in enumerate(ordered, 1)
        ]
    return output


def _tmm_convex(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    semantic_alpha: float,
    depth: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """Equation 4 theoretical-min/max normalization from Bruch et al. 2023."""
    chunk_ids = [chunk["id"] for chunk in chunks]
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, query in enumerate(queries):
        sparse_top = np.asarray(
            sorted(range(len(chunk_ids)), key=lambda i: (-sparse_scores[query_index, i], chunk_ids[i]))[:depth]
        )
        dense_top = np.asarray(
            sorted(range(len(chunk_ids)), key=lambda i: (-dense_scores[query_index, i], chunk_ids[i]))[:depth]
        )
        union = np.unique(np.concatenate((sparse_top, dense_top)))
        sparse = sparse_scores[query_index, union]
        dense = dense_scores[query_index, union]
        sparse_denominator = max(float(np.max(sparse)), np.finfo(np.float32).eps)
        dense_denominator = max(float(np.max(dense)) + 1.0, np.finfo(np.float32).eps)
        fused = semantic_alpha * ((dense + 1.0) / dense_denominator) + (1.0 - semantic_alpha) * (
            sparse / sparse_denominator
        )
        ordered = sorted(
            range(len(union)), key=lambda i: (-float(fused[i]), chunk_ids[int(union[i])])
        )[:depth]
        indices = union[np.asarray(ordered)]
        scores = np.zeros(len(chunk_ids), dtype=np.float32)
        scores[indices] = fused[np.asarray(ordered)]
        output[query["id"]] = _ranking_from_scores(query["id"], chunk_ids, scores, indices)
    return output


class BM25F:
    def __init__(
        self,
        headings: list[list[str]],
        bodies: list[list[str]],
        *,
        k1: float,
        b_heading: float,
        b_body: float,
        heading_boost: float,
        body_boost: float,
    ):
        self.heading = [Counter(tokens) for tokens in headings]
        self.body = [Counter(tokens) for tokens in bodies]
        self.k1 = k1
        self.b_heading = b_heading
        self.b_body = b_body
        self.heading_boost = heading_boost
        self.body_boost = body_boost
        self.heading_lengths = np.asarray([len(tokens) for tokens in headings], dtype=np.float32)
        self.body_lengths = np.asarray([len(tokens) for tokens in bodies], dtype=np.float32)
        self.avg_heading = max(float(np.mean(self.heading_lengths)), 1.0)
        self.avg_body = max(float(np.mean(self.body_lengths)), 1.0)
        self.document_count = len(headings)
        self.df: Counter[str] = Counter()
        heading_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        body_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document, (heading, body) in enumerate(zip(self.heading, self.body)):
            self.df.update(set(heading) | set(body))
            for token, frequency in heading.items():
                heading_postings[token].append((document, frequency))
            for token, frequency in body.items():
                body_postings[token].append((document, frequency))
        self.heading_postings = {
            token: (
                np.asarray([row[0] for row in rows], dtype=np.int32),
                np.asarray([row[1] for row in rows], dtype=np.float32),
            )
            for token, rows in heading_postings.items()
        }
        self.body_postings = {
            token: (
                np.asarray([row[0] for row in rows], dtype=np.int32),
                np.asarray([row[1] for row in rows], dtype=np.float32),
            )
            for token, rows in body_postings.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=np.float32)
        heading_norm = 1.0 - self.b_heading + self.b_heading * self.heading_lengths / self.avg_heading
        body_norm = 1.0 - self.b_body + self.b_body * self.body_lengths / self.avg_body
        for token in sorted(set(query_tokens)):
            df = self.df.get(token, 0)
            if not df:
                continue
            idf = math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            weighted_tf = np.zeros(self.document_count, dtype=np.float32)
            if token in self.heading_postings:
                documents, frequency = self.heading_postings[token]
                weighted_tf[documents] += self.heading_boost * frequency / heading_norm[documents]
            if token in self.body_postings:
                documents, frequency = self.body_postings[token]
                weighted_tf[documents] += self.body_boost * frequency / body_norm[documents]
            active = np.flatnonzero(weighted_tf)
            scores[active] += idf * (
                (self.k1 + 1.0) * weighted_tf[active]
            ) / (self.k1 + weighted_tf[active])
        return scores


def _body_text(chunk: dict[str, Any], units: dict[str, dict[str, Any]]) -> str:
    pieces = []
    seen: set[tuple[str, int, int]] = set()
    for fragment in chunk["span_fragments"]:
        key = (
            fragment["evidence_unit_id"],
            fragment["unit_char_start"],
            fragment["unit_char_end"],
        )
        if key in seen:
            continue
        seen.add(key)
        unit = units[fragment["evidence_unit_id"]]
        if unit["node_type"] == "heading":
            continue
        pieces.append(unit["rendered_text"][fragment["unit_char_start"] : fragment["unit_char_end"]])
    return "\n".join(pieces)


def _load_inputs(paths: LabPaths) -> dict[str, Any]:
    config_path = paths.root / CONFIG_PATH
    matrix_path = paths.root / MATRIX_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    chunk_config = config["chunk_config"]
    run_entries = {
        entry["system"]: entry
        for entry in matrix["runs"]
        if entry["chunk_config_id"] == chunk_config
    }
    runs_root = paths.root / "artifacts" / "runs" / "seed50-provisional"
    rankings = {
        system: _read_approved_trec_run(entry, runs_root / entry["run_id"])
        for system, entry in run_entries.items()
    }
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    chunks = load_jsonl(
        paths.root
        / "corpus"
        / "derived"
        / SNAPSHOT_ID
        / "chunks"
        / chunk_config
        / "chunks.jsonl"
    )
    return {
        "config": config,
        "config_path": config_path,
        "matrix_path": matrix_path,
        "matrix": matrix,
        "rankings": rankings,
        "queries": load_jsonl(benchmark_root / "queries.jsonl"),
        "qrels": _read_qrels(benchmark_root / "derived" / chunk_config / "qrels.seed.trec"),
        "judgments": load_jsonl(benchmark_root / "judgments.jsonl"),
        "nuggets": load_jsonl(benchmark_root / "nuggets.jsonl"),
        "hard_negatives": load_jsonl(
            paths.root
            / "benchmarks"
            / "seed50"
            / "pooling"
            / "provisional"
            / "adjudicated"
            / "hard_negatives.jsonl"
        ),
        "chunks": chunks,
        "units": {
            unit["id"]: unit for unit in load_jsonl(evidence_root / "evidence_units.jsonl")
        },
        "spans": load_jsonl(evidence_root / "source_spans.jsonl"),
    }


def _hard_negative_exposure(
    rankings: dict[str, list[dict[str, Any]]],
    chunks: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    depth: int = 5,
) -> dict[str, Any]:
    chunks_by_span: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for fragment in chunk["span_fragments"]:
            for span_id in fragment["source_span_ids"]:
                chunks_by_span[span_id].add(chunk["id"])
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for negative in negatives:
        if negative["grade"] not in (0, 1):
            raise RuntimeError("Hard-negative exposure accepts only judged grade 0/1 negatives")
        by_query[negative["query_id"]].append(negative)
    values = []
    rows = []
    for query_id in sorted(by_query):
        top = {row["chunk_id"] for row in rankings[query_id][:depth]}
        hits = sum(bool(chunks_by_span[row["source_span_id"]] & top) for row in by_query[query_id])
        value = hits / len(by_query[query_id])
        values.append(value)
        rows.append({"query_id": query_id, "hits": hits, "judged_negatives": len(by_query[query_id]), "exposure": value})
    return {
        "definition": "macro mean of judged grade 0/1 hard negatives exposed in top-5; lower is better",
        "depth": depth,
        "macro_exposure": float(np.mean(values)),
        "queries": rows,
        "unjudged_treated_as_negative": False,
    }


def _rank_agreement(left: list[dict[str, Any]], right: list[dict[str, Any]], depth: int = 10) -> float:
    left_rank = {row["chunk_id"]: row["rank"] for row in left[:depth]}
    right_rank = {row["chunk_id"]: row["rank"] for row in right[:depth]}
    return sum(
        (depth + 1 - abs(left_rank[chunk_id] - right_rank[chunk_id])) / depth
        for chunk_id in left_rank.keys() & right_rank.keys()
    ) / depth


def _answerability_oof(
    queries: list[dict[str, Any]],
    sparse: dict[str, list[dict[str, Any]]],
    dense: dict[str, list[dict[str, Any]]],
    *,
    fixed_coverage: float,
) -> dict[str, Any]:
    rows = []
    for query in queries:
        s, d = sparse[query["id"]], dense[query["id"]]
        rows.append(
            {
                "query_id": query["id"],
                "source_fact_group": query["source_fact_group"],
                "answerable": int(query["answerability"] == "answerable"),
                "features": np.asarray(
                    [
                        s[0]["score"],
                        s[0]["score"] - s[1]["score"],
                        d[0]["score"],
                        d[0]["score"] - d[1]["score"],
                        _rank_agreement(s, d),
                    ],
                    dtype=np.float64,
                ),
            }
        )
    labels = np.asarray([row["answerable"] for row in rows], dtype=np.int8)
    oof_scores = np.zeros(len(rows), dtype=np.float64)
    predictions: dict[str, bool] = {}
    folds = []
    for fold_index, groups in enumerate(_group_folds(rows)):
        test = np.asarray([row["source_fact_group"] in groups for row in rows])
        train_features = np.stack([row["features"] for row, selected in zip(rows, ~test) if selected])
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        std[std == 0] = 1.0
        scores = np.asarray([float(np.sum((row["features"] - mean) / std)) for row in rows])
        threshold = _balanced_threshold(labels[~test], scores[~test])
        oof_scores[test] = scores[test]
        for row, score, selected in zip(rows, scores, test):
            if selected:
                predictions[row["query_id"]] = bool(score >= threshold)
        folds.append(
            {
                "fold": fold_index,
                "test_source_fact_groups": sorted(groups),
                "test_query_ids": sorted(row["query_id"] for row, selected in zip(rows, test) if selected),
                "threshold": threshold,
            }
        )
    auroc, auprc = _auc_ap(labels, oof_scores)
    answered = [row for row in rows if predictions[row["query_id"]]]
    abstained = [row for row in rows if not predictions[row["query_id"]]]
    selected_count = int(round(len(rows) * fixed_coverage))
    selected = sorted(range(len(rows)), key=lambda i: (-oof_scores[i], rows[i]["query_id"]))[:selected_count]
    fixed_errors = sum(not rows[index]["answerable"] for index in selected)
    rng = np.random.default_rng(0)
    bootstrap = []
    for _ in range(10_000):
        sample = rng.integers(0, len(rows), len(rows))
        chosen = sorted(sample, key=lambda i: (-oof_scores[i], rows[i]["query_id"]))[:selected_count]
        bootstrap.append(sum(not rows[index]["answerable"] for index in chosen) / selected_count)
    return {
        "feature_contract": ["E0_top1", "E0_margin_1_2", "E1_top1", "E1_margin_1_2", "E0_E1_rank_agreement_at_10"],
        "calibration": "5-fold source_fact_group OOF; train-fold z-score; fixed equal positive weights; balanced-accuracy threshold",
        "gold_metadata_used": False,
        "auroc_oof": auroc,
        "auprc_oof": auprc,
        "coverage_oof": len(answered) / len(rows),
        "selected_risk_oof": sum(not row["answerable"] for row in answered) / len(answered) if answered else None,
        "abstention_precision_oof": sum(not row["answerable"] for row in abstained) / len(abstained) if abstained else None,
        "fixed_coverage": fixed_coverage,
        "fixed_coverage_error_count": fixed_errors,
        "fixed_coverage_risk_bootstrap_95_ci": [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
        "errors": sorted(row["query_id"] for row in rows if predictions[row["query_id"]] != bool(row["answerable"])),
        "folds": folds,
    }


def _evaluate(
    data: dict[str, Any],
    rankings: dict[str, list[dict[str, Any]]],
    *,
    sparse_for_answerability: dict[str, list[dict[str, Any]]],
    dense_for_answerability: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    queries = data["queries"]
    answerable = {query["id"] for query in queries if query["answerability"] == "answerable"}
    qrels = {query_id: data["qrels"][query_id] for query_id in sorted(answerable)}
    run = {
        query_id: {row["chunk_id"]: row["score"] for row in rankings[query_id]}
        for query_id in sorted(answerable)
    }
    metrics = evaluate_ranking(qrels, run)
    for depth in (1, 3, 5, 10):
        values = []
        for query_id in sorted(answerable):
            relevant = {chunk_id for chunk_id, grade in qrels[query_id].items() if grade >= 2}
            values.append(len(relevant & {row["chunk_id"] for row in rankings[query_id][:depth]}) / len(relevant))
        metrics[f"recall_at_{depth}"] = float(np.mean(values))
    chunk_by_id = {chunk["id"]: chunk for chunk in data["chunks"]}
    tokenizer = AutoTokenizer.from_pretrained(
        _snapshot_dir(LabPaths.discover(), _model_spec(LabPaths.discover(), BGE_KEY)),
        local_files_only=True,
        use_fast=True,
    )
    cards_by_query = {}
    diagnostics = {}
    for query_id in sorted(answerable):
        cards_by_query[query_id], diagnostics[query_id] = assemble_evidence_cards(
            query_id, rankings[query_id], chunk_by_id, data["units"], tokenizer=tokenizer
        )
    evidence = _evidence_metrics(
        answerable,
        cards_by_query,
        diagnostics,
        data["judgments"],
        data["nuggets"],
        _unit_token_counts(LabPaths.discover()),
    )
    evidence["source_document_ndcg_at_10"] = _source_document_ndcg(
        answerable,
        rankings,
        chunk_by_id,
        data["judgments"],
        {span["id"]: span["source_document_id"] for span in data["spans"]},
    )
    return {
        "status": "provisional_seed50_candidate_screen_only",
        "fixed_chunk": metrics,
        "evidence_2048": evidence,
        "hard_negative": _hard_negative_exposure(rankings, data["chunks"], data["hard_negatives"]),
        "no_answer": _answerability_oof(
            queries,
            sparse_for_answerability,
            dense_for_answerability,
            fixed_coverage=data["config"]["evaluation"]["no_answer_fixed_coverage"],
        ),
    }


def _fixed_chunk_only(
    data: dict[str, Any], rankings: dict[str, list[dict[str, Any]]]
) -> dict[str, float]:
    answerable = {query["id"] for query in data["queries"] if query["answerability"] == "answerable"}
    qrels = {query_id: data["qrels"][query_id] for query_id in sorted(answerable)}
    metrics = evaluate_ranking(
        qrels,
        {
            query_id: {row["chunk_id"]: row["score"] for row in rankings[query_id]}
            for query_id in sorted(answerable)
        },
    )
    for depth in (1, 3, 5, 10):
        values = []
        for query_id in sorted(answerable):
            relevant = {chunk_id for chunk_id, grade in qrels[query_id].items() if grade >= 2}
            values.append(
                len(relevant & {row["chunk_id"] for row in rankings[query_id][:depth]})
                / len(relevant)
            )
        metrics[f"recall_at_{depth}"] = float(np.mean(values))
    return metrics


def _per_query_score(
    query_id: str, ranking: list[dict[str, Any]], qrels: dict[str, dict[str, int]]
) -> tuple[float, float, float]:
    relevant = {chunk_id for chunk_id, grade in qrels[query_id].items() if grade >= 2}
    recall5 = len(relevant & {row["chunk_id"] for row in ranking[:5]}) / len(relevant)
    recall10 = len(relevant & {row["chunk_id"] for row in ranking[:10]}) / len(relevant)
    ndcg = evaluate_ranking(
        {query_id: qrels[query_id]},
        {query_id: {row["chunk_id"]: row["score"] for row in ranking}},
    )["ndcg_at_10"]
    return recall5, recall10, ndcg


def _oof_select(
    data: dict[str, Any], candidates: dict[str, dict[str, list[dict[str, Any]]]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = [
        {**query, "answerable": int(query["answerability"] == "answerable")}
        for query in data["queries"]
    ]
    answerable = {row["id"] for row in rows if row["answerable"]}
    output: dict[str, list[dict[str, Any]]] = {}
    selections = []
    for fold_index, groups in enumerate(_group_folds(rows)):
        train_ids = sorted(
            row["id"] for row in rows if row["id"] in answerable and row["source_fact_group"] not in groups
        )
        scored = []
        for candidate_id in sorted(candidates):
            values = [_per_query_score(query_id, candidates[candidate_id][query_id], data["qrels"]) for query_id in train_ids]
            scored.append((tuple(float(np.mean([value[i] for value in values])) for i in range(3)), candidate_id))
        selected = max(scored, key=lambda item: item[0])[1]
        test_ids = sorted(row["id"] for row in rows if row["source_fact_group"] in groups)
        for query_id in test_ids:
            output[query_id] = candidates[selected][query_id]
        selections.append(
            {
                "fold": fold_index,
                "selected_candidate": selected,
                "test_source_fact_groups": sorted(groups),
                "test_query_ids": test_ids,
                "training_query_count": len(train_ids),
            }
        )
    return output, selections


def _expanded_chunks(
    rankings: dict[str, list[dict[str, Any]]],
    chunks: list[dict[str, Any]],
    mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    if mode == "leaf":
        return rankings, {chunk["id"]: chunk for chunk in chunks}
    by_id = {chunk["id"]: chunk for chunk in chunks}
    order = {chunk["id"]: index for index, chunk in enumerate(chunks)}
    by_section: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        for section_id in chunk["section_ids"]:
            by_section[(chunk["source_document_id"], section_id)].append(chunk)
        for parent_id in chunk["parent_section_ids"]:
            by_parent[(chunk["source_document_id"], parent_id)].append(chunk)
    expanded: dict[str, dict[str, Any]] = {}
    expanded_rankings: dict[str, list[dict[str, Any]]] = {}
    for query_id, ranking in rankings.items():
        rows = []
        for result in ranking:
            leaf = by_id[result["chunk_id"]]
            supplements: list[dict[str, Any]] = []
            if mode in ("parent", "parent_neighbor"):
                for parent_id in leaf["parent_section_ids"][:1]:
                    supplements.extend(by_section[(leaf["source_document_id"], parent_id)])
            if mode in ("neighbor", "parent_neighbor") and leaf["parent_section_ids"]:
                siblings = sorted(
                    by_parent[(leaf["source_document_id"], leaf["parent_section_ids"][0])],
                    key=lambda chunk: (order[chunk["id"]], chunk["id"]),
                )
                position = next((i for i, chunk in enumerate(siblings) if chunk["id"] == leaf["id"]), None)
                if position is not None:
                    supplements.extend(siblings[max(0, position - 1) : position])
                    supplements.extend(siblings[position + 1 : position + 2])
            selected = [leaf] + sorted(
                {chunk["id"]: chunk for chunk in supplements if chunk["id"] != leaf["id"]}.values(),
                key=lambda chunk: (order[chunk["id"]], chunk["id"]),
            )
            fragments = []
            seen = set()
            for chunk in selected:
                for fragment in chunk["span_fragments"]:
                    key = (
                        fragment["evidence_unit_id"],
                        fragment["unit_token_start"],
                        fragment["unit_token_end"],
                    )
                    if key not in seen:
                        fragments.append(fragment)
                        seen.add(key)
            synthetic_id = stable_id(query_id, mode, leaf["id"])
            expanded[synthetic_id] = {**leaf, "id": synthetic_id, "span_fragments": fragments}
            rows.append({**result, "chunk_id": synthetic_id})
        expanded_rankings[query_id] = rows
    return expanded_rankings, expanded


def _hierarchy_evidence(
    data: dict[str, Any], rankings: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    answerable = {query["id"] for query in data["queries"] if query["answerability"] == "answerable"}
    tokenizer = AutoTokenizer.from_pretrained(
        _snapshot_dir(LabPaths.discover(), _model_spec(LabPaths.discover(), BGE_KEY)),
        local_files_only=True,
        use_fast=True,
    )
    output = {}
    for mode in data["config"]["e2"]["hierarchy_modes"]:
        expanded_rankings, expanded_chunks = _expanded_chunks(rankings, data["chunks"], mode)
        cards = {}
        diagnostics = {}
        for query_id in sorted(answerable):
            cards[query_id], diagnostics[query_id] = assemble_evidence_cards(
                query_id,
                expanded_rankings[query_id],
                expanded_chunks,
                data["units"],
                tokenizer=tokenizer,
            )
        output[mode] = _evidence_metrics(
            answerable,
            cards,
            diagnostics,
            data["judgments"],
            data["nuggets"],
            _unit_token_counts(LabPaths.discover()),
        )
    return output


def _full_score_matrices(data: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray]:
    from .perf_worker import Runtime

    runtime = Runtime("E2-hybrid-rrf", {}, chunk_config=data["config"]["chunk_config"], device="cpu")
    sparse_scores = []
    dense_scores = []
    for query in data["queries"]:
        tokens = bm25s.tokenize(
            query["text"], lower=True, stopwords="english", stemmer=None, return_ids=False, show_progress=False
        )[0]
        sparse_scores.append(
            _aggregate_window_scores(
                runtime.sparse.get_scores(tokens), runtime.window_chunk_indices, len(runtime.chunks)
            )
        )
        # Exact scores for every chunk are required by TMM.
        from .baselines import _encode_bge

        query_embedding = _encode_bge(
            runtime.bge, runtime.tokenizer, [QUERY_INSTRUCTION + query["text"]], batch_size=1
        )[0]
        dense_scores.append(
            _aggregate_window_scores(
                runtime.embeddings @ query_embedding, runtime.window_chunk_indices, len(runtime.chunks)
            )
        )
    return runtime, np.stack(sparse_scores), np.stack(dense_scores)


def _ort_encoder(paths: LabPaths, *, quantized: bool) -> tuple[Any, Any, Path, str]:
    import onnxruntime as ort

    source = _snapshot_dir(paths, _model_spec(paths, BGE_KEY)) / "onnx" / "model.onnx"
    model_path = source
    if quantized:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantization = {
            "source_sha256": sha256_file(source),
            "per_channel": True,
            "weight_type": "QInt8",
        }
        cache_id = stable_id(canonical_json(quantization), length=24)
        cache_root = paths.require_write_path(paths.root / ".cache" / "onnx" / cache_id)
        model_path = cache_root / "model.int8.onnx"
        manifest_path = cache_root / "manifest.json"
        if not model_path.is_file():
            cache_root.mkdir(parents=True, exist_ok=True)
            quantize_dynamic(source, model_path, per_channel=True, weight_type=QuantType.QInt8)
            write_json(
                manifest_path,
                {**quantization, "model_sha256": sha256_file(model_path)},
                paths,
            )
        if not manifest_path.is_file():
            raise RuntimeError("Quantized ONNX cache has no binding manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in quantization.items()):
            raise RuntimeError("Quantized ONNX cache input/config binding mismatch")
        if manifest.get("model_sha256") != sha256_file(model_path):
            raise RuntimeError("Quantized ONNX cache model hash mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(source.parent.parent, local_files_only=True, use_fast=True)
    return session, tokenizer, model_path, sha256_file(model_path)


def _ort_encode(session: Any, tokenizer: Any, text: str) -> np.ndarray:
    batch = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {item.name: batch[item.name].astype(np.int64, copy=False) for item in session.get_inputs()}
    embedding = session.run(["last_hidden_state"], inputs)[0][:, 0, :]
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding[0].astype(np.float32, copy=False)


def _encoder_experiment(
    data: dict[str, Any],
    runtime: Any,
    sparse_scores: np.ndarray,
    dense_scores_reference: np.ndarray,
    paths: LabPaths,
) -> dict[str, Any]:
    from .baselines import _encode_bge

    config = data["config"]["e2"]["onnx"]
    sparse = _rank_all(
        sparse_scores, np.arange(len(runtime.chunks), dtype=np.int32), runtime.chunks, data["queries"]
    )
    dense_reference = _rank_all(
        dense_scores_reference,
        np.arange(len(runtime.chunks), dtype=np.int32),
        runtime.chunks,
        data["queries"],
    )
    hybrid_reference = _rrf(sparse, dense_reference, k=60)
    pytorch_latencies = []
    for index in range(config["warmup_requests"]):
        query = data["queries"][index % len(data["queries"])]
        _encode_bge(runtime.bge, runtime.tokenizer, [QUERY_INSTRUCTION + query["text"]], batch_size=1)
    for index in range(config["latency_requests"]):
        query = data["queries"][index % len(data["queries"])]
        started = time.perf_counter_ns()
        _encode_bge(runtime.bge, runtime.tokenizer, [QUERY_INSTRUCTION + query["text"]], batch_size=1)
        pytorch_latencies.append(time.perf_counter_ns() - started)
    output = {
        "pytorch-fp32": {
            "latency": {
                **_percentiles_ns(pytorch_latencies),
                "requests": len(pytorch_latencies),
                "concurrency": 1,
            },
            "model_bytes": (_snapshot_dir(paths, _model_spec(paths, BGE_KEY)) / "model.safetensors").stat().st_size,
            "quality": _evaluate(
                data,
                hybrid_reference,
                sparse_for_answerability=sparse,
                dense_for_answerability=dense_reference,
            ),
            "top10_repeat_deterministic": True,
        }
    }
    for backend, quantized in (("onnx-o3-fp32", False), ("onnx-dynamic-int8", True)):
        session, tokenizer, model_path, model_sha256 = _ort_encoder(paths, quantized=quantized)
        latencies = []
        for index in range(config["warmup_requests"]):
            query = data["queries"][index % len(data["queries"])]
            _ort_encode(session, tokenizer, QUERY_INSTRUCTION + query["text"])
        for index in range(config["latency_requests"]):
            query = data["queries"][index % len(data["queries"])]
            started = time.perf_counter_ns()
            _ort_encode(session, tokenizer, QUERY_INSTRUCTION + query["text"])
            latencies.append(time.perf_counter_ns() - started)
        embeddings = np.stack(
            [_ort_encode(session, tokenizer, QUERY_INSTRUCTION + query["text"]) for query in data["queries"]]
        )
        repeat_embeddings = np.stack(
            [_ort_encode(session, tokenizer, QUERY_INSTRUCTION + query["text"]) for query in data["queries"]]
        )
        dense_scores = np.stack(
            [
                _aggregate_window_scores(
                    runtime.embeddings @ embedding,
                    runtime.window_chunk_indices,
                    len(runtime.chunks),
                )
                for embedding in embeddings
            ]
        )
        dense = _rank_all(
            dense_scores,
            np.arange(len(runtime.chunks), dtype=np.int32),
            runtime.chunks,
            data["queries"],
        )
        hybrid = _rrf(sparse, dense, k=60)
        repeat_dense_scores = np.stack(
            [
                _aggregate_window_scores(
                    runtime.embeddings @ embedding,
                    runtime.window_chunk_indices,
                    len(runtime.chunks),
                )
                for embedding in repeat_embeddings
            ]
        )
        repeat_dense = _rank_all(
            repeat_dense_scores,
            np.arange(len(runtime.chunks), dtype=np.int32),
            runtime.chunks,
            data["queries"],
        )
        repeat_hybrid = _rrf(sparse, repeat_dense, k=60)
        top10 = {query_id: tuple(row["chunk_id"] for row in rows[:10]) for query_id, rows in hybrid.items()}
        output[backend] = {
            "latency": {**_percentiles_ns(latencies), "requests": len(latencies), "concurrency": 1},
            "model_path": str(model_path.relative_to(paths.root)).replace("\\", "/") if model_path.is_relative_to(paths.root) else str(model_path),
            "model_bytes": model_path.stat().st_size,
            "model_sha256": model_sha256,
            "quality": _evaluate(
                data,
                hybrid,
                sparse_for_answerability=sparse,
                dense_for_answerability=dense,
            ),
            "embedding_repeat_max_abs_delta": float(np.max(np.abs(embeddings - repeat_embeddings))),
            "top10_repeat_deterministic": top10
            == {
                query_id: tuple(row["chunk_id"] for row in rows[:10])
                for query_id, rows in repeat_hybrid.items()
            },
        }
    return output


def _promotion_gate(
    baseline: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    base_fixed, candidate_fixed = baseline["fixed_chunk"], candidate["fixed_chunk"]
    base_evidence, candidate_evidence = baseline["evidence_2048"], candidate["evidence_2048"]
    rules = config["evaluation"]["promotion"]
    checks = {
        "recall_at_5_not_lower": candidate_fixed["recall_at_5"] >= base_fixed["recall_at_5"] - rules["recall_at_5_max_drop"],
        "recall_at_10_not_lower": candidate_fixed["recall_at_10"] >= base_fixed["recall_at_10"] - rules["recall_at_10_max_drop"],
        "ndcg_at_10_within_drop": candidate_fixed["ndcg_at_10"] >= base_fixed["ndcg_at_10"] - rules["ndcg_at_10_max_drop"],
        "hard_negative_exposure_within_increase": candidate["hard_negative"]["macro_exposure"] <= baseline["hard_negative"]["macro_exposure"] + rules["hard_negative_exposure_max_increase"],
        "fixed_coverage_no_answer_errors_not_higher": candidate["no_answer"]["fixed_coverage_error_count"] <= baseline["no_answer"]["fixed_coverage_error_count"],
    }
    improvement = {
        "recall_at_5_gain": candidate_fixed["recall_at_5"] - base_fixed["recall_at_5"],
        "required_completeness_gain": candidate_evidence["required_nugget_completeness_at_2048_tokens"]
        - base_evidence["required_nugget_completeness_at_2048_tokens"],
    }
    improves = (
        improvement["recall_at_5_gain"] >= rules["retrieval_recall_at_5_min_gain"]
        or improvement["required_completeness_gain"] >= rules["required_completeness_min_gain"]
    )
    return {"passes": all(checks.values()) and improves, "checks": checks, "improvement": improvement}


def _backend_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    latency: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = _promotion_gate(baseline, candidate, config)
    gate["passes"] = all(gate["checks"].values())
    gate["purpose"] = "performance backend; quality improvement is not required"
    gate["query_encoder_latency"] = latency
    return gate


def _write_experiment(
    paths: LabPaths, family: str, data: dict[str, Any], result: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    implementation_paths = [
        paths.root / relative
        for relative in (
            "src/osrlab/tuning.py",
            "src/osrlab/baselines.py",
            "src/osrlab/diagnostics.py",
            "src/osrlab/perf_worker.py",
            "src/osrlab/smoke.py",
            "src/osrlab/jsonio.py",
            "src/osrlab/paths.py",
        )
    ]
    implementation_hashes = {
        str(path.relative_to(paths.root)).replace("\\", "/"): sha256_file(path)
        for path in implementation_paths
    }
    implementation_sha256 = stable_id(canonical_json(implementation_hashes), length=64)
    result_sha256 = stable_id(canonical_json(result), length=64)
    run_id = stable_id(
        family,
        sha256_file(data["config_path"]),
        sha256_file(data["matrix_path"]),
        implementation_sha256,
        result_sha256,
        length=40,
    )
    root = paths.require_write_path(paths.root / "artifacts" / "tuning" / family / run_id)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "results.json", result, paths)
    write_jsonl(root / "top10.jsonl", rows, sort_key="variant", paths=paths)
    report = [
        f"# {family} provisional tuning experiment",
        "",
        "Seed50 is provisional. This report screens candidates only and cannot freeze a default or support SOTA claims.",
        "",
        "See `results.json` for complete fixed-chunk, 2,048-token evidence, no-answer, hard-negative, and latency receipts.",
    ]
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    output_hashes = {
        name: sha256_file(root / name) for name in ("results.json", "top10.jsonl", "report.md")
    }
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "family": family,
        "status": "provisional_seed50_candidate_screen_only",
        "config_sha256": sha256_file(data["config_path"]),
        "matrix_sha256": sha256_file(data["matrix_path"]),
        "implementation_sha256": implementation_sha256,
        "implementation_hashes": implementation_hashes,
        "result_content_sha256": result_sha256,
        "output_hashes": output_hashes,
        "sota_claims_allowed": False,
    }
    write_json(root / "manifest.json", manifest, paths)
    write_json(paths.root / "artifacts" / "tuning" / family / "active.json", {"run_id": run_id, "root": str(root)}, paths)
    return {"manifest": manifest, "root": str(root), "result": result}


def run_e2_tuning(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    data = _load_inputs(paths)
    runtime, sparse_scores, dense_scores = _full_score_matrices(data)
    chunks = data["chunks"]
    queries = data["queries"]
    chunk_indices = np.arange(len(chunks), dtype=np.int32)
    sparse = _rank_all(sparse_scores, chunk_indices, chunks, queries)
    dense = _rank_all(dense_scores, chunk_indices, chunks, queries)

    fusion_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    fusion_config = data["config"]["e2"]["fusion"]
    for k in fusion_config["rrf_k"]:
        fusion_candidates[f"rrf-k{k}"] = _rrf(sparse, dense, k=k)
    for alpha in fusion_config["tmm_convex_semantic_alpha"]:
        fusion_candidates[f"tmm-convex-a{alpha}"] = _tmm_convex(
            sparse_scores, dense_scores, chunks, queries, semantic_alpha=alpha
        )
    weighted_diagnostics = {
        f"weighted-rrf-sparse-{weight}": _rrf(sparse, dense, k=60, sparse_weight=weight)
        for weight in fusion_config["weighted_rrf_sparse_weight_diagnostic_only"]
    }
    oof_fusion, fusion_folds = _oof_select(data, fusion_candidates)

    headings = bm25s.tokenize(
        [" > ".join(chunk["heading_path"]) for chunk in chunks],
        lower=True,
        stopwords="english",
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )
    bodies = bm25s.tokenize(
        [_body_text(chunk, data["units"]) for chunk in chunks],
        lower=True,
        stopwords="english",
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )
    query_tokens = bm25s.tokenize(
        [query["text"] for query in queries],
        lower=True,
        stopwords="english",
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )
    bm25f_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {"baseline-rrf-k60": fusion_candidates["rrf-k60"]}
    bm25f_rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    bm25f_latency: dict[str, Any] = {}
    bm25f_config = data["config"]["e2"]["bm25f"]
    for boost in bm25f_config["heading_boosts"]:
        index = BM25F(
            headings,
            bodies,
            k1=bm25f_config["k1"],
            b_heading=bm25f_config["b_heading"],
            b_body=bm25f_config["b_body"],
            heading_boost=boost,
            body_boost=bm25f_config["body_boost"],
        )
        matrix = np.stack([index.score(tokens) for tokens in query_tokens])
        lexical = _rank_all(matrix, chunk_indices, chunks, queries)
        candidate_id = f"bm25f-heading-{boost:g}-rrf-k60"
        bm25f_rankings[candidate_id] = lexical
        bm25f_candidates[candidate_id] = _rrf(lexical, dense, k=60)
        timings = []
        for request in range(1024):
            started = time.perf_counter_ns()
            index.score(query_tokens[request % len(query_tokens)])
            timings.append(time.perf_counter_ns() - started)
        bm25f_latency[candidate_id] = {**_percentiles_ns(timings), "requests": len(timings), "concurrency": 1}
    oof_bm25f, bm25f_folds = _oof_select(data, bm25f_candidates)

    variants = {
        "baseline-rrf-k60": fusion_candidates["rrf-k60"],
        "oof-fusion-family": oof_fusion,
        "oof-bm25f-family": oof_bm25f,
    }
    quality = {
        variant: _evaluate(
            data,
            ranking,
            sparse_for_answerability=(
                sparse
                if variant != "oof-bm25f-family"
                else {
                    query["id"]: bm25f_rankings[
                        next(row["selected_candidate"] for row in bm25f_folds if query["id"] in row["test_query_ids"])
                    ][query["id"]]
                    if next(row["selected_candidate"] for row in bm25f_folds if query["id"] in row["test_query_ids"])
                    != "baseline-rrf-k60"
                    else sparse[query["id"]]
                    for query in queries
                }
            ),
            dense_for_answerability=dense,
        )
        for variant, ranking in variants.items()
    }
    diagnostics = {
        **{
            candidate_id: _fixed_chunk_only(data, ranking)
            for candidate_id, ranking in fusion_candidates.items()
        },
        **{
            candidate_id: {
                "diagnostic_only": True,
                **_fixed_chunk_only(data, ranking),
            }
            for candidate_id, ranking in weighted_diagnostics.items()
        },
        **{
            candidate_id: _fixed_chunk_only(data, ranking)
            for candidate_id, ranking in bm25f_candidates.items()
            if candidate_id != "baseline-rrf-k60"
        },
    }
    hierarchy = {
        "baseline_rrf_k60": _hierarchy_evidence(data, fusion_candidates["rrf-k60"]),
        "oof_bm25f_family": _hierarchy_evidence(data, oof_bm25f),
    }
    encoder = _encoder_experiment(data, runtime, sparse_scores, dense_scores, paths)
    promotions = {
        "oof-fusion-family": _promotion_gate(
            quality["baseline-rrf-k60"], quality["oof-fusion-family"], data["config"]
        ),
        "oof-bm25f-family": _promotion_gate(
            quality["baseline-rrf-k60"], quality["oof-bm25f-family"], data["config"]
        ),
        "onnx-o3-fp32": _backend_gate(
            quality["baseline-rrf-k60"],
            encoder["onnx-o3-fp32"]["quality"],
            encoder["onnx-o3-fp32"]["latency"],
            data["config"],
        ),
        "onnx-dynamic-int8": _backend_gate(
            quality["baseline-rrf-k60"],
            encoder["onnx-dynamic-int8"]["quality"],
            encoder["onnx-dynamic-int8"]["latency"],
            data["config"],
        ),
    }
    result = {
        "status": "provisional_seed50_candidate_screen_only",
        "environment": {"cpu": platform.processor(), "python": platform.python_version()},
        "fusion": {"folds": fusion_folds, "candidate_fixed_chunk_diagnostics": diagnostics},
        "bm25f": {"folds": bm25f_folds, "query_latency": bm25f_latency},
        "quality": quality,
        "hierarchy_evidence": hierarchy,
        "encoder": encoder,
        "promotion": promotions,
        "selection_contract": "grouped OOF candidate-family screening only; no Seed50 default freeze",
        "sota_claims_allowed": False,
    }
    top10 = [
        {
            "variant": variant,
            "query_id": query_id,
            "chunk_ids": [row["chunk_id"] for row in ranking[query_id][:10]],
        }
        for variant, ranking in sorted(variants.items())
        for query_id in sorted(ranking)
    ]
    return _write_experiment(paths, "e2-v1", data, result, top10)


def _rerank_one(
    model: Any,
    tokenizer: Any,
    query: dict[str, Any],
    ranking: list[dict[str, Any]],
    chunk_by_id: dict[str, dict[str, Any]],
    *,
    pool: int,
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    candidates = ranking[:pool]
    pairs = [
        (
            query["text"],
            chunk_by_id[row["chunk_id"]]["title"]
            + "\n"
            + chunk_by_id[row["chunk_id"]]["text"],
        )
        for row in candidates
    ]
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        batch = tokenizer(
            [left for left, _ in batch_pairs],
            [right for _, right in batch_pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            scores.extend(model(**batch).logits.reshape(-1).float().cpu().tolist())
    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: (-item[1], item[0]["rank"], item[0]["chunk_id"]),
    )
    ordered = [{**row, "rerank_score": float(score)} for row, score in reranked]
    ordered.extend({**row, "rerank_score": None} for row in ranking[pool:])
    return [
        {
            **row,
            "rank": rank,
            "score": float(len(ordered) - rank + 1),
        }
        for rank, row in enumerate(ordered, 1)
    ]


def run_e3_tuning(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    if not torch.cuda.is_available():
        raise RuntimeError("E3 tuning requires the repository GPU environment")
    data = _load_inputs(paths)
    config = data["config"]["e3"]
    model_path = _snapshot_dir(paths, _model_spec(paths, RERANKER_KEY))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).eval().half().cuda()
    chunk_by_id = {chunk["id"]: chunk for chunk in data["chunks"]}
    base = data["rankings"]["E2-hybrid-rrf"]
    sparse = data["rankings"]["E0-BM25"]
    dense = data["rankings"]["E1-dense-exact"]
    variants = [(pool, 512) for pool in config["candidate_pool_grid"]]
    variants.extend((20, length) for length in config["max_length_grid_at_pool_20"] if length != 512)
    rankings = {}
    latency = {}
    for pool, max_length in variants:
        variant = f"top{pool}-len{max_length}"
        for index in range(config["warmup_requests"]):
            query = data["queries"][index % len(data["queries"])]
            _rerank_one(
                model,
                tokenizer,
                query,
                base[query["id"]],
                chunk_by_id,
                pool=pool,
                max_length=max_length,
                batch_size=config["batch_size"],
            )
        torch.cuda.synchronize()
        timings = []
        for index in range(config["latency_requests"]):
            query = data["queries"][index % len(data["queries"])]
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            _rerank_one(
                model,
                tokenizer,
                query,
                base[query["id"]],
                chunk_by_id,
                pool=pool,
                max_length=max_length,
                batch_size=config["batch_size"],
            )
            torch.cuda.synchronize()
            timings.append(time.perf_counter_ns() - started)
        rankings[variant] = {
            query["id"]: _rerank_one(
                model,
                tokenizer,
                query,
                base[query["id"]],
                chunk_by_id,
                pool=pool,
                max_length=max_length,
                batch_size=config["batch_size"],
            )
            for query in data["queries"]
        }
        latency[variant] = {**_percentiles_ns(timings), "requests": len(timings), "concurrency": 1, "rerank_only": True}
    quality = {
        variant: _evaluate(
            data,
            ranking,
            sparse_for_answerability=sparse,
            dense_for_answerability=dense,
        )
        for variant, ranking in rankings.items()
    }
    baseline_quality = _evaluate(
        data, base, sparse_for_answerability=sparse, dense_for_answerability=dense
    )
    result = {
        "status": "provisional_seed50_candidate_screen_only",
        "device": torch.cuda.get_device_name(0),
        "baseline": baseline_quality,
        "quality": quality,
        "latency": latency,
        "promotion": {
            variant: _promotion_gate(baseline_quality, quality[variant], data["config"])
            for variant in rankings
        },
        "latency_scope": "rerank-only after precomputed E2; add measured E2 serving latency for end-to-end planning",
        "sota_claims_allowed": False,
    }
    top10 = [
        {
            "variant": variant,
            "query_id": query_id,
            "chunk_ids": [row["chunk_id"] for row in ranking[query_id][:10]],
        }
        for variant, ranking in sorted(rankings.items())
        for query_id in sorted(ranking)
    ]
    return _write_experiment(paths, "e3-v1", data, result, top10)
