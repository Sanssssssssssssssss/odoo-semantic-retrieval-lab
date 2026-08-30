from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ir_measures

from .baselines import _read_approved_trec_run
from .benchmark import load_jsonl
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json
from .paths import LabPaths


def _ndcg10(qrels: dict[str, int], ranking: list[dict[str, Any]]) -> float:
    measure = ir_measures.nDCG @ 10
    run = {row["chunk_id"]: row["score"] for row in ranking}
    return float(ir_measures.calc_aggregate([measure], {"q": qrels}, {"q": run})[measure])


def _auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = sum((left > right) + 0.5 * (left == right) for left in positives for right in negatives)
    true_positives = 0
    seen = 0
    ap = 0.0
    for score in sorted(set(float(value) for value in scores), reverse=True):
        tied = labels[scores == score]
        previous = true_positives
        true_positives += int(np.sum(tied))
        seen += len(tied)
        ap += (true_positives - previous) / positives.size * (true_positives / seen)
    return float(wins / (positives.size * negatives.size)), ap


def _balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    values = sorted(set(float(value) for value in scores))
    thresholds = [values[0] - 1.0] + [
        (left + right) / 2 for left, right in zip(values, values[1:])
    ] + [values[-1] + 1.0]
    best = None
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr = float(np.mean(predicted[labels == 1]))
        tnr = float(np.mean(~predicted[labels == 0]))
        candidate = ((tpr + tnr) / 2, float(np.mean(predicted)), -threshold)
        if best is None or candidate > best[0]:
            best = candidate, threshold
    return float(best[1])


def _group_folds(rows: list[dict[str, Any]], count: int = 5) -> list[list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_fact_group"]].append(row)
    folds = [[] for _ in range(count)]
    fold_counts = [[0, 0] for _ in range(count)]
    for group in sorted(grouped, key=lambda value: (-len(grouped[value]), value)):
        group_counts = [
            sum(row["answerable"] == label for row in grouped[group]) for label in (0, 1)
        ]
        dominant = max((0, 1), key=lambda label: (group_counts[label], label))
        fold = min(
            range(count),
            key=lambda index: (
                fold_counts[index][dominant],
                sum(fold_counts[index]),
                index,
            ),
        )
        folds[fold].append(group)
        fold_counts[fold][0] += group_counts[0]
        fold_counts[fold][1] += group_counts[1]
    return folds


def _answerability_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["answerable"] for row in rows], dtype=np.int8)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    auroc, auprc = _auc_ap(labels, scores)
    predictions: dict[str, bool] = {}
    fold_records = []
    for fold, groups in enumerate(_group_folds(rows)):
        test = np.asarray([row["source_fact_group"] in groups for row in rows])
        threshold = _balanced_threshold(labels[~test], scores[~test])
        for row, predicted in zip(rows, scores >= threshold):
            if row["source_fact_group"] in groups:
                predictions[row["query_id"]] = bool(predicted)
        fold_records.append(
            {
                "fold": fold,
                "test_source_fact_groups": sorted(groups),
                "threshold": threshold,
                "test_query_ids": sorted(row["query_id"] for row in rows if row["source_fact_group"] in groups),
            }
        )
    answered = [row for row in rows if predictions[row["query_id"]]]
    abstained = [row for row in rows if not predictions[row["query_id"]]]
    curve = []
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        selected = [row for row in rows if row["score"] >= threshold]
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(selected) / len(rows),
                "risk": sum(not row["answerable"] for row in selected) / len(selected),
            }
        )
    return {
        "auroc": auroc,
        "auprc_average_precision": auprc,
        "threshold_selection": "5-fold source_fact_group CV maximizing balanced accuracy; ties maximize coverage",
        "folds": fold_records,
        "coverage": len(answered) / len(rows),
        "selective_risk": sum(not row["answerable"] for row in answered) / len(answered)
        if answered
        else None,
        "abstention_precision": sum(not row["answerable"] for row in abstained) / len(abstained)
        if abstained
        else None,
        "risk_coverage_curve": curve,
    }


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _paired_diagnostic(
    baseline: dict[str, float],
    candidate: dict[str, float],
    groups: dict[str, str],
    rng: np.random.Generator,
    iterations: int = 10_000,
) -> dict[str, Any]:
    query_ids = sorted(baseline)
    deltas = {query_id: candidate[query_id] - baseline[query_id] for query_id in query_ids}
    clustered: dict[str, list[float]] = defaultdict(list)
    for query_id in query_ids:
        clustered[groups[query_id]].append(deltas[query_id])
    cluster_ids = sorted(clustered)
    bootstrap = np.empty(iterations)
    permutation = np.empty(iterations)
    for index in range(iterations):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        bootstrap[index] = np.mean([value for group in sampled for value in clustered[group]])
        signs = rng.choice((-1.0, 1.0), size=len(cluster_ids))
        permutation[index] = np.mean(
            [value * signs[group_index] for group_index, group in enumerate(cluster_ids) for value in clustered[group]]
        )
    values = np.asarray([deltas[query_id] for query_id in query_ids])
    mean = float(np.mean(values))
    return {
        "mean_delta_ndcg_at_10": mean,
        "cluster_bootstrap_95_ci": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
        "paired_cluster_permutation_p": float((np.sum(np.abs(permutation) >= abs(mean)) + 1) / (iterations + 1)),
        "standardized_effect_size": float(mean / np.std(values, ddof=1)) if np.std(values, ddof=1) else 0.0,
        "query_count": len(query_ids),
        "source_fact_group_count": len(cluster_ids),
    }


def run_diagnostics(paths: LabPaths | None = None, require_matrix_approval: bool = True) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    matrix_path = paths.root / "artifacts" / "matrices" / "seed50-provisional" / "manifest.json"
    if require_matrix_approval:
        approval = require_approval("p3b-matrix", paths)
        if approval["matrix_manifest_sha256"] != sha256_file(matrix_path):
            raise RuntimeError("Diagnostics input matrix differs from approved P3b matrix")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    benchmark_manifest_path = benchmark_root / "manifest.json"
    if sha256_file(benchmark_manifest_path) != matrix["benchmark_manifest_sha256"]:
        raise RuntimeError("Diagnostics benchmark manifest differs from the matrix binding")
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(benchmark_root / "queries.jsonl") != benchmark_manifest["output_hashes"]["queries.jsonl"]:
        raise RuntimeError("Diagnostics queries differ from the approved benchmark manifest")
    queries = load_jsonl(benchmark_root / "queries.jsonl")
    query_by_id = {row["id"]: row for row in queries}
    answerable = {row["id"] for row in queries if row["answerability"] == "answerable"}
    groups = {row["id"]: row["source_fact_group"] for row in queries}
    run_root = paths.root / "artifacts" / "runs" / "seed50-provisional"
    per_run = {}
    run_hashes = {}
    for entry in matrix["runs"]:
        ranking = _read_approved_trec_run(entry, run_root / entry["run_id"])
        qrels_path = benchmark_root / "derived" / entry["chunk_config_id"] / "qrels.seed.trec"
        if sha256_file(qrels_path) != benchmark_manifest["projection_reports"][entry["chunk_config_id"]]["trec_qrels_sha256"]:
            raise RuntimeError(f"Diagnostics qrels differ from the approved projection: {entry['chunk_config_id']}")
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for line in qrels_path.read_text(encoding="utf-8").splitlines():
            query_id, _, chunk_id, grade = line.split()
            qrels[query_id][chunk_id] = int(grade)
        scores = {
            query_id: _ndcg10(qrels[query_id], ranking[query_id]) for query_id in sorted(answerable)
        }
        answerability_rows = [
            {
                "query_id": query["id"],
                "answerable": int(query["answerability"] == "answerable"),
                "source_fact_group": query["source_fact_group"],
                "score": ranking[query["id"]][0]["score"],
            }
            for query in queries
        ]
        per_run[entry["run_id"]] = {
            "entry": entry,
            "per_query_ndcg_at_10": scores,
            "no_answer": _answerability_cv(answerability_rows),
        }
        run_hashes[entry["run_id"]] = sha256_file(run_root / entry["run_id"] / "run.trec")

    comparisons = []
    rng = np.random.default_rng(20260831)
    for config_id in sorted({entry["chunk_config_id"] for entry in matrix["runs"]}):
        entries = {entry["system"]: entry for entry in matrix["runs"] if entry["chunk_config_id"] == config_id}
        baseline = per_run[entries["E0-BM25"]["run_id"]]["per_query_ndcg_at_10"]
        for system in ("E1-dense-exact", "E2-hybrid-rrf", "E3-rerank"):
            candidate = per_run[entries[system]["run_id"]]["per_query_ndcg_at_10"]
            comparisons.append(
                {
                    "chunk_config_id": config_id,
                    "baseline": "E0-BM25",
                    "candidate": system,
                    **_paired_diagnostic(baseline, candidate, groups, rng),
                }
            )
    adjusted = _holm([row["paired_cluster_permutation_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value
    report = {
        "schema_version": 1,
        "status": "seed50_diagnostic_only_no_promotion_claim",
        "matrix_id": matrix["id"],
        "iterations": 10_000,
        "random_seed": 20260831,
        "comparisons": comparisons,
        "no_answer_by_run": {
            run_id: value["no_answer"] for run_id, value in sorted(per_run.items())
        },
        "sota_claims_allowed": False,
    }
    output = paths.root / "artifacts" / "diagnostics" / "seed50-provisional"
    write_json(output / "report.json", report, paths)
    manifest = {
        "id": stable_id(matrix["id"], canonical_json(run_hashes), sha256_file(output / "report.json"), length=40),
        "schema_version": 1,
        "status": report["status"],
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "queries_sha256": sha256_file(benchmark_root / "queries.jsonl"),
        "run_hashes": run_hashes,
        "implementation_sha256": sha256_file(paths.root / "src" / "osrlab" / "diagnostics.py"),
        "report_sha256": sha256_file(output / "report.json"),
        "sota_claims_allowed": False,
    }
    write_json(output / "manifest.json", manifest, paths)
    return manifest
