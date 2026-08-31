from __future__ import annotations

import json
import hashlib
import os
import platform
import site
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from packaging.utils import canonicalize_name
from transformers import AutoTokenizer

from .baselines import BGE_KEY, RERANKER_KEY, _unit_token_counts
from .benchmark import SNAPSHOT_ID, load_jsonl
from .chunking import _model_spec, _snapshot_dir, verify_model_snapshot
from .diagnostics import _answerability_cv
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json
from .paths import LabPaths
from .performance import _bytes, _launch_worker, _percentiles, _power_scheme
from .smoke import (
    _evidence_metrics,
    _read_qrels,
    _source_document_ndcg,
    assemble_evidence_cards,
)
from .verify import verify_source


CONCURRENCIES = (1, 4, 16, 64)
VARIANTS = (
    ("C2-structure-bounded", "E0-BM25", "cpu", "float32", CONCURRENCIES),
    ("C2-structure-bounded", "E1-dense-exact", "cpu", "float32", CONCURRENCIES),
    ("C2-structure-bounded", "E2-hybrid-rrf", "cpu", "float32", CONCURRENCIES),
    ("C2-structure-bounded", "E1-dense-exact", "cuda", "float16", CONCURRENCIES),
    ("C2-structure-bounded", "E2-hybrid-rrf", "cuda", "float16", CONCURRENCIES),
    ("C3-structure-merged", "E2-hybrid-rrf", "cpu", "float32", CONCURRENCIES),
    ("C3-structure-merged", "E2-hybrid-rrf", "cuda", "float16", CONCURRENCIES),
    ("C2-structure-bounded", "E3-rerank", "cuda", "float16", (1,)),
    ("C3-structure-merged", "E3-rerank", "cuda", "float16", (1,)),
)


def _variant_id(chunk: str, system: str, device: str, dtype: str) -> str:
    return f"{chunk}__{system}__{device}-{dtype}"


def _implementation_hashes(paths: LabPaths) -> dict[str, str]:
    names = (
        "src/osrlab/p5.py",
        "src/osrlab/perf_worker.py",
        "src/osrlab/performance.py",
        "src/osrlab/baselines.py",
        "src/osrlab/chunking.py",
        "src/osrlab/smoke.py",
        "src/osrlab/diagnostics.py",
        "src/osrlab/jsonio.py",
        "src/osrlab/paths.py",
        "src/osrlab/gates.py",
        "lab.ps1",
        "configs/gpu-requirements.lock",
        "uv.lock",
    )
    return {name: sha256_file(paths.root / name) for name in names}


def _input_hashes(paths: LabPaths, p4_root: Path) -> dict[str, str]:
    benchmark = paths.root / "benchmarks" / "seed50" / "provisional"
    matrix = paths.root / "artifacts" / "matrices" / "seed50-provisional"
    files = {
        "p4/manifest.json": p4_root / "manifest.json",
        "p4/summary.json": p4_root / "summary.json",
        "p4/report.md": p4_root / "report.md",
        "matrix/manifest.json": matrix / "manifest.json",
        "matrix/metrics_summary.json": matrix / "metrics_summary.json",
        "benchmark/queries.jsonl": benchmark / "queries.jsonl",
        "benchmark/judgments.jsonl": benchmark / "judgments.jsonl",
        "benchmark/nuggets.jsonl": benchmark / "nuggets.jsonl",
        "pool/hard_negatives.jsonl": paths.root
        / "benchmarks"
        / "seed50"
        / "pooling"
        / "provisional"
        / "adjudicated"
        / "hard_negatives.jsonl",
        "configs/models.json": paths.root / "configs" / "models.json",
        "source_snapshot.json": paths.snapshot_manifest,
        "diagnostics.py": paths.root / "src" / "osrlab" / "diagnostics.py",
    }
    for chunk in ("C2-structure-bounded", "C3-structure-merged"):
        files[f"qrels/{chunk}.trec"] = benchmark / "derived" / chunk / "qrels.seed.trec"
        files[f"chunks/{chunk}/manifest.json"] = (
            paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / chunk / "manifest.json"
        )
        for system in ("E0-BM25", "E1-dense-exact"):
            files[f"indexes/{chunk}/{system}/manifest.json"] = (
                paths.root / "indexes" / SNAPSHOT_ID / chunk / system / "manifest.json"
            )
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"P5 direct input is absent: {missing}")
    return {name: sha256_file(path) for name, path in sorted(files.items())}


def _verify_gpu_environment(paths: LabPaths) -> dict[str, Any]:
    lock_path = paths.root / "configs" / "gpu-requirements.lock"
    expected = {
        canonicalize_name(name): version
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for name, version in (line.split("==", 1),)
    }
    installed = {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in distributions(path=site.getsitepackages())
        if distribution.metadata.get("Name")
    }
    mismatch = {
        name: {"expected": version, "observed": installed.get(name)}
        for name, version in expected.items()
        if installed.get(name) != version
    }
    extras = sorted(set(installed) - set(expected) - {"odoo-semantic-retrieval-lab"})
    if mismatch or extras:
        raise RuntimeError(f"GPU environment differs from lock: mismatch={mismatch}, extras={extras}")
    if torch.__version__ != expected["torch"] or torch.version.cuda != "12.6":
        raise RuntimeError("Pinned CUDA torch runtime mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic CuBLAS workspace contract is absent")
    return {
        "lock_sha256": sha256_file(lock_path),
        "package_count": len(expected),
        "packages": {name: installed[name] for name in sorted(expected)},
        "editable_project": installed.get("odoo-semantic-retrieval-lab"),
    }


def _git_state(paths: LabPaths) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=paths.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.replace("\r\n", "\n")
    return {
        "head": head,
        "dirty": bool(status),
        "status_lines": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _nvidia_driver() -> str:
    import pynvml

    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetDriverVersion()
    return driver.decode("ascii") if isinstance(driver, bytes) else str(driver)


def _worker_args(
    chunk: str,
    device: str,
    dtype: str,
    concurrency: int,
    minimum_requests: int,
    minimum_seconds: int,
) -> tuple[str, ...]:
    return (
        "--chunk-config",
        chunk,
        "--device",
        device,
        "--dtype",
        dtype,
        "--concurrency",
        str(concurrency),
        "--minimum-requests",
        str(minimum_requests),
        "--minimum-seconds",
        str(minimum_seconds),
    )


def _matches(
    receipt: dict[str, Any],
    run_id: str,
    chunk: str,
    system: str,
    mode: str,
    device: str,
    dtype: str,
    concurrency: int,
    minimum_requests: int,
    minimum_seconds: int,
) -> bool:
    identity = all(
        (
            receipt.get("run_id") == run_id,
            receipt.get("chunk_config") == chunk,
            receipt.get("system") == system,
            receipt.get("mode") == mode,
            receipt.get("device") == device,
            receipt.get("dtype") == dtype,
            receipt.get("concurrency") == concurrency,
            bool(receipt.get("requests")),
            "external_peak_process_tree_rss_bytes" in receipt,
        )
    )
    if not identity:
        return False
    if mode == "cold":
        return len(receipt["requests"]) == 1 and "runtime_initialization_duration_ns" in receipt
    return all(
        (
            len(receipt["requests"]) >= minimum_requests,
            receipt.get("minimum_requests") == minimum_requests,
            receipt.get("minimum_seconds") == minimum_seconds,
            receipt.get("measured_loop_duration_ns", 0) >= minimum_seconds * 1_000_000_000,
            len({row["query_id"] for row in receipt["requests"]}) == 50,
        )
    )


def _run_or_load(
    paths: LabPaths,
    output: Path,
    run_id: str,
    chunk: str,
    system: str,
    mode: str,
    device: str,
    dtype: str,
    concurrency: int,
    minimum_requests: int,
    minimum_seconds: int,
) -> dict[str, Any]:
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if _matches(
            cached,
            run_id,
            chunk,
            system,
            mode,
            device,
            dtype,
            concurrency,
            minimum_requests,
            minimum_seconds,
        ):
            return cached
    return _launch_worker(
        paths,
        system,
        mode,
        output,
        run_id,
        worker_args=_worker_args(
            chunk, device, dtype, concurrency, minimum_requests, minimum_seconds
        ),
    )


def _quality(paths: LabPaths, chunk_config: str, receipt: dict[str, Any]) -> dict[str, Any]:
    queries = load_jsonl(paths.root / "benchmarks" / "seed50" / "provisional" / "queries.jsonl")
    query_by_id = {query["id"]: query for query in queries}
    qrels = _read_qrels(
        paths.root
        / "benchmarks"
        / "seed50"
        / "provisional"
        / "derived"
        / chunk_config
        / "qrels.seed.trec"
    )
    rankings: dict[str, tuple[str, ...]] = {}
    scores: dict[str, float] = {}
    observed: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for request in receipt["requests"]:
        ranking = tuple(request["top10_chunk_ids"])
        observed[request["query_id"]].add(ranking)
        rankings.setdefault(request["query_id"], ranking)
        scores.setdefault(request["query_id"], float(request["top_score"]))
    if set(rankings) != set(query_by_id):
        raise RuntimeError("P5 warm run did not cover every Seed50 query")
    if any(len(values) != 1 for values in observed.values()):
        raise RuntimeError("P5 top-10 rankings changed between repeated requests")

    answerable = [query for query in queries if query["answerability"] == "answerable"]
    recall = {}
    for depth in (1, 3, 5, 10):
        per_query = []
        for query in answerable:
            relevant = {chunk_id for chunk_id, grade in qrels[query["id"]].items() if grade >= 2}
            per_query.append(len(relevant & set(rankings[query["id"]][:depth])) / len(relevant))
        recall[f"recall_at_{depth}"] = float(np.mean(per_query))

    chunks = load_jsonl(
        paths.root
        / "corpus"
        / "derived"
        / SNAPSHOT_ID
        / "chunks"
        / chunk_config
        / "chunks.jsonl"
    )
    chunks_by_span: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for fragment in chunk["span_fragments"]:
            for span_id in fragment["source_span_ids"]:
                chunks_by_span[span_id].add(chunk["id"])
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = {unit["id"]: unit for unit in load_jsonl(evidence_root / "evidence_units.jsonl")}
    spans = load_jsonl(evidence_root / "source_spans.jsonl")
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    judgments = load_jsonl(benchmark_root / "judgments.jsonl")
    nuggets = load_jsonl(benchmark_root / "nuggets.jsonl")
    bge_path = _snapshot_dir(paths, _model_spec(paths, BGE_KEY))
    tokenizer = AutoTokenizer.from_pretrained(bge_path, local_files_only=True, use_fast=True)
    ranked_rows = {
        query_id: [
            {
                "query_id": query_id,
                "rank": rank,
                "chunk_id": chunk_id,
                "score": float(11 - rank),
            }
            for rank, chunk_id in enumerate(ranking, 1)
        ]
        for query_id, ranking in rankings.items()
    }
    cards_by_query = {}
    card_diagnostics = {}
    for query in answerable:
        cards, diagnostics = assemble_evidence_cards(
            query["id"], ranked_rows[query["id"]], chunk_by_id, units, tokenizer=tokenizer
        )
        cards_by_query[query["id"]] = cards
        card_diagnostics[query["id"]] = diagnostics
    answerable_ids = {query["id"] for query in answerable}
    cross_chunker = _evidence_metrics(
        answerable_ids,
        cards_by_query,
        card_diagnostics,
        judgments,
        nuggets,
        _unit_token_counts(paths),
    )
    cross_chunker["source_document_ndcg_at_10"] = _source_document_ndcg(
        answerable_ids,
        ranked_rows,
        chunk_by_id,
        judgments,
        nuggets,
        {span["id"]: span["source_document_id"] for span in spans},
    )
    negatives = load_jsonl(
        paths.root
        / "benchmarks"
        / "seed50"
        / "pooling"
        / "provisional"
        / "adjudicated"
        / "hard_negatives.jsonl"
    )
    hard_negative = {"count": len(negatives), "status": "provisional"}
    for depth in (1, 3, 5, 10):
        hits = [
            bool(chunks_by_span[item["source_span_id"]] & set(rankings[item["query_id"]][:depth]))
            for item in negatives
        ]
        hard_negative[f"hit_rate_at_{depth}"] = sum(hits) / len(hits)
    by_provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in negatives:
        by_provenance[item["provenance"]].append(item)
    hard_negative["hit_rate_at_10_by_provenance"] = {
        provenance: sum(
            bool(chunks_by_span[item["source_span_id"]] & set(rankings[item["query_id"]][:10]))
            for item in items
        )
        / len(items)
        for provenance, items in sorted(by_provenance.items())
    }
    no_answer = _answerability_cv(
        [
            {
                "query_id": query["id"],
                "source_fact_group": query["source_fact_group"],
                "answerable": int(query["answerability"] == "answerable"),
                "score": scores[query["id"]],
            }
            for query in queries
        ]
    )
    return {
        "status": "provisional_agent_only",
        "answerable_queries": len(answerable),
        "no_answer_queries": len(queries) - len(answerable),
        "binary_relevance_threshold": 2,
        "fixed_chunk_metrics_not_cross_chunker_comparable": recall,
        "cross_chunker_2048_token_metrics": cross_chunker,
        "no_answer": no_answer,
        "hard_negatives": hard_negative,
        "top10_repeat_deterministic": True,
    }


def _summarize_warm(receipt: dict[str, Any]) -> dict[str, Any]:
    requests = receipt["requests"]
    return {
        "requests": len(requests),
        "measured_seconds": receipt["measured_loop_duration_ns"] / 1_000_000_000,
        "client_latency": _percentiles([row["total_duration_ns"] for row in requests]),
        "service_latency": _percentiles([row["service_duration_ns"] for row in requests]),
        "retrieval_latency": _percentiles([row["retrieval_duration_ns"] for row in requests]),
        "queue_latency": _percentiles([row["queue_duration_ns"] for row in requests]),
        "qps": len(requests) / (receipt["measured_loop_duration_ns"] / 1_000_000_000),
        "runtime_resident_rss_delta_after_load_bytes": max(
            0,
            receipt["resource_samples"][1]["rss_bytes"]
            - receipt["resource_samples"][0]["rss_bytes"],
        ),
        "memory_breakdown": receipt["runtime_memory_breakdown"],
        "peak_process_tree_rss_bytes": receipt["external_peak_process_tree_rss_bytes"],
        "peak_torch_allocated_vram_bytes": receipt["peak_torch_allocated_vram_bytes"],
        "peak_torch_reserved_vram_bytes": receipt["peak_torch_reserved_vram_bytes"],
        "peak_nvml_used_bytes_system_global": receipt["external_peak_nvml_used_bytes"],
    }


def _summarize_cold(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "processes": len(receipts),
        "load": _percentiles([row["runtime_initialization_duration_ns"] for row in receipts]),
        "first_query_end_to_end": _percentiles(
            [row["requests"][0]["total_duration_ns"] for row in receipts]
        ),
        "first_query_retrieval": _percentiles(
            [row["requests"][0]["retrieval_duration_ns"] for row in receipts]
        ),
    }


def _run_p5_impl(
    paths: LabPaths | None = None,
    *,
    minimum_seconds: int = 60,
    minimum_requests: int = 1024,
    cold_processes: int = 5,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    verify_source(paths)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    p4_receipt_path = paths.root / "reviews" / "p4" / "approval.json"
    p4_hint = json.loads(p4_receipt_path.read_text(encoding="utf-8"))
    p4_root = paths.root / "artifacts" / "performance" / "p4" / p4_hint["run_id"]
    p4_approval = require_approval(
        "p4",
        paths,
        {
            "manifest_sha256": p4_root / "manifest.json",
            "summary_sha256": p4_root / "summary.json",
            "report_sha256": p4_root / "report.md",
        },
    )
    matrix_path = paths.root / "artifacts" / "matrices" / "seed50-provisional" / "manifest.json"
    matrix_approval = require_approval(
        "p3b-matrix", paths, {"matrix_manifest_sha256": matrix_path}
    )
    model_receipts = {
        BGE_KEY: verify_model_snapshot(paths, BGE_KEY),
        RERANKER_KEY: verify_model_snapshot(paths, RERANKER_KEY),
    }
    gpu_environment = _verify_gpu_environment(paths)
    if not torch.cuda.is_available():
        raise RuntimeError("P5 requires the repository .venv-gpu CUDA environment")
    if minimum_seconds < 1 or minimum_requests < 50 or cold_processes < 1:
        raise ValueError("P5 requires seconds>=1, requests>=50, and cold_processes>=1")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix["id"] != matrix_approval["matrix_id"]:
        raise RuntimeError("P5 matrix identity differs from approved P3b matrix")
    specification = {
        "variants": [
            {
                "chunk_config": chunk,
                "system": system,
                "device": device,
                "dtype": dtype,
                "concurrencies": list(concurrencies),
            }
            for chunk, system, device, dtype, concurrencies in VARIANTS
        ],
        "minimum_seconds": minimum_seconds,
        "minimum_requests": minimum_requests,
        "cold_processes": cold_processes,
        "warmup_requests": 20,
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "candidate_quality_status": "provisional_agent_only",
        "latency_scopes": ["retrieval", "service_end_to_end", "client_with_queue"],
    }
    implementation_hashes = _implementation_hashes(paths)
    input_hashes = _input_hashes(paths, p4_root)
    git_state = _git_state(paths)
    power_scheme_before = _power_scheme()
    run_id = stable_id(
        matrix["id"],
        canonical_json(specification),
        canonical_json(implementation_hashes),
        canonical_json(input_hashes),
        gpu_environment["lock_sha256"],
        length=40,
    )
    root = paths.require_write_path(paths.root / "artifacts" / "performance" / "p5" / run_id)
    workers = paths.require_write_path(root / "workers")
    workers.mkdir(parents=True, exist_ok=True)
    progress = {
        "run_id": run_id,
        "status": "running",
        "completed_workers": 0,
        "total_workers": sum(cold_processes + len(item[4]) for item in VARIANTS),
        "current": None,
    }
    write_json(root / "progress.json", progress, paths)
    write_json(
        paths.root / "artifacts" / "performance" / "p5" / "active.json",
        {"run_id": run_id, "root": str(root), "status": "running"},
        paths,
    )
    summary: dict[str, Any] = {}
    for chunk, system, device, dtype, concurrencies in VARIANTS:
        variant = _variant_id(chunk, system, device, dtype)
        progress["current"] = variant
        cold = []
        for index in range(1, cold_processes + 1):
            output = workers / f"cold__{variant}__{index}.json"
            cold.append(
                _run_or_load(
                    paths,
                    output,
                    run_id,
                    chunk,
                    system,
                    "cold",
                    device,
                    dtype,
                    1,
                    minimum_requests,
                    minimum_seconds,
                )
            )
            progress["completed_workers"] += 1
            write_json(root / "progress.json", progress, paths)
        warm = {}
        for concurrency in concurrencies:
            output = workers / f"warm__{variant}__c{concurrency}.json"
            receipt = _run_or_load(
                paths,
                output,
                run_id,
                chunk,
                system,
                "warm",
                device,
                dtype,
                concurrency,
                minimum_requests,
                minimum_seconds,
            )
            warm[str(concurrency)] = {
                **_summarize_warm(receipt),
                "quality": _quality(paths, chunk, receipt),
            }
            progress["completed_workers"] += 1
            write_json(root / "progress.json", progress, paths)
        summary[variant] = {
            "chunk_config": chunk,
            "system": system,
            "device": device,
            "dtype": dtype,
            "index_bytes": _bytes(paths.root / "indexes" / SNAPSHOT_ID / chunk),
            "cold": _summarize_cold(cold),
            "warm": warm,
        }

    control = summary[_variant_id("C2-structure-bounded", "E2-hybrid-rrf", "cpu", "float32")][
        "warm"
    ]["1"]
    optimized = summary[_variant_id("C2-structure-bounded", "E2-hybrid-rrf", "cuda", "float16")][
        "warm"
    ]["1"]
    control_recall = control["quality"]["fixed_chunk_metrics_not_cross_chunker_comparable"][
        "recall_at_5"
    ]
    optimized_recall = optimized["quality"]["fixed_chunk_metrics_not_cross_chunker_comparable"][
        "recall_at_5"
    ]
    recall_drop = control_recall - optimized_recall
    promotion = {
        "target": "C2 E2 CUDA-fp16 unique-query concurrency=1 retrieval p50 <10 ms",
        "quality_gate": "Recall@5 absolute drop <=0.005 against P5 CPU-fp32 control",
        "retrieval_p50_ms": optimized["retrieval_latency"]["p50_ms"],
        "recall_at_5": optimized_recall,
        "control_recall_at_5": control_recall,
        "recall_at_5_absolute_drop": recall_drop,
        "passes_latency": optimized["retrieval_latency"]["p50_ms"] < 10,
        "passes_quality": recall_drop <= 0.005,
    }
    promotion["passes"] = promotion["passes_latency"] and promotion["passes_quality"]
    write_json(root / "summary.json", summary, paths)
    write_json(root / "promotion.json", promotion, paths)
    report = [
        "# P5 retrieval serving and CUDA rerank experiment",
        "",
        f"Run `{run_id}` on `{platform.platform()}` with torch `{torch.__version__}` / CUDA `{torch.version.cuda}`.",
        "Seed50 quality remains provisional; no SOTA or production claim is permitted.",
        "",
        "Chunk-qrels Recall is a fixed-chunker diagnostic only and must not be compared between C2 and C3.",
    ]
    for chunk in ("C2-structure-bounded", "C3-structure-merged"):
        report.extend(
            [
                "",
                f"## {chunk} fixed-chunker serving metrics",
                "",
                "| Variant | C | retrieval p50 | p95 | p99 | client p50 | QPS | Recall@1/3/5/10 | index RSS MiB | runtime RSS MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, row in summary.items():
            if row["chunk_config"] != chunk:
                continue
            for concurrency, warm in row["warm"].items():
                fixed = warm["quality"]["fixed_chunk_metrics_not_cross_chunker_comparable"]
                report.append(
                    f"| {variant} | {concurrency} | {warm['retrieval_latency']['p50_ms']:.3f} | "
                    f"{warm['retrieval_latency']['p95_ms']:.3f} | {warm['retrieval_latency']['p99_ms']:.3f} | "
                    f"{warm['client_latency']['p50_ms']:.3f} | {warm['qps']:.2f} | "
                    f"{fixed['recall_at_1']:.4f}/{fixed['recall_at_3']:.4f}/{fixed['recall_at_5']:.4f}/{fixed['recall_at_10']:.4f} | "
                    f"{warm['memory_breakdown']['index_resident_rss_delta_bytes'] / 1048576:.1f} | "
                    f"{warm['peak_process_tree_rss_bytes'] / 1048576:.1f} |"
                )
    report.extend(
        [
            "",
            "## Cold-start metrics",
            "",
            "| Variant | Processes | load p50/p95/p99 ms | first retrieval p50/p95/p99 ms | first end-to-end p50/p95/p99 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant, row in summary.items():
        cold = row["cold"]
        report.append(
            f"| {variant} | {cold['processes']} | "
            f"{cold['load']['p50_ms']:.3f}/{cold['load']['p95_ms']:.3f}/{cold['load']['p99_ms']:.3f} | "
            f"{cold['first_query_retrieval']['p50_ms']:.3f}/{cold['first_query_retrieval']['p95_ms']:.3f}/{cold['first_query_retrieval']['p99_ms']:.3f} | "
            f"{cold['first_query_end_to_end']['p50_ms']:.3f}/{cold['first_query_end_to_end']['p95_ms']:.3f}/{cold['first_query_end_to_end']['p99_ms']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Cross-chunker fixed-budget evidence metrics",
            "",
            "| Variant | Evidence recall@2048 | Required completeness@2048 | Irrelevant ratio@2048 | Duplicate evidence@2048 | Source-doc nDCG@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, row in summary.items():
        if row["system"] not in ("E2-hybrid-rrf", "E3-rerank"):
            continue
        cross = row["warm"]["1"]["quality"]["cross_chunker_2048_token_metrics"]
        report.append(
            f"| {variant} | {cross['evidence_recall_at_2048_tokens']:.6f} | "
            f"{cross['required_nugget_completeness_at_2048_tokens']:.6f} | "
            f"{cross['irrelevant_token_ratio_at_2048_tokens']:.6f} | "
            f"{cross['duplicate_evidence_rate_at_2048_tokens']:.6f} | "
            f"{cross['source_document_ndcg_at_10']:.6f} |"
        )
    report.extend(
        [
            "",
            "## Provisional no-answer and hard-negative diagnostics",
            "",
            "| Variant | AUROC | AUPRC | Abstention precision | Coverage | Selective risk | Hard-negative hit@1/3/5/10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, row in summary.items():
        quality = row["warm"]["1"]["quality"]
        no_answer = quality["no_answer"]
        hard = quality["hard_negatives"]
        report.append(
            f"| {variant} | {no_answer['auroc']:.4f} | {no_answer['auprc_average_precision']:.4f} | "
            f"{no_answer['abstention_precision']:.4f} | {no_answer['coverage']:.4f} | "
            f"{no_answer['selective_risk']:.4f} | {hard['hit_rate_at_1']:.4f}/"
            f"{hard['hit_rate_at_3']:.4f}/{hard['hit_rate_at_5']:.4f}/{hard['hit_rate_at_10']:.4f} |"
        )
    report.extend(
        [
            "",
            "Retrieval latency ends before EvidenceCard assembly. Service latency includes EvidenceCard assembly; client latency also includes executor queueing.",
            f"Promotion gate: `{promotion['passes']}`; retrieval p50 `{promotion['retrieval_p50_ms']:.3f}` ms; "
            f"Recall@5 drop `{promotion['recall_at_5_absolute_drop']:.6f}`.",
        ]
    )
    report_path = root / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    power_scheme_after = _power_scheme()
    if power_scheme_before != power_scheme_after:
        raise RuntimeError("Windows power scheme changed during P5")
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "status": "complete",
        "input_matrix_id": matrix["id"],
        "p4_input_run_id": p4_approval["run_id"],
        "specification": specification,
        "implementation_hashes": implementation_hashes,
        "input_hashes": input_hashes,
        "model_receipts": model_receipts,
        "git_state": git_state,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "execution_provider": "pytorch_cuda",
            "nvidia_driver": _nvidia_driver(),
            "power_scheme_before": power_scheme_before,
            "power_scheme_after": power_scheme_after,
            "power_scheme_modified": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "gpu_dependency_environment": gpu_environment,
        },
        "promotion": promotion,
        "output_hashes": {
            "summary.json": sha256_file(root / "summary.json"),
            "promotion.json": sha256_file(root / "promotion.json"),
            "report.md": sha256_file(report_path),
            **{
                f"workers/{path.name}": sha256_file(path)
                for path in sorted(workers.glob("*.json"))
            },
        },
        "sota_claims_allowed": False,
    }
    write_json(root / "manifest.json", manifest, paths)
    progress.update({"status": "complete", "current": None, "manifest": str(root / "manifest.json")})
    write_json(root / "progress.json", progress, paths)
    write_json(
        paths.root / "artifacts" / "performance" / "p5" / "active.json",
        {"run_id": run_id, "root": str(root), "status": "complete"},
        paths,
    )
    return manifest


def run_p5(
    paths: LabPaths | None = None,
    *,
    minimum_seconds: int = 60,
    minimum_requests: int = 1024,
    cold_processes: int = 5,
) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    try:
        return _run_p5_impl(
            paths,
            minimum_seconds=minimum_seconds,
            minimum_requests=minimum_requests,
            cold_processes=cold_processes,
        )
    except BaseException as error:
        active_path = paths.root / "artifacts" / "performance" / "p5" / "active.json"
        if active_path.is_file():
            active = json.loads(active_path.read_text(encoding="utf-8"))
            if active.get("status") == "running":
                root = paths.require_write_path(Path(active["root"]))
                progress_path = root / "progress.json"
                progress = (
                    json.loads(progress_path.read_text(encoding="utf-8"))
                    if progress_path.is_file()
                    else {"run_id": active["run_id"]}
                )
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                progress.update(
                    {
                        "status": status,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                write_json(progress_path, progress, paths)
                write_json(active_path, {**active, "status": status}, paths)
        raise
