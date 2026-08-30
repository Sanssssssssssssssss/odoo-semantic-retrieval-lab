from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import bm25s
import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .baselines import (
    BGE_KEY,
    QUERY_INSTRUCTION,
    RERANKER_KEY,
    _aggregate_window_scores,
    _encode_bge,
    _hybrid,
    _rerank,
    _unit_token_counts,
    _window_texts,
)
from .benchmark import SNAPSHOT_ID, load_jsonl
from .chunking import _c2, _chunk_configs, _model_spec, _snapshot_dir, _tokenizer
from .extraction import CANONICAL_FILES as EXTRACTION_FILES
from .extraction import run_worker, validate_extraction_output
from .jsonio import sha256_file, write_json
from .paths import LabPaths
from .smoke import _lexical_chunk_text, _rank, assemble_evidence_cards


T = TypeVar("T")
SYSTEMS = ("E0-BM25", "E1-dense-exact", "E2-hybrid-rrf", "E3-rerank")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(timings: dict[str, int], stage: str, function: Callable[[], T]) -> T:
    _sync()
    started = time.perf_counter_ns()
    value = function()
    _sync()
    timings[stage] = timings.get(stage, 0) + time.perf_counter_ns() - started
    return value


def _resource() -> dict[str, int]:
    process = psutil.Process()
    children = process.children(recursive=True)
    own = process.memory_info().rss
    return {
        "monotonic_ns": time.perf_counter_ns(),
        "rss_bytes": own,
        "process_tree_rss_bytes": own
        + sum(child.memory_info().rss for child in children if child.is_running()),
        "child_processes": len(children),
        "torch_allocated_vram_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "torch_reserved_vram_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
    }


class Runtime:
    def __init__(self, system: str, timings: dict[str, int]):
        self.system = system
        self.paths = LabPaths.discover()
        self.chunks = _timed(
            timings,
            "load_chunks",
            lambda: load_jsonl(
                self.paths.root
                / "corpus"
                / "derived"
                / SNAPSHOT_ID
                / "chunks"
                / "C2-structure-bounded"
                / "chunks.jsonl"
            ),
        )
        evidence = self.paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
        self.units = _timed(
            timings,
            "load_evidence_units",
            lambda: {unit["id"]: unit for unit in load_jsonl(evidence / "evidence_units.jsonl")},
        )
        bge_path = _snapshot_dir(self.paths, _model_spec(self.paths, BGE_KEY))
        self.tokenizer = _timed(
            timings,
            "load_bge_tokenizer",
            lambda: AutoTokenizer.from_pretrained(bge_path, local_files_only=True, use_fast=True),
        )
        self.unit_token_counts = _unit_token_counts(self.paths)
        configured_index = os.environ.get("OSRLAB_PERF_INDEX_ROOT")
        index = (
            Path(configured_index)
            if configured_index
            else self.paths.root / "indexes" / SNAPSHOT_ID / "C2-structure-bounded"
        )
        self.window_chunk_indices = _timed(
            timings,
            "load_window_mapping",
            lambda: np.load(
                index / "E1-dense-exact" / "window_chunk_indices.npy",
                allow_pickle=False,
            ),
        )
        self.chunk_ids = [chunk["id"] for chunk in self.chunks]
        self.sparse = None
        self.embeddings = None
        self.bge = None
        self.reranker = None
        self.dense_windows: list[str] | None = None
        if system in ("E0-BM25", "E2-hybrid-rrf", "E3-rerank"):
            self.sparse = _timed(
                timings,
                "load_bm25_index",
                lambda: bm25s.BM25.load(index / "E0-BM25", load_corpus=False),
            )
        if system in ("E1-dense-exact", "E2-hybrid-rrf", "E3-rerank"):
            self.embeddings = _timed(
                timings,
                "load_dense_index",
                lambda: np.load(
                    index / "E1-dense-exact" / "embeddings.float32.npy",
                    allow_pickle=False,
                ),
            )
            self.bge = _timed(
                timings,
                "load_bge_model",
                lambda: AutoModel.from_pretrained(bge_path, local_files_only=True).eval(),
            )
        if system == "E3-rerank":
            reranker_path = _snapshot_dir(self.paths, _model_spec(self.paths, RERANKER_KEY))
            self.reranker_tokenizer = _timed(
                timings,
                "load_reranker_tokenizer",
                lambda: AutoTokenizer.from_pretrained(
                    reranker_path, local_files_only=True, use_fast=True
                ),
            )
            self.reranker = _timed(
                timings,
                "load_reranker_model",
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    reranker_path, local_files_only=True
                ).eval(),
            )
            self.dense_windows = _timed(
                timings,
                "materialize_rerank_windows",
                lambda: _window_texts(
                    self.chunks,
                    self.units,
                    self.unit_token_counts,
                    self.tokenizer,
                )[1],
            )

    def _sparse_rank(self, query: dict[str, Any], timings: dict[str, int]) -> list[dict[str, Any]]:
        tokens = _timed(
            timings,
            "query_sparse_tokenize",
            lambda: bm25s.tokenize(
                query["text"],
                lower=True,
                stopwords="english",
                stemmer=None,
                return_ids=False,
                show_progress=False,
            )[0],
        )
        window_scores = _timed(timings, "sparse_search", lambda: self.sparse.get_scores(tokens))
        scores = _aggregate_window_scores(
            window_scores, self.window_chunk_indices, len(self.chunks)
        )
        indices = _rank(scores, self.chunk_ids, 100)
        return [
            {"query_id": query["id"], "rank": rank, "chunk_id": self.chunk_ids[index], "score": float(scores[index])}
            for rank, index in enumerate(indices, 1)
        ]

    def _dense_rank(self, query: dict[str, Any], timings: dict[str, int]) -> list[dict[str, Any]]:
        query_embedding = _timed(
            timings,
            "query_embedding",
            lambda: _encode_bge(
                self.bge, self.tokenizer, [QUERY_INSTRUCTION + query["text"]], batch_size=1
            )[0],
        )
        window_scores = _timed(timings, "dense_exact_search", lambda: self.embeddings @ query_embedding)
        scores = _aggregate_window_scores(
            window_scores, self.window_chunk_indices, len(self.chunks)
        )
        indices = _rank(scores, self.chunk_ids, 100)
        return [
            {"query_id": query["id"], "rank": rank, "chunk_id": self.chunk_ids[index], "score": float(scores[index])}
            for rank, index in enumerate(indices, 1)
        ]

    def query(self, query: dict[str, Any]) -> tuple[dict[str, int], list[str], int]:
        timings: dict[str, int] = {}
        sparse = self._sparse_rank(query, timings) if self.sparse is not None else None
        dense = self._dense_rank(query, timings) if self.embeddings is not None else None
        if self.system == "E0-BM25":
            ranking = sparse
        elif self.system == "E1-dense-exact":
            ranking = dense
        else:
            ranking = _timed(
                timings,
                "fusion_rrf",
                lambda: _hybrid({query["id"]: sparse}, {query["id"]: dense})[query["id"]],
            )
        if self.system == "E3-rerank":
            ranking = _timed(
                timings,
                "rerank_top50",
                lambda: _rerank(
                    self.reranker,
                    self.reranker_tokenizer,
                    [query],
                    {query["id"]: ranking},
                    self.chunks,
                    self.dense_windows,
                    self.window_chunk_indices,
                )[query["id"]],
            )
        cards = _timed(
            timings,
            "evidence_card_assembly",
            lambda: assemble_evidence_cards(
                query["id"],
                ranking,
                {chunk["id"]: chunk for chunk in self.chunks},
                self.units,
                tokenizer=self.tokenizer,
            )[0],
        )
        return timings, [item["chunk_id"] for item in ranking[:10]], sum(card["token_count"] for card in cards)


def run_queries(system: str, mode: str, output: Path, run_id: str) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    load_timings: dict[str, int] = {}
    resources = [_resource()]
    runtime_started = time.perf_counter_ns()
    runtime = Runtime(system, load_timings)
    runtime_initialization_duration_ns = time.perf_counter_ns() - runtime_started
    resources.append(_resource())
    query_path = runtime.paths.root / "benchmarks" / "seed50" / "provisional" / "queries.jsonl"
    queries = load_jsonl(query_path)
    requests: list[dict[str, Any]] = []
    if mode == "cold":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        query = queries[0]
        started = time.perf_counter_ns()
        timings, top10, card_tokens = runtime.query(query)
        requests.append(
            {
                "sequence": 1,
                "query_id": query["id"],
                "total_duration_ns": time.perf_counter_ns() - started,
                "stage_durations_ns": timings,
                "top10_chunk_ids": top10,
                "evidence_card_tokens": card_tokens,
            }
        )
        resources.append(_resource())
    else:
        for index in range(20):
            runtime.query(queries[index % len(queries)])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started_loop = time.perf_counter_ns()
        sequence = 0
        while sequence < 1024 or time.perf_counter_ns() - started_loop < 60_000_000_000:
            query = queries[sequence % len(queries)]
            started = time.perf_counter_ns()
            timings, top10, card_tokens = runtime.query(query)
            sequence += 1
            requests.append(
                {
                    "sequence": sequence,
                    "query_id": query["id"],
                    "total_duration_ns": time.perf_counter_ns() - started,
                    "stage_durations_ns": timings,
                    "top10_chunk_ids": top10,
                    "evidence_card_tokens": card_tokens,
                }
            )
        loop_duration_ns = time.perf_counter_ns() - started_loop
        resources.append(_resource())
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "system": system,
        "mode": mode,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "load_stage_durations_ns": load_timings,
        "runtime_initialization_duration_ns": runtime_initialization_duration_ns,
        "warmup_requests": 0 if mode == "cold" else 20,
        "measured_loop_duration_ns": None if mode == "cold" else loop_duration_ns,
        "peak_torch_allocated_vram_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "peak_torch_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())
        if torch.cuda.is_available()
        else 0,
        "requests": requests,
        "resource_samples": resources,
    }
    write_json(output, receipt, runtime.paths)
    return receipt


def run_build(output: Path, run_id: str) -> dict[str, Any]:
    paths = LabPaths.discover()
    timings: dict[str, int] = {}
    resources = [_resource()]
    extraction_output = output.parent / "fresh-extraction"
    if extraction_output.exists():
        shutil.rmtree(extraction_output)
    extraction = _timed(timings, "fresh_sphinx_extraction", lambda: run_worker(extraction_output))
    extraction_validation = _timed(
        timings, "extraction_validation", lambda: validate_extraction_output(extraction_output, paths)
    )
    resources.append(_resource())
    bge_spec = _model_spec(paths, BGE_KEY)
    tokenizer = _timed(timings, "load_tokenizer", lambda: _tokenizer(paths, bge_spec))
    configs = _chunk_configs(bge_spec)
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = sorted(
        load_jsonl(extraction_output / "evidence_units.jsonl"),
        key=lambda item: (item["source_document_id"], item["ordinal"], item["id"]),
    )
    chunks = _timed(
        timings,
        "chunk_c2_generation",
        lambda: _c2(configs["C2-structure-bounded"], tokenizer, units),
    )
    chunks.sort(key=lambda row: row["id"])
    canonical_chunks = load_jsonl(
        paths.root
        / "corpus"
        / "derived"
        / SNAPSHOT_ID
        / "chunks"
        / "C2-structure-bounded"
        / "chunks.jsonl"
    )
    chunk_match = [(row["id"], row["text_sha256"]) for row in chunks] == [
        (row["id"], row["text_sha256"]) for row in canonical_chunks
    ]
    unit_map = {unit["id"]: unit for unit in units}
    lexical, dense, mapping = _timed(
        timings,
        "window_materialization",
        lambda: _window_texts(chunks, unit_map, _unit_token_counts(paths), tokenizer),
    )
    sparse_tokens = bm25s.tokenize(
        lexical, lower=True, stopwords="english", stemmer=None, show_progress=False
    )
    sparse_model = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    _timed(timings, "bm25_index_build", lambda: sparse_model.index(sparse_tokens, show_progress=False))
    bge_path = _snapshot_dir(paths, bge_spec)
    bge = _timed(
        timings,
        "load_bge_model",
        lambda: AutoModel.from_pretrained(bge_path, local_files_only=True).eval(),
    )
    embeddings = _timed(timings, "dense_corpus_embedding", lambda: _encode_bge(bge, tokenizer, dense))
    staging = output.parent / "fresh-index"
    if staging.exists():
        shutil.rmtree(staging)

    def persist() -> None:
        sparse_model.save(staging / "E0-BM25")
        dense_root = staging / "E1-dense-exact"
        dense_root.mkdir(parents=True, exist_ok=True)
        np.save(dense_root / "embeddings.float32.npy", embeddings, allow_pickle=False)
        np.save(dense_root / "window_chunk_indices.npy", mapping, allow_pickle=False)

    _timed(timings, "index_persist", persist)
    resources.append(_resource())
    canonical_dense = (
        paths.root
        / "indexes"
        / SNAPSHOT_ID
        / "C2-structure-bounded"
        / "E1-dense-exact"
        / "embeddings.float32.npy"
    )
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "system": "C2-index-build",
        "acquisition_included": False,
        "stage_durations_ns": timings,
        "resource_samples": resources,
        "extraction_counts": extraction["counts"],
        "extraction_validation": extraction_validation,
        "extraction_canonical_hashes_match": all(
            sha256_file(extraction_output / name) == sha256_file(evidence_root / name)
            for name in EXTRACTION_FILES
        ),
        "c2_chunks_match_canonical": chunk_match,
        "dense_index_sha256": sha256_file(staging / "E1-dense-exact" / "embeddings.float32.npy"),
        "canonical_dense_index_sha256": sha256_file(canonical_dense),
        "dense_index_matches_canonical": sha256_file(
            staging / "E1-dense-exact" / "embeddings.float32.npy"
        )
        == sha256_file(canonical_dense),
    }
    write_json(output, receipt, paths)
    return receipt


def main() -> int:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=(*SYSTEMS, "BUILD"), required=True)
    parser.add_argument("--mode", choices=("cold", "warm", "build"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = (
        run_build(args.output, args.run_id)
        if args.mode == "build"
        else run_queries(args.system, args.mode, args.output, args.run_id)
    )
    print(json.dumps({"system": args.system, "mode": args.mode, "requests": len(result.get("requests", []))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
