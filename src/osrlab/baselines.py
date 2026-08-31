from __future__ import annotations

import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import torch
import torch.nn.functional as torch_functional
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .benchmark import SNAPSHOT_ID, load_jsonl
from .chunking import _model_spec, _snapshot_dir, verify_model_snapshot
from .contract import validate_record
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .smoke import (
    EXPECTED_SEED50_MANIFEST_SHA256,
    _answerability_diagnostics,
    _evidence_metrics,
    _lexical_chunk_text,
    _rank,
    _read_qrels,
    _source_document_ndcg,
    _verify_manifest_outputs,
    assemble_evidence_cards,
    evaluate_ranking,
)
from .verify import verify_source


CHUNK_CONFIGS = ("C0-fixed", "C1-section-native", "C2-structure-bounded", "C3-structure-merged")
BGE_KEY = "bge_small_en_v1_5"
RERANKER_KEY = "ms_marco_minilm_l6_v2"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _system_configs(paths: LabPaths) -> dict[str, dict[str, Any]]:
    models = json.loads((paths.root / "configs" / "models.json").read_text(encoding="utf-8"))
    return {
        "E0-BM25": {
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
        },
        "E1-dense-exact": {
            "system": "E1-dense-exact",
            "implementation": "transformers-cls-pooling",
            "implementation_version": version("transformers"),
            "model": models[BGE_KEY]["repo_id"],
            "model_revision": models[BGE_KEY]["revision"],
            "query_instruction": QUERY_INSTRUCTION,
            "normalization": "L2",
            "dtype": "float32",
            "search": "numpy-exact-cosine",
            "batch_size": 64,
            "max_length": 512,
            "top_k": 100,
            "tie_break": "chunk_id_ascending",
        },
        "E2-hybrid-rrf": {
            "system": "E2-hybrid-rrf",
            "inputs": ["E0-BM25:top100", "E1-dense-exact:top100"],
            "rrf_k": 60,
            "top_k": 100,
            "tie_break": "chunk_id_ascending",
        },
        "E3-rerank": {
            "system": "E3-rerank",
            "implementation": "transformers-sequence-classification",
            "implementation_version": version("transformers"),
            "model": models[RERANKER_KEY]["repo_id"],
            "model_revision": models[RERANKER_KEY]["revision"],
            "candidate_pool": "E2-hybrid-rrf:top50",
            "logical_chunk_score": "max_subwindow",
            "max_length": 512,
            "batch_size": 32,
            "tie_break": ["E2-rank", "chunk_id"],
        },
    }


def _unit_token_counts(paths: LabPaths) -> dict[str, int]:
    source = (
        paths.root
        / "corpus"
        / "derived"
        / SNAPSHOT_ID
        / "chunks"
        / "C1-section-native"
        / "chunks.jsonl"
    )
    counts: dict[str, int] = {}
    for chunk in load_jsonl(source):
        for fragment in chunk["span_fragments"]:
            unit_id = fragment["evidence_unit_id"]
            counts[unit_id] = max(counts.get(unit_id, 0), fragment["unit_token_end"])
    return counts


def _window_texts(
    chunks: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    unit_token_counts: dict[str, int],
    tokenizer: Any,
) -> tuple[list[str], list[str], np.ndarray]:
    lexical: list[str] = []
    dense: list[str] = []
    chunk_indices: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        windows = chunk["scoring_windows"]
        if len(windows) == 1 and windows[0] == {"token_start": 0, "token_end": chunk["token_count"]}:
            lexical.append(_lexical_chunk_text(chunk, units, unit_token_counts))
            dense.append(chunk["title"] + "\n" + chunk["text"])
            chunk_indices.append(chunk_index)
            continue
        offsets = tokenizer(
            chunk["text"], add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )["offset_mapping"]
        for window in windows:
            start, end = window["token_start"], window["token_end"]
            text = chunk["text"][offsets[start][0] : offsets[end - 1][1]]
            heading = " > ".join(chunk["heading_path"])
            lexical.append(heading + "\n" + text)
            dense.append(chunk["title"] + "\n" + text)
            chunk_indices.append(chunk_index)
    return lexical, dense, np.asarray(chunk_indices, dtype=np.int32)


def _encode_bge(model: Any, tokenizer: Any, texts: list[str], batch_size: int = 64) -> np.ndarray:
    device = next(model.parameters()).device
    output: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            embeddings = model(**batch).last_hidden_state[:, 0]
            embeddings = torch_functional.normalize(embeddings, p=2, dim=1)
        output.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(output, axis=0)


def _aggregate_window_scores(
    window_scores: np.ndarray, window_chunk_indices: np.ndarray, chunk_count: int
) -> np.ndarray:
    chunk_scores = np.full(chunk_count, -np.inf, dtype=np.float32)
    np.maximum.at(chunk_scores, window_chunk_indices, window_scores.astype(np.float32, copy=False))
    return chunk_scores


def _rank_all(
    score_matrix: np.ndarray,
    window_chunk_indices: np.ndarray,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    chunk_ids = [chunk["id"] for chunk in chunks]
    rankings: dict[str, list[dict[str, Any]]] = {}
    for query_index, query in enumerate(queries):
        scores = _aggregate_window_scores(score_matrix[query_index], window_chunk_indices, len(chunks))
        indices = _rank(scores, chunk_ids, top_k)
        rankings[query["id"]] = [
            {
                "query_id": query["id"],
                "rank": rank,
                "chunk_id": chunk_ids[index],
                "score": float(scores[index]),
            }
            for rank, index in enumerate(indices, 1)
        ]
    return rankings


def _hybrid(
    sparse: dict[str, list[dict[str, Any]]], dense: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for query_id in sorted(sparse):
        scores: dict[str, float] = defaultdict(float)
        for ranking in (sparse[query_id], dense[query_id]):
            for result in ranking[:100]:
                scores[result["chunk_id"]] += 1 / (60 + result["rank"])
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:100]
        output[query_id] = [
            {"query_id": query_id, "rank": rank, "chunk_id": chunk_id, "score": scores[chunk_id]}
            for rank, chunk_id in enumerate(ordered, 1)
        ]
    return output


def _rerank(
    model: Any,
    tokenizer: Any,
    queries: list[dict[str, Any]],
    hybrid: dict[str, list[dict[str, Any]]],
    chunks: list[dict[str, Any]],
    dense_windows: list[str],
    window_chunk_indices: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    device = next(model.parameters()).device
    chunk_index_by_id = {chunk["id"]: index for index, chunk in enumerate(chunks)}
    windows_by_chunk: dict[int, list[int]] = defaultdict(list)
    for window_index, chunk_index in enumerate(window_chunk_indices.tolist()):
        windows_by_chunk[chunk_index].append(window_index)
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[str, str]] = []
    e2_rank: dict[tuple[str, str], int] = {}
    for query in queries:
        query_id = query["id"]
        for result in hybrid[query_id][:50]:
            chunk_id = result["chunk_id"]
            e2_rank[(query_id, chunk_id)] = result["rank"]
            for window_index in windows_by_chunk[chunk_index_by_id[chunk_id]]:
                pairs.append((query["text"], dense_windows[window_index]))
                owners.append((query_id, chunk_id))
    logits: list[np.ndarray] = []
    for start in range(0, len(pairs), 32):
        batch_pairs = pairs[start : start + 32]
        batch = tokenizer(
            [query for query, _ in batch_pairs],
            [passage for _, passage in batch_pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits.append(model(**batch).logits.reshape(-1).cpu().numpy().astype(np.float32, copy=False))
    window_scores = np.concatenate(logits)
    scores: dict[tuple[str, str], float] = {}
    for owner, score in zip(owners, window_scores.tolist()):
        scores[owner] = max(scores.get(owner, -float("inf")), score)
    output: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        query_id = query["id"]
        candidates = [result["chunk_id"] for result in hybrid[query_id][:50]]
        ordered = sorted(
            candidates,
            key=lambda chunk_id: (-scores[(query_id, chunk_id)], e2_rank[(query_id, chunk_id)], chunk_id),
        )
        output[query_id] = [
            {
                "query_id": query_id,
                "rank": rank,
                "chunk_id": chunk_id,
                "score": scores[(query_id, chunk_id)],
            }
            for rank, chunk_id in enumerate(ordered, 1)
        ]
    return output


def _save_indexes(
    paths: LabPaths,
    config_id: str,
    chunk_manifest: dict[str, Any],
    retriever: bm25s.BM25,
    window_chunk_indices: np.ndarray,
    embeddings: np.ndarray,
    system_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = paths.require_write_path(paths.root / "indexes" / SNAPSHOT_ID / config_id)
    bm25_root = paths.require_write_path(root / "E0-BM25")
    dense_root = paths.require_write_path(root / "E1-dense-exact")
    bm25_root.mkdir(parents=True, exist_ok=True)
    dense_root.mkdir(parents=True, exist_ok=True)
    retriever.save(bm25_root)
    np.save(bm25_root / "window_chunk_indices.npy", window_chunk_indices, allow_pickle=False)
    np.save(dense_root / "window_chunk_indices.npy", window_chunk_indices, allow_pickle=False)
    np.save(dense_root / "embeddings.float32.npy", embeddings, allow_pickle=False)
    manifests: dict[str, Any] = {}
    for system_id, directory in (("E0-BM25", bm25_root), ("E1-dense-exact", dense_root)):
        files = {
            str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "id": stable_id(config_id, system_id, canonical_json(files)),
            "schema_version": 1,
            "system": system_id,
            "source_snapshot_id": SNAPSHOT_ID,
            "chunk_config_hash": chunk_manifest["chunk_config_hash"],
            "config_hash": stable_id(canonical_json(system_configs[system_id]), length=64),
            "files": files,
        }
        write_json(directory / "manifest.json", manifest, paths)
        manifests[system_id] = manifest
    return manifests


def _write_run(
    paths: LabPaths,
    config_id: str,
    system_id: str,
    system_config: dict[str, Any],
    rankings: dict[str, list[dict[str, Any]]],
    chunks: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    tokenizer: Any,
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    judgments: list[dict[str, Any]],
    nuggets: list[dict[str, Any]],
    span_to_document: dict[str, str],
    unit_token_counts: dict[str, int],
    chunk_manifest: dict[str, Any],
    benchmark_manifest_path: Path,
) -> dict[str, Any]:
    answerable_ids = {query["id"] for query in queries if query["answerability"] == "answerable"}
    no_answer_ids = {query["id"] for query in queries if query["answerability"] == "no_answer"}
    if set(qrels) & no_answer_ids:
        raise RuntimeError("No-answer query leaked into ordinary ranking qrels")
    run_for_metrics = {
        query_id: {result["chunk_id"]: result["score"] for result in rankings[query_id]}
        for query_id in sorted(answerable_ids)
    }
    fixed = evaluate_ranking(
        {query_id: qrels[query_id] for query_id in sorted(answerable_ids)}, run_for_metrics
    )
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    cards_by_query: dict[str, list[dict[str, Any]]] = {}
    card_diagnostics: dict[str, dict[str, float | int]] = {}
    for query in queries:
        cards, diagnostics = assemble_evidence_cards(
            query["id"], rankings[query["id"]], chunk_by_id, units, tokenizer=tokenizer
        )
        cards_by_query[query["id"]] = cards
        card_diagnostics[query["id"]] = diagnostics
    fixed["source_document_ndcg_at_10"] = _source_document_ndcg(
        answerable_ids, rankings, chunk_by_id, judgments, nuggets, span_to_document
    )
    fixed.update(
        _evidence_metrics(
            answerable_ids, cards_by_query, card_diagnostics, judgments, nuggets, unit_token_counts
        )
    )
    answerability = _answerability_diagnostics(
        [
            (query["id"], int(query["answerability"] == "answerable"), rankings[query["id"]][0]["score"])
            for query in queries
        ]
    )
    metrics = {
        "status": "provisional_unpooled",
        "ranking_scope": {
            "answerable_only": True,
            "query_count": len(answerable_ids),
            "excluded_no_answer_query_ids": sorted(no_answer_ids),
            "binary_relevance_threshold": 2,
        },
        "fixed_chunk_metrics": fixed,
        "no_answer_diagnostics": answerability,
        "sota_claims_allowed": False,
    }
    corpus_manifest_path = (
        paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / config_id / "manifest.json"
    )
    config_hash = stable_id(canonical_json(system_config), length=64)
    run_id = stable_id(
        SNAPSHOT_ID,
        config_id,
        system_id,
        sha256_file(corpus_manifest_path),
        sha256_file(benchmark_manifest_path),
        config_hash,
        length=40,
    )
    output = paths.require_write_path(paths.root / "artifacts" / "runs" / "seed50-provisional" / run_id)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", system_config, paths)
    run_tag = f"{system_id}-{config_id}".replace("-", "_")
    trec = "".join(
        f"{query_id} Q0 {result['chunk_id']} {result['rank']} {result['score']:.9g} {run_tag}\n"
        for query_id in sorted(rankings)
        for result in rankings[query_id]
    )
    paths.require_write_path(output / "run.trec").write_text(trec, encoding="utf-8", newline="\n")
    query_by_id = {query["id"]: query for query in queries}
    top10 = [
        {
            **result,
            "query_text": query_by_id[query_id]["text"],
            "answerability": query_by_id[query_id]["answerability"],
            "title": chunk_by_id[result["chunk_id"]]["title"],
            "source_uri": chunk_by_id[result["chunk_id"]]["source_uri"],
            "qrel_grade": qrels.get(query_id, {}).get(result["chunk_id"]),
        }
        for query_id in sorted(rankings)
        for result in rankings[query_id][:10]
    ]
    cards = [card for query_id in sorted(cards_by_query) for card in cards_by_query[query_id]]
    write_jsonl(output / "top10.jsonl", top10, sort_key="query_id", paths=paths)
    write_jsonl(output / "evidence_cards.jsonl", cards, sort_key="query_id", paths=paths)
    write_json(output / "metrics.json", metrics, paths)
    output_hashes = {
        relative: sha256_file(output / relative)
        for relative in ("config.json", "run.trec", "top10.jsonl", "evidence_cards.jsonl", "metrics.json")
    }
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "system": system_id,
        "source_snapshot_id": SNAPSHOT_ID,
        "chunk_config_id": config_id,
        "chunk_config_hash": chunk_manifest["chunk_config_hash"],
        "corpus_hash": sha256_file(corpus_manifest_path),
        "benchmark_hash": sha256_file(benchmark_manifest_path),
        "config_hash": config_hash,
        "query_set": "Seed50",
        "status": "provisional_unpooled",
        "run_tag": run_tag,
        "acquisition_included": False,
        "output_hashes": output_hashes,
    }
    validate_record("RunManifest", manifest)
    write_json(output / "manifest.json", manifest, paths)
    return {"manifest": manifest, "metrics": metrics, "output": str(output)}


def _read_trec_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        query_id, q0, chunk_id, rank, score, _ = line.split()
        if q0 != "Q0":
            raise RuntimeError(f"Invalid TREC run row: {line}")
        rankings[query_id].append(
            {
                "query_id": query_id,
                "rank": int(rank),
                "chunk_id": chunk_id,
                "score": float(score),
            }
        )
    for query_id, ranking in rankings.items():
        if [item["rank"] for item in ranking] != list(range(1, len(ranking) + 1)):
            raise RuntimeError(f"Non-consecutive TREC ranks for {query_id}: {path}")
    return dict(rankings)


def _read_approved_trec_run(
    entry: dict[str, Any], run_root: Path
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = run_root / "manifest.json"
    if sha256_file(manifest_path) != entry["manifest_sha256"]:
        raise RuntimeError(f"Persisted run manifest hash mismatch: {entry['run_id']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = (manifest["id"], manifest["chunk_config_id"], manifest["system"])
    expected = (entry["run_id"], entry["chunk_config_id"], entry["system"])
    if identity != expected:
        raise RuntimeError(f"Persisted run identity mismatch: {entry['run_id']}")
    run_path = run_root / "run.trec"
    if sha256_file(run_path) != manifest["output_hashes"].get("run.trec"):
        raise RuntimeError(f"Persisted TREC run hash mismatch: {entry['run_id']}")
    return _read_trec_run(run_path)


def _preflight_persisted_rankings(
    entries: list[dict[str, Any]], runs_root: Path, expected_query_ids: set[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    validated: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in entries:
        rankings = _read_approved_trec_run(entry, runs_root / entry["run_id"])
        if set(rankings) != expected_query_ids:
            raise RuntimeError(f"Persisted run query set mismatch: {entry['run_id']}")
        validated[entry["run_id"]] = rankings
    return validated


def refresh_provisional_outputs(paths: LabPaths | None = None) -> dict[str, Any]:
    """Rebuild cards/metrics/manifests from the already persisted, approved rankings."""
    paths = paths or LabPaths.discover()
    require_approval("p3a", paths)
    verify_source(paths)
    model_receipts = {
        BGE_KEY: verify_model_snapshot(paths, BGE_KEY),
        RERANKER_KEY: verify_model_snapshot(paths, RERANKER_KEY),
    }
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    benchmark_manifest_path = benchmark_root / "manifest.json"
    if sha256_file(benchmark_manifest_path) != EXPECTED_SEED50_MANIFEST_SHA256:
        raise RuntimeError("Approved provisional Seed50 manifest hash mismatch")
    matrix_root = paths.root / "artifacts" / "matrices" / "seed50-provisional"
    previous = json.loads((matrix_root / "manifest.json").read_text(encoding="utf-8"))
    system_configs = _system_configs(paths)
    queries = load_jsonl(benchmark_root / "queries.jsonl")
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = {unit["id"]: unit for unit in load_jsonl(evidence_root / "evidence_units.jsonl")}
    span_to_document = {
        span["id"]: span["source_document_id"]
        for span in load_jsonl(evidence_root / "source_spans.jsonl")
    }
    judgments = load_jsonl(benchmark_root / "judgments.jsonl")
    nuggets = load_jsonl(benchmark_root / "nuggets.jsonl")
    unit_token_counts = _unit_token_counts(paths)
    persisted_rankings = _preflight_persisted_rankings(
        previous["runs"],
        paths.root / "artifacts" / "runs" / "seed50-provisional",
        {query["id"] for query in queries},
    )
    bge_path = _snapshot_dir(paths, _model_spec(paths, BGE_KEY))
    tokenizer = AutoTokenizer.from_pretrained(bge_path, local_files_only=True, use_fast=True)
    runs: list[dict[str, Any]] = []
    for entry in previous["runs"]:
        config_id, system_id, run_id = (
            entry["chunk_config_id"],
            entry["system"],
            entry["run_id"],
        )
        chunk_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / config_id
        chunk_manifest = _verify_manifest_outputs(chunk_root)
        rankings = persisted_rankings[run_id]
        runs.append(
            _write_run(
                paths,
                config_id,
                system_id,
                system_configs[system_id],
                rankings,
                load_jsonl(chunk_root / "chunks.jsonl"),
                units,
                tokenizer,
                queries,
                _read_qrels(benchmark_root / "derived" / config_id / "qrels.seed.trec"),
                judgments,
                nuggets,
                span_to_document,
                unit_token_counts,
                chunk_manifest,
                benchmark_manifest_path,
            )
        )
    matrix_binding = [
        {
            "run_id": run["manifest"]["id"],
            "system": run["manifest"]["system"],
            "chunk_config_id": run["manifest"]["chunk_config_id"],
            "manifest_sha256": sha256_file(Path(run["output"]) / "manifest.json"),
        }
        for run in runs
    ]
    write_json(
        matrix_root / "metrics_summary.json",
        {
            f"{run['manifest']['chunk_config_id']}/{run['manifest']['system']}": run["metrics"]
            for run in runs
        },
        paths,
    )
    matrix = {
        "id": stable_id(canonical_json({"runs": matrix_binding}), length=40),
        "schema_version": 1,
        "status": "provisional_unpooled",
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "models": model_receipts,
        "runs": matrix_binding,
        "indexes": previous["indexes"],
        "metrics_summary_sha256": sha256_file(matrix_root / "metrics_summary.json"),
        "sota_claims_allowed": False,
    }
    write_json(matrix_root / "manifest.json", matrix, paths)
    return matrix


def run_provisional_matrix(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    require_approval("p3a", paths)
    verify_source(paths)
    bge_receipt = verify_model_snapshot(paths, BGE_KEY)
    reranker_receipt = verify_model_snapshot(paths, RERANKER_KEY)
    benchmark_root = paths.root / "benchmarks" / "seed50" / "provisional"
    benchmark_manifest_path = benchmark_root / "manifest.json"
    if sha256_file(benchmark_manifest_path) != EXPECTED_SEED50_MANIFEST_SHA256:
        raise RuntimeError("Approved provisional Seed50 manifest hash mismatch")
    system_configs = _system_configs(paths)
    queries = load_jsonl(benchmark_root / "queries.jsonl")
    evidence_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence"
    units = {unit["id"]: unit for unit in load_jsonl(evidence_root / "evidence_units.jsonl")}
    spans = load_jsonl(evidence_root / "source_spans.jsonl")
    span_to_document = {span["id"]: span["source_document_id"] for span in spans}
    judgments = load_jsonl(benchmark_root / "judgments.jsonl")
    nuggets = load_jsonl(benchmark_root / "nuggets.jsonl")
    unit_token_counts = _unit_token_counts(paths)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    bge_spec = _model_spec(paths, BGE_KEY)
    bge_path = _snapshot_dir(paths, bge_spec)
    bge_tokenizer = AutoTokenizer.from_pretrained(bge_path, local_files_only=True, use_fast=True)
    bge_model = AutoModel.from_pretrained(bge_path, local_files_only=True).eval()
    query_embeddings = _encode_bge(
        bge_model,
        bge_tokenizer,
        [QUERY_INSTRUCTION + query["text"] for query in queries],
    )
    reranker_spec = _model_spec(paths, RERANKER_KEY)
    reranker_path = _snapshot_dir(paths, reranker_spec)
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path, local_files_only=True, use_fast=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        reranker_path, local_files_only=True
    ).eval()
    runs: list[dict[str, Any]] = []
    indexes: dict[str, Any] = {}
    for config_id in CHUNK_CONFIGS:
        print(f"[{config_id}] load and window", flush=True)
        chunk_root = paths.root / "corpus" / "derived" / SNAPSHOT_ID / "chunks" / config_id
        chunk_manifest = _verify_manifest_outputs(chunk_root)
        chunks = load_jsonl(chunk_root / "chunks.jsonl")
        lexical_windows, dense_windows, window_chunk_indices = _window_texts(
            chunks, units, unit_token_counts, bge_tokenizer
        )
        print(f"[{config_id}] E0 BM25", flush=True)
        sparse_tokens = bm25s.tokenize(
            lexical_windows, lower=True, stopwords="english", stemmer=None, show_progress=False
        )
        sparse_model = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
        sparse_model.index(sparse_tokens, show_progress=False)
        sparse_scores = np.stack(
            [
                sparse_model.get_scores(
                    bm25s.tokenize(
                        query["text"],
                        lower=True,
                        stopwords="english",
                        stemmer=None,
                        return_ids=False,
                        show_progress=False,
                    )[0]
                )
                for query in queries
            ]
        ).astype(np.float32, copy=False)
        sparse = _rank_all(sparse_scores, window_chunk_indices, chunks, queries)
        print(f"[{config_id}] E1 dense encode {len(dense_windows)} windows", flush=True)
        embeddings = _encode_bge(bge_model, bge_tokenizer, dense_windows)
        dense_scores = query_embeddings @ embeddings.T
        dense = _rank_all(dense_scores, window_chunk_indices, chunks, queries)
        print(f"[{config_id}] E2 hybrid and E3 rerank", flush=True)
        hybrid = _hybrid(sparse, dense)
        reranked = _rerank(
            reranker_model,
            reranker_tokenizer,
            queries,
            hybrid,
            chunks,
            dense_windows,
            window_chunk_indices,
        )
        indexes[config_id] = _save_indexes(
            paths,
            config_id,
            chunk_manifest,
            sparse_model,
            window_chunk_indices,
            embeddings,
            system_configs,
        )
        qrels = _read_qrels(benchmark_root / "derived" / config_id / "qrels.seed.trec")
        for system_id, ranking in (
            ("E0-BM25", sparse),
            ("E1-dense-exact", dense),
            ("E2-hybrid-rrf", hybrid),
            ("E3-rerank", reranked),
        ):
            print(f"[{config_id}] write {system_id}", flush=True)
            runs.append(
                _write_run(
                    paths,
                    config_id,
                    system_id,
                    system_configs[system_id],
                    ranking,
                    chunks,
                    units,
                    bge_tokenizer,
                    queries,
                    qrels,
                    judgments,
                    nuggets,
                    span_to_document,
                    unit_token_counts,
                    chunk_manifest,
                    benchmark_manifest_path,
                )
            )
    matrix_binding = [
        {
            "run_id": run["manifest"]["id"],
            "system": run["manifest"]["system"],
            "chunk_config_id": run["manifest"]["chunk_config_id"],
            "manifest_sha256": sha256_file(Path(run["output"]) / "manifest.json"),
        }
        for run in runs
    ]
    metrics_summary = {
        f"{run['manifest']['chunk_config_id']}/{run['manifest']['system']}": run["metrics"]
        for run in runs
    }
    matrix_root = paths.root / "artifacts" / "matrices" / "seed50-provisional"
    write_json(matrix_root / "metrics_summary.json", metrics_summary, paths)
    matrix = {
        "id": stable_id(canonical_json({"runs": matrix_binding}), length=40),
        "schema_version": 1,
        "status": "provisional_unpooled",
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "models": {BGE_KEY: bge_receipt, RERANKER_KEY: reranker_receipt},
        "runs": matrix_binding,
        "indexes": indexes,
        "metrics_summary_sha256": sha256_file(matrix_root / "metrics_summary.json"),
        "sota_claims_allowed": False,
    }
    write_json(matrix_root / "manifest.json", matrix, paths)
    return matrix
