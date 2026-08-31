from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
from docutils import nodes

import osrlab.cli as cli
import osrlab.p5 as p5

from osrlab.contract import validate_record
from osrlab.baselines import (
    _aggregate_window_scores,
    _hybrid,
    _preflight_persisted_rankings,
    _read_approved_trec_run,
)
from osrlab.benchmark import validate_topic_briefs
from osrlab.chunking import _c1_scoring_windows, _c3, _chunk_configs, _make_chunk, _model_spec, _tokenizer, _windows, verify_evidence_snapshot
from osrlab.extraction import EvidenceCollector, _normalize_rendered
from osrlab.gates import require_approval
from osrlab.jsonio import sha256_file, write_json, write_jsonl
from osrlab.paths import LabPaths, PathBoundaryError
from osrlab.pooling import (
    _agent_pool_is_stable,
    _annotation_label,
    _application_family,
    _is_wrong_version_candidate,
    _negative_classes,
    _order_tau,
    _reported_annotation_sha,
    _run_provenance,
    _system_order,
    _true_sibling_headings,
)
from osrlab.performance import _framework_overhead, _percentiles, _runtime_initialization_ns
from osrlab.p5 import _matches
from osrlab.diagnostics import _answerability_cv, _auc_ap, _holm, _ndcg10
from osrlab.smoke import _lexical_chunk_text, _rank, assemble_evidence_cards, evaluate_ranking
from osrlab.tuning import BM25F, _rrf, _tmm_convex


def test_path_allowlist_rejects_source_and_external_paths(tmp_path: Path) -> None:
    root = tmp_path / "lab"
    paths = LabPaths(root)
    assert paths.require_write_path("artifacts/run.json") == (root / "artifacts/run.json").resolve()
    assert paths.require_write_path(".venv-gpu/receipt.json") == (root / ".venv-gpu/receipt.json").resolve()
    assert paths.require_write_path(".private/v0/hidden.jsonl") == (root / ".private/v0/hidden.jsonl").resolve()
    with pytest.raises(PathBoundaryError):
        paths.require_write_path(root.parent / "erp-openai" / "result.json")
    with pytest.raises(PathBoundaryError):
        paths.require_write_path(paths.docs / "changed.rst")
    with pytest.raises(PathBoundaryError):
        paths.require_write_path("artifacts/../benchmarks/run.json")


def test_canonical_jsonl_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "b", "value": 2}, {"value": 1, "id": "a"}])
    assert path.read_bytes() == b'{"id":"a","value":1}\n{"id":"b","value":2}\n'


def test_approval_rejects_stale_hash_binding(tmp_path: Path) -> None:
    paths = LabPaths(tmp_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("first", encoding="utf-8")
    write_json(
        tmp_path / "reviews" / "phase" / "approval.json",
        {
            "decision": "APPROVE",
            "phase": "phase",
            "artifact_sha256": sha256_file(artifact),
        },
        paths,
    )
    require_approval("phase", paths, {"artifact_sha256": artifact})
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash binding is stale"):
        require_approval("phase", paths, {"artifact_sha256": artifact})


def test_public_writers_enforce_allowlist() -> None:
    paths = LabPaths.discover()
    with pytest.raises(PathBoundaryError):
        write_json(paths.docs / "forbidden.json", {"value": 1})


def test_source_span_schema_accepts_multi_origin() -> None:
    digest = "a" * 64
    validate_record(
        "SourceSpan",
        {
            "id": "span-1",
            "source_document_id": "doc-1",
            "span_refs": [
                {
                    "source_path": "applications/example.rst",
                    "source_sha256": digest,
                    "start_line": 10,
                    "start_column": 1,
                    "end_line": 11,
                    "end_column": 20,
                    "anchor": "example",
                    "quote_sha256": digest,
                    "origin_kind": "direct",
                },
                {
                    "source_path": "include/shared.rst",
                    "source_sha256": digest,
                    "start_line": 1,
                    "start_column": 1,
                    "end_line": 1,
                    "end_column": 2,
                    "anchor": None,
                    "quote_sha256": digest,
                    "origin_kind": "include",
                },
            ],
        },
    )


def test_rendered_text_normalization_is_stable() -> None:
    assert _normalize_rendered("\r\nAlpha  \r\n\r\n\r\nBeta\r\n") == "Alpha\n\nBeta"


def test_tab_label_is_preserved_as_structure_context() -> None:
    tabs = nodes.container(classes=["sphinx-tabs"])
    tab = nodes.container()
    label = nodes.container()
    label += nodes.paragraph(text="Structure")
    content = nodes.paragraph(text="Insert a table")
    tab += label
    tab += content
    tabs += tab
    assert EvidenceCollector._structure_context(content) == [{"kind": "tab", "label": "Structure"}]


def test_rawsource_location_handles_directive_indentation(tmp_path: Path) -> None:
    paths = LabPaths(tmp_path)
    source = paths.docs / "content" / "sample.rst"
    source.parent.mkdir(parents=True)
    source.write_text(".. code-block:: xml\n\n   <field a=\"1\"/>\n   <field b=\"2\"/>\n", encoding="utf-8")
    collector = EvidenceCollector(paths)
    assert collector._locate_rawsource(source, '<field a="1"/>\n<field b="2"/>', hint_line=1) == 3


def test_discrete_source_ranges_are_not_bridged() -> None:
    assert EvidenceCollector._merge_ranges([(447, 447), (448, 449), (1536, 1536)]) == [
        (447, 449),
        (1536, 1536),
    ]


def test_token_windows_respect_size_and_overlap() -> None:
    assert _windows(1_000, 512, 100) == [(0, 512), (412, 924), (824, 1_000)]
    assert _windows(845, 512, 100) == [(0, 512), (412, 845)]
    assert _windows(832, 512, 100) == [(0, 512), (412, 832)]
    assert _windows(833, 512, 100) == [(0, 512), (412, 833)]
    assert _windows(128, 512, 100) == [(0, 128)]
    assert _c1_scoring_windows(512) == [(0, 512)]
    assert _c1_scoring_windows(513) == [(0, 480), (416, 513)]


def test_chunk_retains_partial_evidence_fragment_mapping() -> None:
    paths = LabPaths.discover()
    tokenizer = _tokenizer(paths, _model_spec(paths))
    config = _chunk_configs(_model_spec(paths))["C0-fixed"]
    digest = "a" * 32
    unit = {
        "id": digest,
        "source_document_id": "b" * 32,
        "source_span_ids": ["c" * 32],
        "section_id": "d" * 32,
        "parent_section_id": None,
        "heading_path": ["Test"],
        "source_uri": "https://example.invalid/test",
        "rendered_text": "alpha beta gamma delta",
    }
    chunk = _make_chunk(config, tokenizer, [unit], 1, 3)
    assert chunk["text"] == "beta gamma"
    assert chunk["token_count"] == 2
    assert chunk["span_fragments"] == [
        {
            "evidence_unit_id": digest,
            "source_span_ids": ["c" * 32],
            "source_uri": "https://example.invalid/test",
            "anchor": None,
            "unit_token_start": 1,
            "unit_token_end": 3,
            "chunk_token_start": 0,
            "chunk_token_end": 2,
            "unit_char_start": 6,
            "unit_char_end": 16,
        }
    ]


def test_frozen_evidence_manifest_matches_every_output() -> None:
    manifest = verify_evidence_snapshot()
    assert manifest["deterministic_double_run"] is True
    assert manifest["counts"]["application_documents"] == 935


def test_c3_backfills_short_parent_tail_when_hard_limit_allows() -> None:
    paths = LabPaths.discover()
    spec = _model_spec(paths)
    tokenizer = _tokenizer(paths, spec)
    configs = _chunk_configs(spec)

    def unit(identifier: str, token_count: int) -> dict:
        return {
            "id": identifier * 32,
            "source_document_id": "b" * 32,
            "source_span_ids": [identifier * 16 + "c" * 16],
            "section_id": identifier * 16 + "d" * 16,
            "parent_section_id": "e" * 32,
            "heading_path": ["Parent", identifier],
            "source_uri": f"https://example.invalid/{identifier}",
            "anchor": identifier,
            "rendered_text": " ".join(["alpha"] * token_count),
        }

    c2 = [
        _make_chunk(configs["C2-structure-bounded"], tokenizer, [unit("1", 380)]),
        _make_chunk(configs["C2-structure-bounded"], tokenizer, [unit("2", 100)]),
    ]
    merged = _c3(configs["C3-structure-merged"], tokenizer, c2)
    assert len(merged) == 1
    assert merged[0]["token_count"] == 480

    c2 = [
        _make_chunk(configs["C2-structure-bounded"], tokenizer, [unit("3", 380)]),
        _make_chunk(configs["C2-structure-bounded"], tokenizer, [unit("4", 80)]),
        _make_chunk(configs["C2-structure-bounded"], tokenizer, [unit("5", 450)]),
    ]
    rebalanced = _c3(configs["C3-structure-merged"], tokenizer, c2)
    assert [chunk["token_count"] for chunk in rebalanced] == [460, 450]

    top_level = unit("6", 100)
    top_level["parent_section_id"] = None
    top_level["section_id"] = "p" * 32
    child = unit("7", 100)
    child["parent_section_id"] = "p" * 32
    child["section_id"] = "q" * 32
    separated = _c3(
        configs["C3-structure-merged"],
        tokenizer,
        [
            _make_chunk(configs["C2-structure-bounded"], tokenizer, [top_level]),
            _make_chunk(configs["C2-structure-bounded"], tokenizer, [child]),
        ],
    )
    assert len(separated) == 2


def test_seed50_topic_briefs_match_preregistered_distribution() -> None:
    summary = validate_topic_briefs()
    assert summary["topics"] == 50
    assert summary["smoke_topics"] == 12


def test_bm25_ties_are_broken_by_chunk_id() -> None:
    assert _rank(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), ["b", "a", "c"], 3) == [2, 1, 0]
    assert _rank(np.asarray([1.0, 1.0, 2.0], dtype=np.float32), ["b", "a", "c"], 2) == [2, 1]


def test_ir_metrics_golden_fixture_uses_grade_two_binary_threshold() -> None:
    metrics = evaluate_ranking(
        {"q": {"a": 3, "b": 1, "c": 0}},
        {"q": {"a": 2.0, "x": 1.0, "b": 0.5}, "no-answer": {"z": 9.0}},
    )
    assert metrics["ndcg_at_10"] == pytest.approx(0.9639404333166532)
    assert metrics["p_at_3"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["map"] == 1.0
    assert metrics["judged_at_10"] == pytest.approx(2 / 3)
    assert metrics["bpref"] == 1.0


def test_evidence_cards_remove_duplicate_fragments_and_respect_budget() -> None:
    fragment = {
        "evidence_unit_id": "unit",
        "source_span_ids": ["span"],
        "source_uri": "https://example.invalid/doc#anchor",
        "anchor": "anchor",
        "unit_token_start": 0,
        "unit_token_end": 2,
        "chunk_token_start": 0,
        "chunk_token_end": 2,
        "unit_char_start": 0,
        "unit_char_end": 10,
    }
    chunks = {
        chunk_id: {
            "id": chunk_id,
            "source_uri": "https://example.invalid/doc#wrong-chunk-anchor",
            "span_fragments": [fragment],
        }
        for chunk_id in ("chunk-a", "chunk-b")
    }
    cards, diagnostics = assemble_evidence_cards(
        "query",
        [
            {"rank": 1, "chunk_id": "chunk-a", "score": 2.0},
            {"rank": 2, "chunk_id": "chunk-b", "score": 1.0},
        ],
        chunks,
        {"unit": {"rendered_text": "alpha beta"}},
        token_budget=2,
    )
    assert [card["chunk_id"] for card in cards] == ["chunk-a"]
    assert cards[0]["token_count"] == cards[0]["cumulative_token_count"] == 2
    assert cards[0]["source_uri"].endswith("#anchor")
    assert cards[0]["source_uri"].rsplit("#", 1)[1] == cards[0]["anchor"]
    assert diagnostics["selected_tokens"] == 2


def test_e0_uses_lexical_text_without_leaking_past_partial_fragment() -> None:
    unit = {
        "id": "unit",
        "heading_path": ["Parent", "Section"],
        "rendered_text": "alpha beta",
        "lexical_text": "Parent > Section alpha beta",
    }
    base = {
        "heading_path": ["Parent", "Section"],
        "span_fragments": [
            {
                "evidence_unit_id": "unit",
                "unit_token_start": 0,
                "unit_token_end": 2,
                "unit_char_start": 0,
                "unit_char_end": 10,
            }
        ],
    }
    assert "Parent > Section alpha beta" in _lexical_chunk_text(base, {"unit": unit}, {"unit": 2})
    base["span_fragments"][0]["unit_token_end"] = 1
    base["span_fragments"][0]["unit_char_end"] = 5
    partial = _lexical_chunk_text(base, {"unit": unit}, {"unit": 2})
    assert "alpha" in partial and "beta" not in partial


def test_evidence_card_last_cut_uses_sentence_boundary() -> None:
    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"offset_mapping": [(0, 5), (5, 6), (7, 11), (11, 12)]}

    fragment = {
        "evidence_unit_id": "unit",
        "source_span_ids": ["span"],
        "source_uri": "https://example.invalid/doc#anchor",
        "anchor": "anchor",
        "unit_token_start": 0,
        "unit_token_end": 4,
        "chunk_token_start": 0,
        "chunk_token_end": 4,
        "unit_char_start": 0,
        "unit_char_end": 12,
    }
    cards, _ = assemble_evidence_cards(
        "query",
        [{"rank": 1, "chunk_id": "chunk", "score": 1.0}],
        {
            "chunk": {
                "id": "chunk",
                "source_uri": "https://example.invalid/doc#anchor",
                "span_fragments": [fragment],
            }
        },
        {"unit": {"rendered_text": "alpha. beta."}},
        token_budget=2,
        tokenizer=Tokenizer(),
    )
    assert cards[0]["excerpt"] == "alpha."
    assert cards[0]["token_count"] == 2
    assert cards[0]["truncated"] is True
    assert cards[0]["cut"]["kind"] == "sentence_boundary"


def test_logical_chunk_score_is_max_subwindow() -> None:
    scores = _aggregate_window_scores(
        np.asarray([0.2, 0.9, 0.4], dtype=np.float32),
        np.asarray([0, 0, 1], dtype=np.int32),
        2,
    )
    assert scores.tolist() == pytest.approx([0.9, 0.4])


def test_rrf_uses_fixed_k_and_chunk_id_tie_break() -> None:
    sparse = {
        "q": [
            {"rank": 1, "chunk_id": "b"},
            {"rank": 2, "chunk_id": "a"},
        ]
    }
    dense = {
        "q": [
            {"rank": 1, "chunk_id": "a"},
            {"rank": 2, "chunk_id": "b"},
        ]
    }
    ranking = _hybrid(sparse, dense)["q"]
    assert [item["chunk_id"] for item in ranking] == ["a", "b"]
    assert ranking[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_tuning_rrf_weight_and_tmm_are_deterministic() -> None:
    sparse = {"q": [{"rank": 1, "chunk_id": "a", "score": 3.0}]}
    dense = {"q": [{"rank": 1, "chunk_id": "b", "score": 0.9}]}
    assert _rrf(sparse, dense, k=60, sparse_weight=2.0)["q"][0]["chunk_id"] == "a"
    chunks = [{"id": "a"}, {"id": "b"}]
    queries = [{"id": "q"}]
    tmm = _tmm_convex(
        np.asarray([[3.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.9]], dtype=np.float32),
        chunks,
        queries,
        semantic_alpha=0.8,
        depth=2,
    )["q"]
    assert [row["chunk_id"] for row in tmm] == ["b", "a"]


def test_bm25f_combines_field_tf_before_saturation() -> None:
    low = BM25F(
        [["needle"], []],
        [[], ["needle"]],
        k1=1.5,
        b_heading=0.0,
        b_body=0.0,
        heading_boost=1.0,
        body_boost=1.0,
    ).score(["needle"])
    high = BM25F(
        [["needle"], []],
        [[], ["needle"]],
        k1=1.5,
        b_heading=0.0,
        b_body=0.0,
        heading_boost=4.0,
        body_boost=1.0,
    ).score(["needle"])
    assert low[0] == pytest.approx(low[1])
    assert high[0] > high[1]


def test_refresh_rejects_tampered_persisted_trec_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    run_path = run_root / "run.trec"
    run_path.write_text("q Q0 c 1 1.0 tag\n", encoding="utf-8")
    manifest = {
        "id": "run-id",
        "chunk_config_id": "C2-structure-bounded",
        "system": "E0-BM25",
        "output_hashes": {"run.trec": sha256_file(run_path)},
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    entry = {
        "run_id": "run-id",
        "chunk_config_id": "C2-structure-bounded",
        "system": "E0-BM25",
        "manifest_sha256": sha256_file(manifest_path),
    }
    assert _read_approved_trec_run(entry, run_root)["q"][0]["chunk_id"] == "c"
    run_path.write_text("q Q0 evil 1 9.0 tag\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="TREC run hash mismatch"):
        _read_approved_trec_run(entry, run_root)


def test_refresh_preflights_every_run_before_any_write(tmp_path: Path) -> None:
    entries = []
    for run_id in ("first", "second"):
        run_root = tmp_path / run_id
        run_root.mkdir()
        run_path = run_root / "run.trec"
        run_path.write_text(f"q Q0 {run_id} 1 1.0 tag\n", encoding="utf-8")
        manifest = {
            "id": run_id,
            "chunk_config_id": "C2-structure-bounded",
            "system": "E0-BM25",
            "output_hashes": {"run.trec": sha256_file(run_path)},
        }
        manifest_path = run_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        entries.append(
            {
                "run_id": run_id,
                "chunk_config_id": "C2-structure-bounded",
                "system": "E0-BM25",
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    (tmp_path / "second" / "run.trec").write_text(
        "q Q0 tampered 1 9.0 tag\n", encoding="utf-8"
    )
    write_calls = []
    with pytest.raises(RuntimeError, match="TREC run hash mismatch"):
        rankings = _preflight_persisted_rankings(entries, tmp_path, {"q"})
        write_calls.extend(rankings)
    assert write_calls == []


def test_pool_provenance_is_system_specific_and_canonical() -> None:
    assert _run_provenance("E0-BM25") == "lexical_candidate"
    assert _run_provenance("E1-dense-exact") == "semantic_candidate"
    assert _run_provenance("E3-rerank") == "rerank_candidate"
    assert _application_family("documentation/content/applications/finance/accounting.rst") == "finance"


def test_expert_siblings_are_adjacent_heading_units_only() -> None:
    headings = [
        {
            "id": value,
            "node_type": "heading",
            "source_document_id": "doc",
            "section_id": value,
            "parent_section_id": "parent",
            "ordinal": index,
        }
        for index, value in enumerate(("left", "middle", "right"))
    ]
    by_section = {("doc", item["section_id"]): item for item in headings}
    by_parent = {("doc", "parent"): headings}
    paragraph = {**headings[1], "id": "paragraph", "node_type": "paragraph"}
    assert [item["id"] for item in _true_sibling_headings(paragraph, by_section, by_parent)] == [
        "left",
        "right",
    ]


def test_wrong_version_requires_explicit_reason_and_judged_nonrelevance() -> None:
    query = {"no_answer_reason": "wrong_version"}
    assert _is_wrong_version_candidate(query, 0)
    assert _is_wrong_version_candidate(query, 1)
    assert not _is_wrong_version_candidate(query, None)
    assert not _is_wrong_version_candidate(query, 2)
    assert not _is_wrong_version_candidate({"no_answer_reason": "out_of_scope"}, 0)


def test_pool_stability_requires_two_consecutive_added_depths() -> None:
    rows = [
        {"new_grade_2_or_3_yield": 0.02, "leave_one_run_out_min_kendall_tau": 1.0},
        {"new_grade_2_or_3_yield": 0.009, "leave_one_run_out_min_kendall_tau": 1.0},
        {"new_grade_2_or_3_yield": 0.008, "leave_one_run_out_min_kendall_tau": 1.0},
    ]
    assert _agent_pool_is_stable(rows)
    rows[-1]["leave_one_run_out_min_kendall_tau"] = 0.94
    assert not _agent_pool_is_stable(rows)


def test_pool_system_order_and_kendall_tau_are_deterministic() -> None:
    reference = _system_order({"E1": 0.5, "E0": 0.5, "E2": 0.4})
    assert reference == ["E0", "E1", "E2"]
    assert _order_tau(reference, reference) == pytest.approx(1.0)


def test_pool_review_label_and_negative_provenance_are_deterministic() -> None:
    row = {
        "retrieval_grade": 1,
        "selected_source_span_ids": ["span-1"],
        "selected_nugget_ids": [],
    }
    assert _annotation_label(row) == (1, ("span-1",), ())
    candidate = {
        "provenance": ["rerank_candidate", "semantic_candidate", "wrong_module_candidate"]
    }
    query = {"no_answer_reason": None}
    assert _negative_classes(candidate, query) == [
        "baseline_false_positive",
        "semantic_nearest",
        "wrong_module",
    ]
    assert _negative_classes(candidate, {"no_answer_reason": "wrong_version"}) == [
        "wrong_version"
    ]
    assert _reported_annotation_sha({"annotation_file_sha256": "abc"}) == "abc"
    assert _reported_annotation_sha(
        {"bound_outputs": [{"path": "x/adjudicator.jsonl", "sha256": "def"}]}
    ) == "def"


def test_performance_percentiles_use_milliseconds() -> None:
    summary = _percentiles([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p99_ms"] == pytest.approx(3.97)


def test_performance_framework_overhead_excludes_request_time() -> None:
    overhead, fraction = _framework_overhead(10_000, [4_500, 4_500])
    assert overhead == 1_000
    assert fraction == pytest.approx(0.1)


def test_performance_cold_load_includes_full_runtime_initialization() -> None:
    old_receipt = {"resource_samples": [{"monotonic_ns": 100}, {"monotonic_ns": 350}]}
    assert _runtime_initialization_ns(old_receipt) == 250
    assert _runtime_initialization_ns({"runtime_initialization_duration_ns": 400}) == 400


def test_p5_resume_receipt_requires_exact_execution_profile() -> None:
    receipt = {
        "run_id": "run",
        "chunk_config": "C2-structure-bounded",
        "system": "E2-hybrid-rrf",
        "mode": "warm",
        "device": "cuda",
        "dtype": "float16",
        "concurrency": 4,
        "minimum_requests": 50,
        "minimum_seconds": 1,
        "measured_loop_duration_ns": 1_000_000_000,
        "external_peak_process_tree_rss_bytes": 1,
        "requests": [{"query_id": f"q{index}"} for index in range(50)],
    }
    assert _matches(
        receipt,
        "run",
        "C2-structure-bounded",
        "E2-hybrid-rrf",
        "warm",
        "cuda",
        "float16",
        4,
        50,
        1,
    )
    assert not _matches(
        receipt,
        "run",
        "C3-structure-merged",
        "E2-hybrid-rrf",
        "warm",
        "cuda",
        "float16",
        4,
        50,
        1,
    )
    insufficient = {**receipt, "requests": receipt["requests"][:-1]}
    assert not _matches(
        insufficient,
        "run",
        "C2-structure-bounded",
        "E2-hybrid-rrf",
        "warm",
        "cuda",
        "float16",
        4,
        50,
        1,
    )


def test_p5_failure_updates_progress(monkeypatch, tmp_path: Path) -> None:
    paths = LabPaths(tmp_path)
    root = tmp_path / "artifacts" / "performance" / "p5" / "run"
    active = tmp_path / "artifacts" / "performance" / "p5" / "active.json"
    write_json(root / "progress.json", {"run_id": "run", "status": "running"}, paths)
    write_json(active, {"run_id": "run", "root": str(root), "status": "running"}, paths)
    monkeypatch.setattr(p5, "_run_p5_impl", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        p5.run_p5(paths, minimum_seconds=1, minimum_requests=50, cold_processes=1)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["error_type"] == "RuntimeError"


def test_all_seed_runs_only_after_every_agent_approval(monkeypatch, tmp_path: Path) -> None:
    paths = LabPaths(tmp_path)
    approved: list[str] = []
    executed: list[str] = []
    monkeypatch.setattr(
        cli,
        "require_approval",
        lambda phase, _, *args: approved.append(phase) or {},
    )
    monkeypatch.setattr(
        cli,
        "create_environment_receipt",
        lambda _: executed.append("verify") or tmp_path / "environment.json",
    )
    for name in (
        "extract_twice",
        "chunk_twice",
        "run_smoke",
        "run_baseline",
        "build_pool",
        "run_performance",
    ):
        monkeypatch.setattr(
            cli,
            name,
            lambda _, stage=name: executed.append(stage) or {"stage": stage},
        )
    result = cli.run_seed_pipeline(paths, {"status": "pass"})
    assert approved == [*cli.SEED_APPROVALS, "seed50-diagnostics"]
    assert executed == [
        "verify",
        "extract_twice",
        "chunk_twice",
        "run_smoke",
        "run_baseline",
        "build_pool",
        "run_performance",
    ]
    assert result["status"] == "agent_provisional_complete_human_review_pending"
    assert result["human_review_complete"] is result["seed_frozen"] is False
    assert list(result["stages"]) == [
        "verify",
        "extract",
        "chunk",
        "smoke",
        "baseline",
        "pool",
        "perf",
    ]


def test_statistical_diagnostics_are_grouped_and_deterministic() -> None:
    rows = [
        {
            "query_id": f"q{index}",
            "answerable": int(index % 5 != 0),
            "source_fact_group": f"g{index // 2}",
            "score": float(20 - index),
        }
        for index in range(20)
    ]
    result = _answerability_cv(rows)
    seen = set()
    for fold in result["folds"]:
        assert fold["test_query_ids"]
        assert not (seen & set(fold["test_source_fact_groups"]))
        seen.update(fold["test_source_fact_groups"])
    assert seen == {row["source_fact_group"] for row in rows}
    assert 0 <= result["abstention_precision"] <= 1
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    qrels = {"a": 3, "b": 2}
    assert _ndcg10(
        qrels, [{"chunk_id": "a", "score": 2.0}, {"chunk_id": "b", "score": 1.0}]
    ) == pytest.approx(1.0)
    assert _auc_ap(np.asarray([1, 0]), np.asarray([1.0, 1.0]))[1] == pytest.approx(0.5)


def test_depth20_annotation_package_binds_all_inputs_and_outputs() -> None:
    root = LabPaths.discover().root
    pool = root / "benchmarks" / "seed50" / "pooling" / "provisional"
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    package_path = pool / "annotations" / "manifest.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert manifest["annotation_package_sha256"] == sha256_file(package_path)
    assert package["human_review_complete"] is package["pooling_stable"] is package["seed_frozen"] is False
    for name, digest in package["annotation_hashes"].items():
        assert sha256_file(pool / "annotations" / name) == digest
    agreement = json.loads(
        (pool / "annotations" / "agreement_report.json").read_text(encoding="utf-8")
    )
    assert agreement["pool_contract_sha256"] == package["pool_contract_sha256"]
    assert agreement["input_hashes"] == package["input_hashes"]
