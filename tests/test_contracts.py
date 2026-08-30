from __future__ import annotations

import json
from pathlib import Path

import pytest
from docutils import nodes

from osrlab.contract import validate_record
from osrlab.chunking import _c1_scoring_windows, _c3, _chunk_configs, _make_chunk, _model_spec, _tokenizer, _windows, verify_evidence_snapshot
from osrlab.extraction import EvidenceCollector, _normalize_rendered
from osrlab.jsonio import write_json, write_jsonl
from osrlab.paths import LabPaths, PathBoundaryError


def test_path_allowlist_rejects_source_and_external_paths(tmp_path: Path) -> None:
    root = tmp_path / "lab"
    paths = LabPaths(root)
    assert paths.require_write_path("artifacts/run.json") == (root / "artifacts/run.json").resolve()
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
