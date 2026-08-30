from __future__ import annotations

import json
from pathlib import Path

import pytest
from docutils import nodes

from osrlab.contract import validate_record
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
