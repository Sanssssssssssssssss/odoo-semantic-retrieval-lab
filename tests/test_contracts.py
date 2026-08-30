from __future__ import annotations

import json
from pathlib import Path

import pytest

from osrlab.contract import validate_record
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
                    "start_column": None,
                    "end_line": None,
                    "end_column": None,
                    "anchor": None,
                    "quote_sha256": digest,
                    "origin_kind": "include",
                },
            ],
        },
    )
