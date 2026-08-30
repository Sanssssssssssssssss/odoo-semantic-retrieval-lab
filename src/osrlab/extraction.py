from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from docutils import nodes
from sphinx import __version__ as sphinx_version
from sphinx import addnodes
from sphinx.application import Sphinx

from .contract import validate_record
from .jsonio import canonical_json, sha256_file, stable_id, write_json, write_jsonl
from .paths import LabPaths
from .verify import verify_source


SNAPSHOT_ID = "32a8b8d77833f22b4bc74ed4ea78b6a82b5338fd"
BOUNDARY_DOCS = {
    "administration/odoo_sh/getting_started/online_editor",
    "developer/reference/user_interface/view_architectures",
}
CANONICAL_FILES = (
    "source_documents.jsonl",
    "boundary_source_documents.jsonl",
    "source_spans.jsonl",
    "evidence_units.jsonl",
    "boundary_evidence_units.jsonl",
    "exclusions.jsonl",
    "warnings.jsonl",
)
SUBSTITUTION_RE = re.compile(r"\|([^|\n]+)\|")
INCLUDE_RE = re.compile(r"^[ \t]*\.\.[ \t]+(?:literal)?include::[ \t]+(.+?)[ \t]*$", re.MULTILINE)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SOURCE_REPLACEMENT_KEYS = {
    "BRANCH",
    "CURRENT_BRANCH",
    "CURRENT_VERSION",
    "CURRENT_MAJOR_BRANCH",
    "CURRENT_MAJOR_VERSION",
    "GITHUB_PATH",
    "GITHUB_ENT_PATH",
    "GITHUB_TUTO_PATH",
    "OWL_PATH",
}


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_rendered(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    output: list[str] = []
    for line in lines:
        if line or not output or output[-1]:
            output.append(line)
    return "\n".join(output)


def _lexical(text: str) -> str:
    return " ".join(text.split())


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class EvidenceCollector:
    def __init__(self, paths: LabPaths) -> None:
        self.paths = paths
        self.content = paths.docs / "content"
        self.documents: list[dict[str, Any]] = []
        self.boundary_documents: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.units: list[dict[str, Any]] = []
        self.boundary_units: list[dict[str, Any]] = []
        self.exclusions: list[dict[str, Any]] = []
        self._lines: dict[Path, list[str]] = {}
        self._include_sites: dict[str, dict[Path, int]] = {}

    def prepare_documents(self) -> None:
        for source in sorted((self.content / "applications").rglob("*.rst")):
            docname = source.relative_to(self.content).with_suffix("").as_posix()
            record = self._document_record(docname, source, "applications")
            validate_record("SourceDocument", record, self.paths)
            self.documents.append(record)

    def _document_record(self, docname: str, source: Path, scope: str) -> dict[str, Any]:
        source_path = source.relative_to(self.content).as_posix()
        return {
            "id": stable_id(SNAPSHOT_ID, source_path),
            "source_snapshot_id": SNAPSHOT_ID,
            "source_path": source_path,
            "source_sha256": sha256_file(source),
            "public_uri": f"https://www.odoo.com/documentation/19.0/{docname}.html",
            "scope": scope,
        }

    def capture(self, app: Sphinx, doctree: nodes.document, docname: str) -> None:
        if not (docname.startswith("applications/") or docname in BOUNDARY_DOCS):
            return
        source = (self.content / f"{docname}.rst").resolve()
        if not source.is_file():
            return
        scope = "applications" if docname.startswith("applications/") else "boundary"
        document = self._document_record(docname, source, scope)
        if scope == "boundary" and not any(item["id"] == document["id"] for item in self.boundary_documents):
            validate_record("SourceDocument", document, self.paths)
            self.boundary_documents.append(document)
        substitutions = {
            name: definition
            for definition in doctree.traverse(nodes.substitution_definition)
            for name in definition.get("names", [])
        }
        self._include_sites[docname] = self._scan_include_sites(source)
        ordinal = 0
        for position, node in enumerate(doctree.traverse(nodes.Element)):
            selected = self._selected(node)
            if selected is None:
                continue
            node_type, rendered = selected
            rendered = _normalize_rendered(rendered)
            if not rendered:
                continue
            refs = self._span_refs(node, source, docname, substitutions)
            if not refs:
                self.exclusions.append(
                    {
                        "id": stable_id(docname, str(position), node_type, str(getattr(node, "line", "")), rendered),
                        "docname": docname,
                        "node_type": node_type,
                        "reason": "no_trustworthy_source_range",
                        "rendered_text_sha256": _digest_text(rendered),
                    }
                )
                continue
            headings, section_ids = self._section_context(node, docname)
            anchor = section_ids[-1]["anchor"] if section_ids else None
            span = {
                "id": stable_id(document["id"], str(position), node_type, canonical_json({"refs": refs}), rendered),
                "source_document_id": document["id"],
                "span_refs": refs,
            }
            unit = {
                "id": stable_id(SNAPSHOT_ID, document["id"], node_type, str(ordinal), span["id"], _digest_text(rendered)),
                "source_document_id": document["id"],
                "source_span_ids": [span["id"]],
                "node_type": node_type,
                "ordinal": ordinal,
                "heading_path": headings,
                "section_id": section_ids[-1]["id"] if section_ids else stable_id(document["id"], "document-root"),
                "parent_section_id": section_ids[-2]["id"] if len(section_ids) > 1 else None,
                "anchor": anchor,
                "anchors": section_ids[-1]["anchors"] if section_ids else [],
                "source_uri": document["public_uri"] + (f"#{anchor}" if anchor else ""),
                "structure_context": self._structure_context(node),
                "rendered_text": rendered,
                "lexical_text": _lexical(" > ".join(headings) + " " + rendered),
                "cross_references": self._cross_references(node),
                "content_sha256": _digest_text(rendered),
            }
            validate_record("SourceSpan", span, self.paths)
            validate_record("EvidenceUnit", unit, self.paths)
            self.spans.append(span)
            (self.units if scope == "applications" else self.boundary_units).append(unit)
            ordinal += 1

    def _selected(self, node: nodes.Element) -> tuple[str, str] | None:
        if isinstance(node, nodes.title):
            return "heading", node.astext()
        if isinstance(node, nodes.literal_block):
            return "code", node.astext()
        if isinstance(node, nodes.image):
            return "image_alt", node.get("alt", "")
        if isinstance(node, nodes.row):
            return "table_row", " | ".join(entry.astext() for entry in node.children if isinstance(entry, nodes.entry))
        if isinstance(node, nodes.list_item):
            text = "\n".join(
                child.astext()
                for child in node.children
                if not isinstance(child, (nodes.bullet_list, nodes.enumerated_list))
            )
            return "list_item", text
        if isinstance(node, nodes.Admonition):
            return f"admonition:{type(node).__name__}", node.astext()
        if isinstance(node, nodes.paragraph) and not self._has_ancestor(
            node, (nodes.list_item, nodes.row, nodes.Admonition, nodes.literal_block)
        ):
            return "paragraph", node.astext()
        return None

    @staticmethod
    def _has_ancestor(node: nodes.Node, kinds: tuple[type, ...]) -> bool:
        parent = node.parent
        while parent is not None:
            if isinstance(parent, kinds):
                return True
            parent = parent.parent
        return False

    def _section_context(self, node: nodes.Element, docname: str) -> tuple[list[str], list[dict[str, Any]]]:
        sections: list[nodes.section] = []
        current: nodes.Node | None = node
        while current is not None:
            if isinstance(current, nodes.section):
                sections.append(current)
            current = current.parent
        sections.reverse()
        headings: list[str] = []
        ids: list[dict[str, Any]] = []
        for section in sections:
            title = next((child.astext() for child in section.children if isinstance(child, nodes.title)), "")
            if title:
                headings.append(_lexical(title))
            anchors = []
            for child in section.children:
                if isinstance(child, nodes.target):
                    anchors.extend(child.get("ids", []))
                elif not isinstance(child, (nodes.title, nodes.substitution_definition)):
                    break
            anchors.extend(section.get("ids", []))
            anchors = list(dict.fromkeys(anchors))
            anchor = anchors[0] if anchors else stable_id(docname, *headings, length=16)
            ids.append({"id": stable_id(SNAPSHOT_ID, docname, anchor), "anchor": anchor, "anchors": anchors})
        return headings, ids

    def _span_refs(
        self,
        node: nodes.Element,
        document_source: Path,
        docname: str,
        substitutions: dict[str, nodes.substitution_definition],
    ) -> list[dict[str, Any]]:
        candidates = list(node.traverse(nodes.Element))
        if (
            len(candidates) > 1
            and not getattr(node, "line", None)
            and any(getattr(candidate, "line", None) for candidate in candidates[1:])
        ):
            candidates = candidates[1:]
        grouped: dict[Path, list[tuple[int, int]]] = defaultdict(list)
        for candidate in candidates:
            source_text = getattr(candidate, "source", None)
            line = getattr(candidate, "line", None)
            if not source_text:
                continue
            source = Path(source_text).resolve()
            if not source.is_file() or not source.is_relative_to(self.content):
                continue
            raw = getattr(candidate, "rawsource", "") or ""
            if raw:
                line = self._locate_rawsource(source, raw, int(line) if line else None) or line
            if not line:
                continue
            line_count = max(1, len(raw.splitlines()))
            grouped[source].append((int(line), int(line) + line_count - 1))
        refs: list[dict[str, Any]] = []
        section_anchors = self._section_context(node, docname)[1]
        anchor = section_anchors[-1]["anchor"] if section_anchors else None
        merged = {source: self._merge_ranges(ranges) for source, ranges in grouped.items()}
        for source in sorted(merged, key=lambda path: path.as_posix()):
            origin = "direct" if source == document_source else "include"
            for start, end in merged[source]:
                ref = self._source_ref(source, start, end, anchor, origin)
                if ref:
                    refs.append(ref)
            if origin == "include" and source in self._include_sites.get(docname, {}):
                include_line = self._include_sites[docname][source]
                site_ref = self._source_ref(document_source, include_line, include_line, anchor, "direct")
                if site_ref:
                    refs.append(site_ref)
        rawsource = getattr(node, "rawsource", "") or ""
        for name in SUBSTITUTION_RE.findall(rawsource):
            definition = substitutions.get(name)
            if definition is None or not definition.source or not definition.line:
                continue
            source = Path(definition.source).resolve()
            ref = self._source_ref(source, definition.line, definition.line + max(0, len(definition.rawsource.splitlines()) - 1), anchor, "substitution")
            if ref:
                refs.append(ref)
        direct_lines = self._source_lines(document_source)
        direct_ranges = merged.get(document_source, [])
        if direct_ranges:
            for start, end in direct_ranges:
                end = min(len(direct_lines), end)
                if any(
                    f"{{{key}}}" in line
                    for line in direct_lines[start - 1 : end]
                    for key in SOURCE_REPLACEMENT_KEYS
                ):
                    generated = self._source_ref(document_source, start, end, anchor, "generated")
                    if generated:
                        refs.append(generated)
        unique = {canonical_json(ref): ref for ref in refs}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[list[int]] = []
        for start, end in sorted(set(ranges)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _locate_rawsource(self, source: Path, rawsource: str, hint_line: int | None = None) -> int | None:
        source_text = "\n".join(self._source_lines(source))
        rawsource = rawsource.replace("\r\n", "\n").replace("\r", "\n")
        offsets: list[int] = []
        offset = source_text.find(rawsource)
        while offset >= 0:
            offsets.append(offset)
            offset = source_text.find(rawsource, offset + 1)
        lines = [source_text.count("\n", 0, item) + 1 for item in offsets]
        if not lines:
            source_lines = self._source_lines(source)
            raw_lines = rawsource.split("\n")
            lines = [
                index + 1
                for index in range(len(source_lines) - len(raw_lines) + 1)
                if all(source_lines[index + offset].strip() == raw_line.strip() for offset, raw_line in enumerate(raw_lines))
            ]
        if not lines:
            return None
        return min(lines, key=lambda item: abs(item - hint_line)) if hint_line else lines[0]

    def _source_ref(
        self,
        source: Path,
        start: int,
        end: int,
        anchor: str | None,
        origin_kind: str,
    ) -> dict[str, Any] | None:
        if not source.is_file() or not source.is_relative_to(self.content):
            return None
        lines = self._source_lines(source)
        if start < 1 or start > len(lines):
            return None
        end = max(start, min(end, len(lines)))
        quote = "\n".join(lines[start - 1 : end])
        return {
            "source_path": source.relative_to(self.content).as_posix(),
            "source_sha256": sha256_file(source),
            "start_line": start,
            "start_column": len(lines[start - 1]) - len(lines[start - 1].lstrip()) + 1,
            "end_line": end,
            "end_column": len(lines[end - 1]) + 1,
            "anchor": anchor,
            "quote_sha256": _digest_text(quote),
            "origin_kind": origin_kind,
        }

    def _source_lines(self, source: Path) -> list[str]:
        if source not in self._lines:
            self._lines[source] = source.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return self._lines[source]

    def _scan_include_sites(self, source: Path) -> dict[Path, int]:
        text = source.read_text(encoding="utf-8-sig")
        result: dict[Path, int] = {}
        for match in INCLUDE_RE.finditer(text):
            target = match.group(1).strip().strip('"\'')
            candidate = (source.parent / target).resolve()
            result[candidate] = text.count("\n", 0, match.start()) + 1
        return result

    @staticmethod
    def _cross_references(node: nodes.Element) -> list[dict[str, str]]:
        references: list[dict[str, str]] = []
        for reference in node.traverse(lambda child: isinstance(child, (nodes.reference, addnodes.pending_xref))):
            target = reference.get("refuri") or reference.get("refid") or reference.get("reftarget")
            if target:
                references.append({"text": _lexical(reference.astext()), "target": str(target)})
        return references

    @staticmethod
    def _structure_context(node: nodes.Element) -> list[dict[str, str]]:
        current: nodes.Node = node
        while current.parent is not None:
            parent = current.parent
            if isinstance(parent, nodes.container) and "sphinx-tabs" in parent.get("classes", []):
                label = next((candidate.astext() for candidate in current.traverse(nodes.paragraph)), "")
                return [{"kind": "tab", "label": _lexical(label)}] if label else [{"kind": "tab", "label": ""}]
            current = parent
        return []


def run_worker(output: Path) -> dict[str, Any]:
    paths = LabPaths.discover()
    verify_source(paths)
    output = paths.require_write_path(output)
    output.mkdir(parents=True, exist_ok=True)
    build_out = paths.require_write_path(output / "sphinx-out")
    doctrees = paths.require_write_path(output / "doctrees")
    status = io.StringIO()
    warning = io.StringIO()
    collector = EvidenceCollector(paths)
    collector.prepare_documents()
    with _cwd(paths.docs):
        app = Sphinx(
            srcdir=str(paths.docs / "content"),
            confdir=str(paths.docs),
            outdir=str(build_out),
            doctreedir=str(doctrees),
            buildername="dummy",
            status=status,
            warning=warning,
            freshenv=True,
            warningiserror=False,
            parallel=1,
        )
        app.connect("doctree-resolved", collector.capture)
        app.build(force_all=True)
    warnings = [ANSI_RE.sub("", line).strip() for line in warning.getvalue().splitlines() if line.strip()]
    unknown = [line for line in warnings if "Unknown directive type" in line or "Unknown interpreted text role" in line]
    if unknown:
        raise RuntimeError("Unknown Sphinx constructs detected: " + " | ".join(unknown))
    warning_records = [{"id": stable_id(str(index), line), "message": line} for index, line in enumerate(warnings)]
    for document in collector.documents:
        if not any(unit["source_document_id"] == document["id"] for unit in collector.units):
            collector.exclusions.append(
                {
                    "id": stable_id(document["id"], "document_without_evidence"),
                    "docname": document["source_path"],
                    "node_type": "document",
                    "reason": "document_without_evidence_or_explicit_exclusion",
                    "rendered_text_sha256": _digest_text(""),
                }
            )
    write_jsonl(output / "source_documents.jsonl", collector.documents, paths=paths)
    write_jsonl(output / "boundary_source_documents.jsonl", collector.boundary_documents, paths=paths)
    write_jsonl(output / "source_spans.jsonl", collector.spans, paths=paths)
    write_jsonl(output / "evidence_units.jsonl", collector.units, paths=paths)
    write_jsonl(output / "boundary_evidence_units.jsonl", collector.boundary_units, paths=paths)
    write_jsonl(output / "exclusions.jsonl", collector.exclusions, paths=paths)
    write_jsonl(output / "warnings.jsonl", warning_records, paths=paths)
    manifest = {
        "schema_version": 1,
        "source_snapshot_id": SNAPSHOT_ID,
        "sphinx_version": sphinx_version,
        "build": {"freshenv": True, "force_all": True, "parallel": 1, "builder": "dummy"},
        "counts": {
            "application_documents": len(collector.documents),
            "application_evidence_units": len(collector.units),
            "source_spans": len(collector.spans),
            "boundary_evidence_units": len(collector.boundary_units),
            "exclusions": len(collector.exclusions),
            "warnings": len(warning_records),
        },
    }
    write_json(output / "worker_manifest.json", manifest, paths)
    return manifest


def validate_extraction_output(output: Path, paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()

    def load(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (output / name).read_text(encoding="utf-8").splitlines()]

    documents = load("source_documents.jsonl")
    boundary_documents = load("boundary_source_documents.jsonl")
    spans = load("source_spans.jsonl")
    units = load("evidence_units.jsonl")
    boundary_units = load("boundary_evidence_units.jsonl")
    exclusions = load("exclusions.jsonl")
    warnings = load("warnings.jsonl")
    for records, name in (
        (documents, "source documents"),
        (boundary_documents, "boundary documents"),
        (spans, "source spans"),
        (units, "evidence units"),
        (boundary_units, "boundary evidence units"),
        (exclusions, "exclusions"),
    ):
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate IDs in {name}")
    if len(documents) != 935:
        raise RuntimeError(f"Expected 935 applications documents, got {len(documents)}")
    document_ids = {record["id"] for record in [*documents, *boundary_documents]}
    span_by_id = {record["id"]: record for record in spans}
    unit_document_ids = {unit["source_document_id"] for unit in units}
    excluded_documents = {item.get("docname") for item in exclusions if item["node_type"] == "document"}
    missing = [
        document["source_path"]
        for document in documents
        if document["id"] not in unit_document_ids and document["source_path"] not in excluded_documents
    ]
    if missing:
        raise RuntimeError(f"Applications documents lack evidence or exclusion: {missing[:5]}")
    file_cache: dict[Path, tuple[str, list[str]]] = {}
    quote_by_ref: dict[str, str] = {}
    for span in spans:
        if span["source_document_id"] not in document_ids:
            raise RuntimeError(f"Span references unknown source document: {span['id']}")
        for ref in span["span_refs"]:
            source = (paths.docs / "content" / ref["source_path"]).resolve()
            if not source.is_relative_to(paths.docs / "content") or not source.is_file():
                raise RuntimeError(f"Span source is outside corpus: {ref['source_path']}")
            if source not in file_cache:
                file_cache[source] = (
                    sha256_file(source),
                    source.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").split("\n"),
                )
            source_hash, lines = file_cache[source]
            if source_hash != ref["source_sha256"]:
                raise RuntimeError(f"Span source hash mismatch: {span['id']}")
            quote = "\n".join(lines[ref["start_line"] - 1 : ref["end_line"]])
            if _digest_text(quote) != ref["quote_sha256"]:
                raise RuntimeError(f"Span quote hash mismatch: {span['id']}")
            quote_by_ref[canonical_json(ref)] = quote
    for unit in [*units, *boundary_units]:
        if any(span_id not in span_by_id for span_id in unit["source_span_ids"]):
            raise RuntimeError(f"Evidence unit references unknown span: {unit['id']}")
    node_types = {unit["node_type"].split(":", 1)[0] for unit in units}
    required_types = {"heading", "paragraph", "list_item", "table_row", "code", "admonition", "image_alt"}
    if not required_types.issubset(node_types):
        raise RuntimeError(f"Missing required structural node types: {sorted(required_types - node_types)}")
    refs = [ref for span in spans for ref in span["span_refs"]]
    heading_mismatches: list[str] = []
    for unit in (item for item in units if item["node_type"] == "heading"):
        rendered_tokens = _word_tokens(unit["rendered_text"])
        direct_quotes = [
            quote_by_ref[canonical_json(ref)]
            for span_id in unit["source_span_ids"]
            for ref in span_by_id[span_id]["span_refs"]
            if ref["origin_kind"] == "direct"
        ]
        best_overlap = max(
            (len(rendered_tokens & _word_tokens(quote)) / max(1, len(rendered_tokens)) for quote in direct_quotes),
            default=0.0,
        )
        if best_overlap < 0.8:
            heading_mismatches.append(unit["id"])
    readonly_code = next(
        (
            unit
            for unit in boundary_units
            if unit["node_type"] == "code" and '<field name="fname_a" readonly="True"/>' in unit["rendered_text"]
        ),
        None,
    )
    readonly_refs = (
        [ref for span_id in readonly_code["source_span_ids"] for ref in span_by_id[span_id]["span_refs"]]
        if readonly_code
        else []
    )
    include_body_ok = any(
        ref["origin_kind"] == "include"
        and ref["source_path"].endswith("field_attribute_readonly.rst")
        and ref["start_line"] == 10
        and ref["end_line"] == 11
        for ref in readonly_refs
    )
    include_site_ok = any(
        ref["origin_kind"] == "direct"
        and ref["source_path"].endswith("developer/reference/user_interface/view_architectures.rst")
        and ref["start_line"] == 1374
        and ref["end_line"] == 1374
        for ref in readonly_refs
    )
    generated_refs = [ref for ref in refs if ref["origin_kind"] == "generated"]
    generated_ok = bool(generated_refs) and all(
        any(f"{{{key}}}" in quote_by_ref[canonical_json(ref)] for key in SOURCE_REPLACEMENT_KEYS)
        for ref in generated_refs
    )
    checks = {
        "applications_documents": len(documents) == 935,
        "all_application_documents_covered": not missing,
        "control_bills_substitution": any(
            ref["origin_kind"] == "substitution" and ref["source_path"].endswith("control_bills.rst") for ref in refs
        ),
        "developer_include_boundary": any(
            ref["origin_kind"] == "include" and "view_architectures/" in ref["source_path"] for ref in refs
        ),
        "heading_quote_alignment": not heading_mismatches,
        "include_body_golden": include_body_ok,
        "include_site_golden": include_site_ok,
        "generated_provenance_golden": generated_ok,
        "span_tightness": all(ref["end_line"] - ref["start_line"] + 1 <= 100 for ref in refs),
        "tabs_preserved": any(unit["structure_context"] for unit in units),
        "cross_references_preserved": any(unit["cross_references"] for unit in units),
        "unknown_constructs_absent": not any(
            "Unknown directive type" in record["message"] or "Unknown interpreted text role" in record["message"]
            for record in warnings
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Extraction integrity checks failed: {checks}")
    return {
        "schema_version": 1,
        "status": "pass",
        "checks": checks,
        "counts": {
            "source_documents": len(documents),
            "source_spans": len(spans),
            "evidence_units": len(units),
            "boundary_evidence_units": len(boundary_units),
            "exclusions": len(exclusions),
        },
    }


def extract_twice(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    verify_source(paths)
    base = paths.require_write_path(paths.root / ".cache" / "extraction")
    for name in ("run-a", "run-b"):
        run_dir = paths.require_write_path(base / name)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        command = [sys.executable, "-m", "osrlab.extraction", "--worker", str(run_dir)]
        result = subprocess.run(command, cwd=paths.docs, capture_output=True, text=True, errors="replace")
        if result.returncode:
            raise RuntimeError(f"Extraction worker {name} failed:\n{result.stdout}\n{result.stderr}")
    first, second = base / "run-a", base / "run-b"
    mismatches = [name for name in CANONICAL_FILES if (first / name).read_bytes() != (second / name).read_bytes()]
    if mismatches:
        raise RuntimeError(f"Extraction is not byte deterministic: {mismatches}")
    final = paths.require_write_path(paths.root / "corpus" / "derived" / SNAPSHOT_ID / "evidence")
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True)
    for name in (*CANONICAL_FILES, "worker_manifest.json"):
        shutil.copyfile(first / name, final / name)
    snapshot_manifest_hash = sha256_file(paths.snapshot_manifest)
    source_snapshot = {
        "schema_version": 1,
        "repository_url": "https://github.com/odoo/documentation.git",
        "commit": SNAPSHOT_ID,
        "manifest_sha256": snapshot_manifest_hash,
    }
    validate_record("SourceSnapshot", source_snapshot, paths)
    write_json(final / "source_snapshot.json", source_snapshot, paths)
    output_hashes = {name: sha256_file(final / name) for name in CANONICAL_FILES}
    manifest = json.loads((first / "worker_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "id": stable_id(SNAPSHOT_ID, canonical_json(output_hashes)),
            "deterministic_double_run": True,
            "output_hashes": output_hashes,
        }
    )
    integrity = validate_extraction_output(final, paths)
    write_json(final / "integrity_report.json", integrity, paths)
    write_json(final / "extraction_manifest.json", manifest, paths)
    verify_source(paths)
    return {"output": str(final), **manifest}


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(args.worker), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
