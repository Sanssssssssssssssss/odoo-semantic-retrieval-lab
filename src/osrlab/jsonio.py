from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import LabPaths


def canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, record: Mapping[str, Any], paths: LabPaths | None = None) -> None:
    path = (paths or LabPaths.discover()).require_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(record) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    sort_key: str = "id",
    paths: LabPaths | None = None,
) -> None:
    path = (paths or LabPaths.discover()).require_write_path(path)
    ordered = sorted(records, key=lambda item: str(item.get(sort_key, "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(canonical_json(record) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: str, length: int = 32) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
