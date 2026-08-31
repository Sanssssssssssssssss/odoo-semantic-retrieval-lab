from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import LabPaths


def canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_lines(path: Path, lines: Iterable[str]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, record: Mapping[str, Any], paths: LabPaths | None = None) -> None:
    path = (paths or LabPaths.discover()).require_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_lines(path, (canonical_json(record) + "\n",))


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
    _atomic_lines(path, (canonical_json(record) + "\n" for record in ordered))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: str, length: int = 32) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
