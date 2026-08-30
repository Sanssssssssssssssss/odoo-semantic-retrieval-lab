from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .jsonio import sha256_file
from .paths import LabPaths


EXPECTED_MANIFEST_SHA256 = "c779918daa43c93ef715c3a83ce759019a82629f23fb22c987fc1b0ec599dfad"


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_source(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    manifest_hash = sha256_file(paths.snapshot_manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Snapshot manifest hash mismatch: {manifest_hash}")
    manifest = json.loads(paths.snapshot_manifest.read_text(encoding="utf-8"))
    head = _git(paths.docs, "rev-parse", "HEAD")
    status = _git(paths.docs, "status", "--porcelain")
    symbolic = subprocess.run(
        ["git", "-C", str(paths.docs), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if head != manifest["commit"]:
        raise RuntimeError(f"Documentation HEAD mismatch: {head}")
    if status:
        raise RuntimeError("Official documentation checkout is dirty")
    if symbolic.returncode == 0:
        raise RuntimeError(f"Official documentation checkout is not detached: {symbolic.stdout.strip()}")
    if symbolic.returncode != 1:
        raise RuntimeError(f"Unable to verify detached HEAD: {symbolic.stderr.strip()}")
    return {
        "source_snapshot_id": manifest["commit"],
        "manifest_sha256": manifest_hash,
        "docs_head": head,
        "docs_clean": True,
        "docs_detached": True,
    }
