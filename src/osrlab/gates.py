from __future__ import annotations

import json
from pathlib import Path

from .paths import LabPaths


def require_approval(phase: str, paths: LabPaths | None = None) -> dict:
    paths = paths or LabPaths.discover()
    receipt = paths.root / "reviews" / phase / "approval.json"
    if not receipt.is_file():
        raise RuntimeError(f"Missing approval receipt: {receipt}")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record.get("phase") != phase or record.get("decision") != "APPROVE":
        raise RuntimeError(f"Approval receipt is not valid for {phase}: {receipt}")
    return record
