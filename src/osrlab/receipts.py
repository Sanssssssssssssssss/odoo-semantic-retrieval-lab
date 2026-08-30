from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import psutil

from .jsonio import canonical_json, sha256_file, stable_id, write_json
from .paths import LabPaths
from .verify import verify_source


def _git_state(path: Path) -> dict[str, Any] | None:
    if not (path / ".git").exists():
        return None
    command = ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path)]
    head = subprocess.run([*command, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run([*command, "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
    return {"path": str(path), "head": head, "status": status}


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, errors="replace")
    except OSError:
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def create_environment_receipt(paths: LabPaths | None = None, phase: str = "verify") -> Path:
    paths = paths or LabPaths.discover()
    siblings = [paths.root.parent / "erp-openai", paths.root.parent / "erp-agent-odoo"]
    source = verify_source(paths)
    packages = sorted(
        ({"name": dist.metadata["Name"], "version": dist.version} for dist in distributions()),
        key=lambda item: (item["name"] or "").lower(),
    )
    receipt = {
        "schema_version": 1,
        "phase": phase,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hardware": {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": _command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"),
            "power_scheme": _command("powercfg", "/getactivescheme"),
        },
        "dependencies": {
            "lock_sha256": sha256_file(paths.root / "uv.lock"),
            "installed": packages,
        },
        "environment_roots": {
            "uv_cache": str(paths.root / ".cache" / "uv"),
            "hf_home": str(paths.root / ".cache" / "huggingface"),
            "torch_home": str(paths.root / ".cache" / "torch"),
            "temp": str(paths.root / ".cache" / "tmp"),
        },
        "source": source,
        "lab": _git_state(paths.root),
        "protected_projects": [state for path in siblings if (state := _git_state(path)) is not None],
    }
    receipt_id = stable_id(canonical_json(receipt), length=40)
    output = paths.require_write_path(Path("artifacts") / "receipts" / f"{phase}-{receipt_id}.json")
    write_json(output, receipt, paths)
    return output
