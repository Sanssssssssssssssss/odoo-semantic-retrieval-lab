from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .baselines import run_provisional_matrix
from .chunking import chunk_twice
from .extraction import extract_twice
from .gates import require_approval
from .paths import LabPaths
from .performance import run_performance
from .pooling import build_pool
from .receipts import create_environment_receipt
from .smoke import run_smoke
from .verify import verify_source


COMMANDS = ("verify", "extract", "chunk", "smoke", "baseline", "pool", "perf", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osrlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify isolation and pinned source")
    for command in COMMANDS[1:]:
        child = subparsers.add_parser(command)
        if command == "all":
            child.add_argument("--profile", default="seed", choices=("seed",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    args = build_parser().parse_args(argv)
    paths = LabPaths.discover()
    result = verify_source(paths)
    if args.command == "verify":
        result["environment_receipt"] = str(create_environment_receipt(paths))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "extract":
        require_approval("p0", paths)
        print(json.dumps(extract_twice(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "chunk":
        require_approval("p1a", paths)
        print(json.dumps(chunk_twice(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "smoke":
        require_approval("p2a", paths)
        print(json.dumps(run_smoke(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "baseline":
        require_approval("p3a", paths)
        print(json.dumps(run_provisional_matrix(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "pool":
        print(json.dumps(build_pool(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "perf":
        print(json.dumps(run_performance(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "all":
        require_approval("p3b-pool", paths)
        human_receipt = (
            paths.root
            / "benchmarks"
            / "seed50"
            / "provisional"
            / "human_review"
            / "receipt.json"
        )
        if not human_receipt.is_file():
            raise RuntimeError(
                "Seed50 remains provisional: missing human review receipt; all -Profile seed stops before Seed freeze"
            )
        raise RuntimeError("Seed50 freeze/final-matrix path is not enabled until human review is approved")
    raise SystemExit(f"Command '{args.command}' is gated until its implementation phase is approved")
