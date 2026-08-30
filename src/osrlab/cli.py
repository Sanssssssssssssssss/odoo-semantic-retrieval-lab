from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .extraction import extract_twice
from .gates import require_approval
from .paths import LabPaths
from .receipts import create_environment_receipt
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
    raise SystemExit(f"Command '{args.command}' is gated until its implementation phase is approved")
