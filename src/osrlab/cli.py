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
SEED_APPROVALS = (
    "p0",
    "p1a",
    "p1b",
    "p2a",
    "p3a",
    "p3b-matrix",
    "p3b-pool",
    "p3b-depth20-agent-annotations",
    "p3b-depth30-agent-annotations",
    "p3b-depth40-agent-annotations",
    "p3b-depth50-agent-annotations",
    "p4",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osrlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify isolation and pinned source")
    for command in COMMANDS[1:]:
        child = subparsers.add_parser(command)
        if command == "all":
            child.add_argument("--profile", default="seed", choices=("seed",))
    return parser


def run_seed_pipeline(paths: LabPaths, verification: dict) -> dict:
    approvals = {phase: require_approval(phase, paths) for phase in SEED_APPROVALS}
    human_receipt = (
        paths.root / "benchmarks" / "seed50" / "provisional" / "human_review" / "receipt.json"
    )
    human_review_complete = False
    if human_receipt.is_file():
        human = json.loads(human_receipt.read_text(encoding="utf-8"))
        human_review_complete = (
            human.get("decision") == "APPROVE" and human.get("human_review_complete") is True
        )
    stages = {
        "verify": {**verification, "environment_receipt": str(create_environment_receipt(paths))},
        "extract": extract_twice(paths),
        "chunk": chunk_twice(paths),
        "smoke": run_smoke(paths),
        "baseline": run_provisional_matrix(paths),
        "pool": build_pool(paths),
        "perf": run_performance(paths),
    }
    return {
        "profile": "seed",
        "status": (
            "human_review_approved_final_freeze_pending"
            if human_review_complete
            else "agent_provisional_complete_human_review_pending"
        ),
        "approval_phases": list(approvals),
        "human_review_complete": human_review_complete,
        "seed_frozen": False,
        "sota_claims_allowed": False,
        "stages": stages,
    }


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
        print(json.dumps(run_seed_pipeline(paths, result), indent=2, ensure_ascii=False))
        return 0
    raise SystemExit(f"Command '{args.command}' is gated until its implementation phase is approved")
