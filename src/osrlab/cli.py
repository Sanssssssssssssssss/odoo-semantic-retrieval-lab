from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .baselines import run_provisional_matrix
from .chunking import chunk_twice
from .diagnostics import run_diagnostics
from .extraction import extract_twice
from .gates import require_approval
from .paths import LabPaths
from .p5 import run_p5
from .performance import run_performance
from .pooling import build_pool
from .receipts import create_environment_receipt
from .smoke import run_smoke
from .tuning import run_e2_tuning, run_e3_tuning
from .verify import verify_source


COMMANDS = (
    "verify",
    "extract",
    "chunk",
    "smoke",
    "baseline",
    "pool",
    "perf",
    "p5",
    "tune-e2",
    "tune-e3",
    "all",
)
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
    "seed50-diagnostics",
    "p4",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osrlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify isolation and pinned source")
    for command in COMMANDS[1:]:
        child = subparsers.add_parser(command)
        if command == "p5":
            child.add_argument("--minimum-seconds", type=int, default=60)
            child.add_argument("--minimum-requests", type=int, default=1024)
            child.add_argument("--cold-processes", type=int, default=5)
        if command == "all":
            child.add_argument("--profile", default="seed", choices=("seed",))
    return parser


def run_baseline(paths: LabPaths) -> dict:
    return {
        "matrix": run_provisional_matrix(paths),
        "diagnostics": run_diagnostics(paths),
    }


def _require_seed_approval(phase: str, paths: LabPaths) -> dict:
    if phase != "seed50-diagnostics":
        return require_approval(phase, paths)
    root = paths.root / "artifacts" / "diagnostics" / "seed50-provisional"
    return require_approval(
        phase,
        paths,
        {
            "manifest_sha256": root / "manifest.json",
            "report_sha256": root / "report.json",
            "implementation_sha256": paths.root / "src" / "osrlab" / "diagnostics.py",
        },
    )


def run_seed_pipeline(paths: LabPaths, verification: dict) -> dict:
    approvals = {phase: _require_seed_approval(phase, paths) for phase in SEED_APPROVALS}
    human_receipt = (
        paths.root
        / "benchmarks"
        / "seed50"
        / "pooling"
        / "provisional"
        / "human_review"
        / "receipt.json"
    )
    human_review_complete = False
    if human_receipt.is_file():
        human = json.loads(human_receipt.read_text(encoding="utf-8"))
        human_review_complete = (
            human.get("decision") == "APPROVE" and human.get("human_review_complete") is True
        )
    stages = {}
    stages["verify"] = {
        **verification,
        "environment_receipt": str(create_environment_receipt(paths)),
    }
    stages["extract"] = extract_twice(paths)
    stages["chunk"] = chunk_twice(paths)
    stages["smoke"] = run_smoke(paths)
    stages["baseline"] = run_baseline(paths)
    _require_seed_approval("seed50-diagnostics", paths)
    stages["pool"] = build_pool(paths)
    stages["perf"] = run_performance(paths)
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
        print(json.dumps(run_baseline(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "pool":
        print(json.dumps(build_pool(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "perf":
        print(json.dumps(run_performance(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "p5":
        print(
            json.dumps(
                run_p5(
                    paths,
                    minimum_seconds=args.minimum_seconds,
                    minimum_requests=args.minimum_requests,
                    cold_processes=args.cold_processes,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "tune-e2":
        print(json.dumps(run_e2_tuning(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "tune-e3":
        print(json.dumps(run_e3_tuning(paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "all":
        print(json.dumps(run_seed_pipeline(paths, result), indent=2, ensure_ascii=False))
        return 0
    raise SystemExit(f"Command '{args.command}' is gated until its implementation phase is approved")
