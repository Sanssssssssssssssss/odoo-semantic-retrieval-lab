from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .baselines import BGE_KEY, RERANKER_KEY
from .chunking import verify_model_snapshot
from .gates import require_approval
from .jsonio import canonical_json, sha256_file, stable_id, write_json
from .paths import LabPaths
from .perf_worker import SYSTEMS
from .verify import verify_source


def _bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _percentiles(values: list[int]) -> dict[str, float]:
    milliseconds = np.asarray(values, dtype=np.float64) / 1_000_000
    return {
        f"p{percentile}_ms": float(np.percentile(milliseconds, percentile))
        for percentile in (50, 90, 95, 99)
    }


def _framework_overhead(measured_loop_ns: int, request_durations_ns: list[int]) -> tuple[int, float]:
    overhead_ns = max(0, measured_loop_ns - sum(request_durations_ns))
    return overhead_ns, overhead_ns / measured_loop_ns


def _runtime_initialization_ns(receipt: dict[str, Any]) -> int:
    if "runtime_initialization_duration_ns" in receipt:
        return int(receipt["runtime_initialization_duration_ns"])
    samples = receipt["resource_samples"]
    return int(samples[1]["monotonic_ns"] - samples[0]["monotonic_ns"])


def _power_scheme() -> dict[str, str] | None:
    result = subprocess.run(
        ["powercfg", "/getactivescheme"], capture_output=True, check=False
    )
    if result.returncode:
        return None
    match = re.search(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", result.stdout)
    return {"guid": match.group(0).decode("ascii").lower()} if match else None


def _implementation_hashes(paths: LabPaths) -> dict[str, str]:
    names = (
        "src/osrlab/performance.py",
        "src/osrlab/perf_worker.py",
        "src/osrlab/baselines.py",
        "src/osrlab/chunking.py",
        "src/osrlab/extraction.py",
        "src/osrlab/smoke.py",
        "src/osrlab/jsonio.py",
        "configs/models.json",
        "uv.lock",
    )
    return {name: sha256_file(paths.root / name) for name in names}


def _nvml_sampler() -> tuple[dict[str, Any], Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
        info = {
            "available": True,
            "devices": [
                {
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                }
                for handle in handles
            ],
        }

        def sample() -> int:
            return sum(pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles)

        return info, sample
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}, lambda: 0


def _launch_worker(
    paths: LabPaths,
    system: str,
    mode: str,
    output: Path,
    run_id: str,
    index_root: Path | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if index_root is not None:
        environment["OSRLAB_PERF_INDEX_ROOT"] = str(index_root)
    command = [
        sys.executable,
        "-m",
        "osrlab.perf_worker",
        "--system",
        system,
        "--mode",
        mode,
        "--output",
        str(output),
        "--run-id",
        run_id,
    ]
    process = subprocess.Popen(
        command,
        cwd=paths.root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ps_process = psutil.Process(process.pid)
    nvml_info, sample_vram = _nvml_sampler()
    peak_tree_rss = 0
    peak_nvml_used = 0
    while process.poll() is None:
        try:
            children = ps_process.children(recursive=True)
            peak_tree_rss = max(
                peak_tree_rss,
                ps_process.memory_info().rss
                + sum(child.memory_info().rss for child in children if child.is_running()),
            )
            peak_nvml_used = max(peak_nvml_used, sample_vram())
        except (psutil.Error, OSError):
            pass
        time.sleep(1.0)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"Performance worker failed ({system}/{mode}):\n{stdout[-4000:]}\n{stderr[-8000:]}"
        )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["external_peak_process_tree_rss_bytes"] = peak_tree_rss
    receipt["external_peak_nvml_used_bytes"] = peak_nvml_used
    receipt["nvml"] = nvml_info
    receipt["worker_stdout"] = stdout.strip()
    receipt["worker_stderr"] = stderr.strip()
    write_json(output, receipt, paths)
    return receipt


def run_performance(paths: LabPaths | None = None) -> dict[str, Any]:
    paths = paths or LabPaths.discover()
    approval = require_approval("p3b-matrix", paths)
    verify_source(paths)
    matrix_path = paths.root / "artifacts" / "matrices" / "seed50-provisional" / "manifest.json"
    if sha256_file(matrix_path) != approval["matrix_manifest_sha256"]:
        raise RuntimeError("P4 input matrix differs from approved P3b matrix")
    bge = verify_model_snapshot(paths, BGE_KEY)
    reranker = verify_model_snapshot(paths, RERANKER_KEY)
    implementation_hashes = _implementation_hashes(paths)
    specification = {
        "chunk_config": "C2-structure-bounded",
        "systems": list(SYSTEMS),
        "cold_processes_per_system": 5,
        "warmup_requests": 20,
        "minimum_measured_requests": 1024,
        "minimum_measured_seconds": 60,
        "concurrency": 1,
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "resource_sample_interval_ms": 1000,
        "acquisition_excluded": True,
    }
    run_id = stable_id(
        approval["matrix_id"],
        canonical_json(specification),
        canonical_json(implementation_hashes),
        length=40,
    )
    root = paths.require_write_path(paths.root / "artifacts" / "performance" / "p4" / run_id)
    workers = root / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    power_scheme_before = _power_scheme()
    acquisition = {
        "included_in_cold_or_warm": False,
        "dependency_lock_sha256": sha256_file(paths.root / "uv.lock"),
        "models": {BGE_KEY: bge, RERANKER_KEY: reranker},
        "implementation_hashes": implementation_hashes,
        "power_scheme_before": power_scheme_before,
    }
    write_json(root / "acquisition.json", acquisition, paths)
    build = _launch_worker(paths, "BUILD", "build", workers / "build.json", run_id)
    build_checks = (
        "extraction_canonical_hashes_match",
        "c2_chunks_match_canonical",
        "dense_index_matches_canonical",
    )
    if not all(build.get(check) is True for check in build_checks):
        raise RuntimeError(
            "P4 fresh build did not reproduce the frozen corpus/index: "
            + ", ".join(f"{check}={build.get(check)!r}" for check in build_checks)
        )
    fresh_index = workers / "fresh-index"
    cold: dict[str, list[dict[str, Any]]] = {}
    warm: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        cold[system] = [
            _launch_worker(
                paths,
                system,
                "cold",
                workers / f"cold-{system}-{index}.json",
                run_id,
                fresh_index,
            )
            for index in range(1, 6)
        ]
        warm[system] = _launch_worker(
            paths, system, "warm", workers / f"warm-{system}.json", run_id, fresh_index
        )

    summary: dict[str, Any] = {}
    for system in SYSTEMS:
        cold_load = [_runtime_initialization_ns(row) for row in cold[system]]
        cold_first = [row["requests"][0]["total_duration_ns"] for row in cold[system]]
        warm_durations = [row["total_duration_ns"] for row in warm[system]["requests"]]
        loop_seconds = warm[system]["measured_loop_duration_ns"] / 1_000_000_000
        framework_overhead_ns, framework_overhead_fraction = _framework_overhead(
            warm[system]["measured_loop_duration_ns"], warm_durations
        )
        if framework_overhead_fraction > 0.05:
            raise RuntimeError(
                f"Warm harness overhead exceeded 5% for {system}: "
                f"{framework_overhead_fraction:.2%}"
            )
        top10_by_query: dict[str, set[tuple[str, ...]]] = {}
        for request in warm[system]["requests"]:
            top10_by_query.setdefault(request["query_id"], set()).add(
                tuple(request["top10_chunk_ids"])
            )
        top10_repeat_deterministic = all(len(values) == 1 for values in top10_by_query.values())
        if not top10_repeat_deterministic:
            raise RuntimeError(f"Warm top-10 IDs were not deterministic for {system}")
        summary[system] = {
            "cold_processes": 5,
            "cold_load": _percentiles(cold_load),
            "cold_first_query": _percentiles(cold_first),
            "warm_requests": len(warm_durations),
            "warm_measured_seconds": loop_seconds,
            "warm_latency": _percentiles(warm_durations),
            "warm_qps": len(warm_durations) / loop_seconds,
            "warm_framework_overhead_ns": framework_overhead_ns,
            "warm_framework_overhead_fraction": framework_overhead_fraction,
            "top10_repeat_deterministic": top10_repeat_deterministic,
            "peak_process_tree_rss_bytes": max(
                [row["external_peak_process_tree_rss_bytes"] for row in cold[system]]
                + [warm[system]["external_peak_process_tree_rss_bytes"]]
            ),
            "peak_torch_allocated_vram_bytes": max(
                receipt["peak_torch_allocated_vram_bytes"]
                for receipt in [*cold[system], warm[system]]
            ),
            "peak_torch_reserved_vram_bytes": max(
                receipt["peak_torch_reserved_vram_bytes"]
                for receipt in [*cold[system], warm[system]]
            ),
            "peak_nvml_used_bytes": max(
                [row["external_peak_nvml_used_bytes"] for row in cold[system]]
                + [warm[system]["external_peak_nvml_used_bytes"]]
            ),
        }
    power_scheme_after = _power_scheme()
    hardware = {
        "platform": platform.platform(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "power_scheme_before": power_scheme_before,
        "power_scheme_after": power_scheme_after,
        "power_scheme_modified": power_scheme_before != power_scheme_after,
        "torch_device_observed": warm["E3-rerank"]["device"],
        "torch_version": warm["E3-rerank"]["torch_version"],
        "torch_cuda_version": warm["E3-rerank"]["torch_cuda_version"],
        "nvml": {
            **warm["E3-rerank"]["nvml"],
            "measurement_scope": "system_global_all_devices",
        },
    }
    if hardware["power_scheme_modified"]:
        raise RuntimeError("Windows power scheme changed during P4")
    sizes = {
        "indexes_bytes": _bytes(fresh_index),
        "model_cache_bytes": _bytes(paths.root / ".cache" / "huggingface"),
        "lab_cache_bytes": _bytes(paths.root / ".cache"),
    }
    write_json(root / "summary.json", summary, paths)
    report_lines = [
        "# P4 C2 performance baseline",
        "",
        f"Run: `{run_id}`. Acquisition is excluded. Device: `{hardware['torch_device_observed']}`.",
        "",
        "| System | warm requests | seconds | p50 ms | p90 ms | p95 ms | p99 ms | QPS | peak RSS MiB | harness overhead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = summary[system]
        report_lines.append(
            f"| {system} | {row['warm_requests']} | {row['warm_measured_seconds']:.3f} | "
            f"{row['warm_latency']['p50_ms']:.3f} | {row['warm_latency']['p90_ms']:.3f} | "
            f"{row['warm_latency']['p95_ms']:.3f} | "
            f"{row['warm_latency']['p99_ms']:.3f} | {row['warm_qps']:.3f} | "
            f"{row['peak_process_tree_rss_bytes'] / 1048576:.1f} | "
            f"{row['warm_framework_overhead_fraction']:.2%} |"
        )
    report_lines.extend(
        [
            "",
            "Cold start uses five fresh processes per system. Warm measurements use 20 warm-ups, "
            "single-stream concurrency=1, and continue until both 60 seconds and 1,024 requests are met.",
            "Warm QPS is completed single-stream requests divided by measured loop time; it is not a server load-capacity claim.",
            "No system power setting was changed.",
            "",
            "## Cold start",
            "",
            "| System | load p50 ms | load p90 ms | load p95 ms | load p99 ms | first query p50 ms | first query p95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {system} | {summary[system]['cold_load']['p50_ms']:.3f} | "
                f"{summary[system]['cold_load']['p90_ms']:.3f} | "
                f"{summary[system]['cold_load']['p95_ms']:.3f} | "
                f"{summary[system]['cold_load']['p99_ms']:.3f} | "
                f"{summary[system]['cold_first_query']['p50_ms']:.3f} | "
                f"{summary[system]['cold_first_query']['p95_ms']:.3f} |"
                for system in SYSTEMS
            ],
            "",
            "## Resource scope",
            "",
            f"- Torch peak allocated/reserved VRAM: "
            f"`{max(row['peak_torch_allocated_vram_bytes'] for row in summary.values())}` / "
            f"`{max(row['peak_torch_reserved_vram_bytes'] for row in summary.values())}` bytes.",
            f"- NVML available: `{hardware['nvml']['available']}`; scope: "
            f"`{hardware['nvml']['measurement_scope']}`; peak used: "
            f"`{max(row['peak_nvml_used_bytes'] for row in summary.values())}` bytes.",
            f"- Fresh index/model cache/lab cache: `{sizes['indexes_bytes']}` / "
            f"`{sizes['model_cache_bytes']}` / `{sizes['lab_cache_bytes']}` bytes.",
        ]
    )
    report_path = paths.require_write_path(root / "report.md")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    output_hashes = {
        "acquisition.json": sha256_file(root / "acquisition.json"),
        "summary.json": sha256_file(root / "summary.json"),
        "report.md": sha256_file(report_path),
        **{
            f"workers/{path.name}": sha256_file(path)
            for path in sorted(workers.glob("*.json"))
        },
    }
    manifest = {
        "id": run_id,
        "schema_version": 1,
        "status": "complete",
        "input_matrix_id": approval["matrix_id"],
        "implementation_hashes": implementation_hashes,
        "specification": specification,
        "hardware": hardware,
        "sizes": sizes,
        "build": build,
        "output_hashes": output_hashes,
    }
    write_json(root / "manifest.json", manifest, paths)
    return manifest
