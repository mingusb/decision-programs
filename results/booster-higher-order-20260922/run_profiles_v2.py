#!/usr/bin/env python3
"""Root-only serial diagnostics, after unprofiled rankings; default prints a plan.

No profiler duration is a speed ranking. All 983 Delicious labels are retained.
This file is deliberately independent of the frozen timing campaign runner.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
BINARY = ROOT / "bin/ghb_real_bench"
PROVENANCE = ROOT / "production-provenance"
FIXTURES = WORKSPACE / "results/booster-level-batch-20260922/data/fixtures/delicious"
NSYS = Path("/usr/local/bin/nsys")
NCU = Path("/opt/nvidia/nsight-compute/2026.3.0/ncu")
STATS = "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum,nvtx_sum,nvtx_gpu_proj_sum"
CONFIG = {"rounds": 1, "depth": 3, "bins": 32, "learning_rate": .1, "l2": 1,
          "max_leaf_value": 1, "histogram": "global", "tree_build": "output-batch",
          "output_tile": 16, "tree_export_batch": 16}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, new=True):
    with Path(path).open("x" if new else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def kernel_pattern(order, stage):
    # CUDA demanglers may include explicit non-type template parameter casts.
    uint = r"(?:\(unsigned int\)\s*)?"
    yes = r"(?:\(bool\)\s*)?(?:1|true)"
    number = lambda value: uint + str(value) + "u?"
    if order == 2:
        suffix = {"derivative": r"independent_gradients\(",
                  "root": rf"accumulate<{number(16)},\s*{yes},\s*{yes}>\(",
                  "split": rf"split_candidates<{yes}>\("}[stage]
    else:
        suffix = {"derivative": rf"derivatives<{number(order)}>\(",
                  "root": rf"accumulate<{number(order)},\s*{number(16)},\s*{yes},\s*{yes}>\(",
                  "split": rf"split_candidates<{number(order)},\s*{number(256)}>\("}[stage]
    return ".*::" + suffix + ".*"


def benchmark(order, execution, output):
    cmd = [str(BINARY), "--train", str(FIXTURES / "train.ghb"), "--evaluation",
           str(FIXTURES / "validation.ghb"), "--output-dir", str(output),
           "--optimization-order", str(order), "--tree-execution", execution,
           "--instrumentation", "nvtx" if execution == "graph" else "off"]
    for name, value in CONFIG.items():
        cmd.extend(("--" + name.replace("_", "-"), str(value)))
    return cmd


def jobs(output):
    result = []
    for order in (2, 3, 4):
        name = f"nsys-o{order}"
        destination = output / name
        result.append({"name": name, "order": order, "kind": "systems",
                       "command": [str(NSYS), "profile", "--trace=cuda,nvtx,osrt",
                                   "--cuda-graph-trace=node", "--sample=none", "--output",
                                   str(destination / "profile"),
                                   *benchmark(order, "graph", destination / "benchmark")]})
    for order in (2, 3, 4):
        for stage in ("derivative", "root", "split"):
            name = f"ncu-o{order}-{stage}"
            destination = output / name
            pattern = kernel_pattern(order, stage)
            result.append({"name": name, "order": order, "kind": "compute", "stage": stage,
                           "expected_kernel_regex": pattern,
                           "command": [str(NCU), "--set", "full", "--kernel-name-base", "demangled",
                                       "--kernel-name", "regex:" + pattern, "--launch-count", "1",
                                       "--clock-control", "none", "--export", str(destination / "profile"),
                                       *benchmark(order, "stream", destination / "benchmark")]})
    return result


def preflight():
    frozen_sources = read(PROVENANCE / "source-sha256.json")
    current_sources = {name: digest(WORKSPACE / name) for name in frozen_sources}
    if current_sources != frozen_sources:
        raise RuntimeError("source differs from frozen production provenance")
    binary_sha = digest(BINARY)
    if binary_sha != read(PROVENANCE / "binary-sha256.json")[BINARY.name]:
        raise RuntimeError("binary differs from frozen production provenance")
    plan = read(ROOT / "real/plan/protocol.json")
    fixture_hashes = {}
    for split, rows in (("train", 10336), ("validation", 2584)):
        path = FIXTURES / f"{split}.ghb"
        with path.open("rb") as stream:
            header = struct.unpack("<8s6I", stream.read(32))
        if header[:6] != (b"GHBDS001", 1, rows, 500, 983, 1):
            raise RuntimeError(f"unexpected Delicious {split} dimensions/objective")
        if path.stat().st_size != 32 + rows * (500 + 983) * 4:
            raise RuntimeError("unexpected fixture extent")
        fixture_hashes[str(path)] = digest(path)
        if fixture_hashes[str(path)] != plan["fixture_sha256"][f"delicious/{split}"]:
            raise RuntimeError("fixture differs from preregistered campaign")
    return {"binary_sha256": binary_sha, "source_sha256": frozen_sources,
            "runner_sha256": digest(__file__), "fixture_sha256": fixture_hashes,
            "provenance_sha256": {str(p): digest(p) for p in
                                  (PROVENANCE / "source-sha256.json", PROVENANCE / "binary-sha256.json",
                                   ROOT / "real/plan/protocol.json")},
            "profiler_sha256": {str(p): digest(p) for p in (NSYS, NCU)}}


def checked(destination, name, command, identity):
    """Capture each actual command before launch and preserve its actual exit code."""
    receipt = {"command": command, "cwd": str(WORKSPACE), "started_utc": stamp(),
               "binary_sha256": digest(BINARY), "runner_sha256": digest(__file__),
               "executable_sha256": digest(command[0]),
               "environment_overrides": {"OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
    if receipt["binary_sha256"] != identity["binary_sha256"] or receipt["runner_sha256"] != identity["runner_sha256"]:
        raise RuntimeError("frozen executable or runner changed")
    receipt_path = destination / f"{name}-command.json"
    write(receipt_path, receipt)
    env = os.environ.copy()
    env.update(receipt["environment_overrides"])
    started = time.perf_counter()
    try:
        with (destination / f"{name}.stdout").open("x") as out, (destination / f"{name}.stderr").open("x") as err:
            process = subprocess.run(command, cwd=WORKSPACE, env=env, stdout=out, stderr=err, check=False)
        receipt["returncode"] = process.returncode
    except Exception as error:
        receipt.update(returncode=None, execution_error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_utc=stamp(), wall_seconds=time.perf_counter() - started,
                       binary_unchanged=digest(BINARY) == identity["binary_sha256"],
                       executable_unchanged=digest(command[0]) == receipt["executable_sha256"],
                       runner_unchanged=digest(__file__) == identity["runner_sha256"])
        for stream in ("stdout", "stderr"):
            path = destination / f"{name}.{stream}"
            if path.exists():
                receipt[f"{stream}_sha256"] = digest(path)
        write(receipt_path, receipt, new=False)
    if receipt["returncode"] or not all(receipt[k] for k in ("binary_unchanged", "executable_unchanged", "runner_unchanged")):
        raise RuntimeError(f"diagnostic command failed: {destination.name}/{name}")
    return receipt


def verify_benchmark(destination, job):
    value = read(destination / "benchmark/metrics.json")
    expected = {k: v for k, v in CONFIG.items() if k != "tree_export_batch"}
    expected.update(optimization_order=job["order"], tree_execution="graph" if job["kind"] == "systems" else "stream")
    if any(value["parameters"].get(k) != v for k, v in expected.items()):
        raise RuntimeError("profiled configuration differs from requested configuration")
    if (value["backend"], value["prediction_backend"], value["train_rows"], value["evaluation_rows"],
        value["features"], value["outputs"], value["trees"], value["tree_batch_size"]) != ("cuda", "cuda", 10336, 2584, 500, 983, 983, 16):
        raise RuntimeError("profiled workload differs from full Delicious contract")
    return {"configuration_verified": True, "metrics_sha256": digest(destination / "benchmark/metrics.json")}


def verify_kernel(destination, pattern):
    with (destination / "raw.stdout").open(newline="") as stream:
        rows = csv.DictReader(stream)
        observed = [row for row in rows if row.get("ID", "").isdigit()]
    if len(observed) != 1 or not re.fullmatch(pattern, observed[0].get("Kernel Name", "")):
        raise RuntimeError("expected exactly one capture of the requested kernel")
    row = observed[0]
    return {"captured_launches": 1, "kernel": row["Kernel Name"],
            "launch": {key: row.get(key) for key in ("ID", "Process ID", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC")}}


def run(output, requested):
    identity = preflight()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "protocol.json", {"created_utc": stamp(), "identity": identity, "jobs": requested,
          "retry_reason": "Original profiles/ attempt retained: NCU root filter omitted explicit template casts, matched no kernels and produced no report. This retry changes capture filters only; production binary and settings are identical.",
          "scope": "All 10336 training rows, 500 features, all 983 outputs; 2584 validation rows; raw-float fixtures and production GPU binning.",
          "comparison": "Equal order-2/3/4 one-round clipped settings. Systems uses graph+NVTX; Compute uses stream+instrumentation off. Do not compare profiler durations as speed rankings.",
          "capture": "First matching derivative, cached-count root histogram, and root split candidate launch only; full NCU metrics, clocks uncontrolled. Real 500-feature workload uses 256-thread split path.",
          "numerics": "Different optimization orders can change topology; same configuration does not imply identical learned models."})
    (output / ".incomplete").touch(exist_ok=False)
    completed = []
    passed = False
    try:
        for job in requested:
            destination = output / job["name"]
            destination.mkdir()
            result = {"name": job["name"], "order": job["order"], "kind": job["kind"], "passed": False}
            print(f"Starting {job['name']}", flush=True)
            try:
                checked(destination, "profile", job["command"], identity)
                result.update(verify_benchmark(destination, job))
                if job["kind"] == "systems":
                    report = destination / "profile.nsys-rep"
                    if not report.is_file() or not report.stat().st_size:
                        raise RuntimeError("missing Nsight Systems report")
                    checked(destination, "stats", [str(NSYS), "stats", "--report", STATS, "--format", "csv",
                            "--output", str(destination / "stats"), str(report)], identity)
                    for name in STATS.split(","):
                        path = destination / f"stats_{name}.csv"
                        if not path.is_file() or not path.stat().st_size:
                            raise RuntimeError(f"missing Systems summary {name}")
                else:
                    report = destination / "profile.ncu-repz"
                    if not report.is_file() or not report.stat().st_size:
                        raise RuntimeError("missing Nsight Compute report")
                    for page in ("details", "raw"):
                        checked(destination, page, [str(NCU), "--import", str(report), "--page", page, "--csv"], identity)
                    result.update(verify_kernel(destination, job["expected_kernel_regex"]))
                result["passed"] = True
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                result["artifacts_sha256"] = {str(path.relative_to(destination)): digest(path)
                                               for path in sorted(destination.rglob("*")) if path.is_file()}
                write(destination / "audit.json", result)
                completed.append(result)
        if preflight() != identity:
            raise RuntimeError("profile inputs changed during execution")
        passed = len(completed) == len(requested) and all(job["passed"] for job in completed)
    finally:
        write(output / "summary.json", {"finished_utc": stamp(), "expected": len(requested),
              "completed": len(completed), "passed": sum(job["passed"] for job in completed),
              "complete": passed, "jobs": completed, "times_are_diagnostic_only": True})
    if passed:
        (output / ".incomplete").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles-v2")
    parser.add_argument("--run", action="store_true", help="Execute serial GPU profiling; root only, after unprofiled measurements")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    requested = jobs(output)
    if args.run:
        run(output, requested)
    else:
        print(json.dumps({"gpu_executed": False, "jobs": requested}, indent=2))
