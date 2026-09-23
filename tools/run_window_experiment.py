#!/usr/bin/env python3
"""Recorded fixed-policy window and located-skew experiments; --audit is CPU-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "results/a5000-profiled/run_round.py"
SEEDS = (2026092271, 2026092272)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"refusing changed evidence: {path}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def workload(n=1 << 24, bins=1 << 20, distribution="uniform", order="shuffled",
             cache="warm", launch="graph"):
    return dict(n=n, bins=bins, input="u32", counter="u64", distribution=distribution,
                order=order, cache=cache, launch=launch, warmup_ms=200)


def cases(stage):
    result = []
    if stage == "window":
        shapes = [(workload(), window) for window in (1 << 20, 1 << 19)]
        shapes += [(workload(bins=bins), 1 << 19)
                   for bins in (524287, 524288, 524289, 786432)]
        shapes += [(shape, 1 << 19) for shape in
                   (workload(n=(1 << 24) + 17), workload(n=1 << 28),
                    workload(cache="cold"), workload(launch="stream"))]
        for shape, window in shapes:
            clear = "runtime" if shape["launch"] == "stream" else "kernel"
            result.append(dict(workload=shape, window=window, variants=[
                f"global:0:48:u32:{clear}", f"global_window:0:48:u32:{clear}"], batch=4))
    elif stage == "skew":
        for bins in (24577, 32768, 1048576):
            locations = sorted({24575, 24576, bins - 1})
            for hot in locations:
                for order in ("shuffled", "sorted"):
                    shape = workload(n=(1 << 20) + 17, bins=bins,
                                     distribution=f"hot99@{hot}", order=order)
                    uniform_choice = ("global:0:48:u32:kernel" if bins == 1048576
                                      else "shared_overflow:15:24:u32:kernel")
                    result.append(dict(workload=shape, window=0, variants=[uniform_choice,
                                       "warp:4:96:native:kernel", "warp:4:96:u32:kernel"], batch=2))
    else:
        raise ValueError("unknown stage")
    for index, case in enumerate(result):
        case["id"] = index
        case["samples"] = 11
    return result


def manifest(exe, stage):
    sources = [ROOT / "CMakeLists.txt", Path(__file__), RECORDER]
    for directory in ("include", "src", "bench", "support", "third_party"):
        sources += sorted(path for path in (ROOT / directory).glob("**/*")
                          if path.is_file() and path.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".h")
                          and "preservation" not in path.relative_to(ROOT).parts)
    jobs = [dict(case=case, seed=seed) for case in cases(stage) for seed in SEEDS]
    random.Random(2026092281).shuffle(jobs)
    for index, job in enumerate(jobs):
        job["stem"] = f"{index:02d}-case{job['case']['id']}-seed{job['seed']}"
    return dict(schema=1, stage=stage, binary=record(exe), sources=[record(p) for p in sources],
                seeds=SEEDS, jobs=jobs, production_default_promotion=False,
                protocol="Fixed policies; 11 randomized rounds; no tuning or selection. Compare medians within each invocation. All raw samples retained.",
                limits=["Two fresh seeds/processes are descriptive replication, not a confidence interval or zero-loss proof.",
                        "Warm means no explicit cache eviction, not full input residency. GPU clocks remain unlocked.",
                        "Separate window sizes run in separate processes, each with a matched existing-kernel control.",
                        "Skew input is a 99% forced-value mixture plus uniform background, including the dominant bin.",
                        "No NVIDIA histogram is requested in either experiment."])


def command(binary, job):
    case = job["case"]
    result = [binary]
    for key, value in case["workload"].items():
        result += ["--" + key.replace("_", "-"), str(value)]
    result += ["--seed", str(job["seed"]), "--samples", str(case["samples"]),
               "--batch", str(case["batch"]), "--variants", ",".join(case["variants"])]
    if case["window"]:
        result += ["--window-bins", str(case["window"])]
    return result


def frozen(value):
    for item in [value["binary"], *value["sources"]]:
        if record(item["path"]) != item:
            raise ValueError("changed frozen input: " + item["path"])


def parse(path, job):
    case = job["case"]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("invalid CSV header")
        rows = list(reader)
    seen, values, environment = set(), [], None
    for row in rows:
        if None in row or any(value is None or value == "" for value in row.values()):
            raise ValueError("incomplete CSV row")
        variant = ":".join(row[key] for key in ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))
        if variant in seen or variant not in case["variants"]:
            raise ValueError("duplicate or undeclared variant")
        seen.add(variant)
        fixed = dict(case["workload"], seed=job["seed"], samples=case["samples"], batch=case["batch"], timing_protocol=3)
        if any(row.get(key) != str(value) for key, value in fixed.items()):
            raise ValueError("workload or protocol differs")
        if int(row["eviction_bytes"]) != (64 << 20 if row["cache"] == "cold" else 0):
            raise ValueError("A5000 cache eviction size differs")
        policy = int(row["tuning"])
        threads, items, replicas = {0: (128, 4, 1), 4: (128, 8, 4), 15: (512, 8, 1)}[policy]
        expected_policy = dict(threads=threads, items=items, replicas=replicas,
                               load_policy="vector4" if policy == 15 else "scalar",
                               shared_limit=98304 if policy == 15 else 49152)
        if any(row[key] != str(value) for key, value in expected_policy.items()):
            raise ValueError("policy resource metadata differs")
        window = case["window"] if row["algorithm"] == "global_window" else 0
        if case["window"] and int(row["window_bins"]) != window:
            raise ValueError("window identity differs")
        expected_scratch = (min(int(row["bins"]), window) * 4 if window else
                            int(row["bins"]) * 4 if row["algorithm"] in ("global", "warp") and row["local_counter"] == "u32" else 0)
        if int(row["scratch_bytes"]) != expected_scratch:
            raise ValueError("scratch size differs")
        raw = [float(value) for value in row["sample_us"].split(";")]
        if len(raw) != case["samples"] or any(not math.isfinite(v) or v <= 0 for v in raw):
            raise ValueError("invalid raw timing samples")
        ordered = sorted(raw)
        derived = dict(min_us=ordered[0], max_us=ordered[-1], median_us=ordered[len(raw) // 2],
                       p95_us=ordered[math.ceil(.95 * len(raw)) - 1])
        derived["input_gb_s"] = int(row["n"]) * 4 / (derived["median_us"] * 1000)
        if any(not math.isclose(float(row[key]), value, rel_tol=1e-6, abs_tol=1e-6)
               for key, value in derived.items()):
            raise ValueError("summary differs from raw timings")
        hardware = {key: row[key] for key in ("gpu", "sm", "driver_api", "runtime", "cub_version")}
        if environment is not None and environment != hardware:
            raise ValueError("mixed environment")
        environment = hardware
        values.append(dict(variant=variant, window_bins=window, raw_samples_us=raw, **derived))
    if seen != set(case["variants"]):
        raise ValueError("missing candidate")
    values.sort(key=lambda item: case["variants"].index(item["variant"]))
    return values, environment


def audit_job(base, value, environment, job):
    stem = base / "measurements" / job["stem"]
    receipt = read(stem.with_suffix(".receipt.json"))
    paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
    if receipt != dict(manifest=record(base / "manifest.json"), environment=record(base / "environment.json"),
                       artifacts=[record(path) for path in paths]):
        raise ValueError("receipt or artifact changed")
    metadata = read(paths[2])
    if (metadata["command"] != command(value["binary"]["path"], job)
            or metadata["exit_code"] != 0 or metadata["executables_unchanged"] is not True
            or metadata["binary_sha256"] != value["binary"]["sha256"]
            or metadata["executable_sha256"] != value["binary"]["sha256"]
            or metadata["binary"] != value["binary"]["path"]
            or metadata["gpus_before"] != environment["gpus"]
            or metadata["gpus_after"] != environment["gpus"]
            or not metadata["telemetry_before"] or not metadata["telemetry_after"]):
        raise ValueError("execution identity differs")
    values, hardware = parse(paths[0], job)
    if hardware["gpu"] not in {gpu["name"] for gpu in environment["gpus"]}:
        raise ValueError("GPU identity differs")
    return dict(case=job["case"], seed=job["seed"], candidates=values, hardware=hardware,
                ratios_to_first={item["variant"]: values[0]["median_us"] / item["median_us"] for item in values},
                artifacts=receipt["artifacts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("window", "skew"))
    parser.add_argument("--exe", type=Path, default=ROOT / "build/window-experiment/histogram_bench")
    parser.add_argument("--output-root", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    base = (options.output_root or ROOT / f"results/a5000-window-{options.stage}").resolve()
    value = json.loads(json.dumps(manifest(options.exe, options.stage)))
    if options.dry_run:
        print(json.dumps(value, indent=2)); return
    if options.audit:
        if read(base / "manifest.json") != value:
            raise ValueError("manifest differs")
        environment = read(base / "environment.json")
    else:
        write_once(base / "manifest.json", value)
        spec = importlib.util.spec_from_file_location("window_recorder", RECORDER)
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)
        recorder.EXE = options.exe.resolve()
        environment = recorder.ensure_environment(base)
    if environment["binary_sha256"] != value["binary"]["sha256"] or environment["binary"] != value["binary"]["path"]:
        raise ValueError("environment binary differs")
    results = []
    for index, job in enumerate(value["jobs"]):
        frozen(value)
        stem = base / "measurements" / job["stem"]
        if not options.audit and not stem.with_suffix(".receipt.json").exists():
            print(f"[{index + 1}/{len(value['jobs'])}] {job['stem']}", flush=True)
            recorder.run(stem, command(value["binary"]["path"], job))
            write_once(stem.with_suffix(".receipt.json"), dict(manifest=record(base / "manifest.json"),
                environment=record(base / "environment.json"), artifacts=[record(stem.with_suffix(suffix))
                for suffix in (".csv", ".log", ".command.json")]))
        result = audit_job(base, value, environment, job)
        if results and results[0]["hardware"] != result["hardware"]:
            raise ValueError("hardware/API changed across processes")
        results.append(result)
    expected = {job["stem"] + suffix for job in value["jobs"]
                for suffix in (".csv", ".log", ".command.json", ".receipt.json")}
    if {path.name for path in (base / "measurements").iterdir()} != expected:
        raise ValueError("measurement artifact set differs")
    frozen(value)
    summary = dict(schema=1, complete=True, manifest=record(base / "manifest.json"),
                   invocations=len(results), candidate_measurements=sum(len(r["candidates"]) for r in results),
                   raw_samples=sum(len(c["raw_samples_us"]) for r in results for c in r["candidates"]),
                   results=results, limits=value["limits"])
    write_once(base / "analysis.json", summary)
    print(f"Audited {summary['invocations']} invocations and {summary['raw_samples']} raw samples.")


if __name__ == "__main__":
    main()
