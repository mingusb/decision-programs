#!/usr/bin/env python3
"""Evaluate frozen large-bin policies on 15 extensions; never retune them.

Root owns GPU execution. --dry-run and --audit are CPU-only. Dry-run can show
pending source selections, but execution requires all selections to be complete.
Existing artifacts are checked and reused only when identical; never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DEFAULT_EXE = ROOT / "build/large-bins-overflow/histogram_bench"
SOURCE_ROOT = BASE / "overflow-evaluation"
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
PARSER_PATH = ROOT / "tools/autotune.py"
spec = importlib.util.spec_from_file_location("extension_recorder", RECORDER_PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
sys.path.insert(0, str(ROOT / "tools"))
import autotune

PROTOCOL = {"timing_protocol": 3, "samples": 11, "batch": 4, "warmup_ms": 200}
SEEDS = [1700450371, 1700450387]
SOURCE_PATHS = [
    "CMakeLists.txt", "include/gh/histogram.hpp", "src/config.cpp", "src/defaults.cpp",
    "src/histogram.cu", "src/global_narrow.cu", "src/shared_overflow.cu", "src/bitplane.cu",
    "bench/benchmark.cpp", "bench/references.hpp", "bench/references.cu",
    "bench/sample256.cu", "bench/cache_flush.hpp", "bench/cache_flush.cu", "support/common.hpp",
]
ROLES = {"chosen": "chosen", "overflow": "best_overflow", "native": "native_comparator",
         "narrow": "narrow_comparator", "default": "default"}
LIMITATIONS = [
    "This is a fixed-policy extension experiment, not a tuning search. No winner is selected from its results.",
    "Policies come from same-bin N=2^24 uniform shuffled warm-graph selections. Only output clearing changes for stream execution.",
    "The default comparator is explicitly global policy 2, grid 192, native u64; automatic dispatch is not invoked.",
    "Ratios compare candidates in the same invocation. Values above one favor the denominator policy; losses are retained.",
    "Two roles can name the same measured row after deduplication; a ratio of one then expresses identity, not an independent equivalence measurement.",
    "Warm means no explicit cache eviction, not guaranteed input residency. Input GB/s counts logical input bytes, not measured memory traffic.",
    "Clocks are unlocked. Two seeds and their median ranges are not confidence intervals or proof of universal superiority.",
    "NVIDIA histogram references are absent from this extension campaign. No production defaults change.",
]


def file_record(path):
    return {"path": str(path.resolve()), "sha256": recorder.sha256(path)}


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def check_file(record):
    if file_record(Path(record["path"])) != record:
        raise ValueError("frozen source changed: " + record["path"])


def variant(config):
    return ":".join(str(config[key]) for key in
                    ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def workload(n, bins, distribution="uniform", order="shuffled", cache="warm", launch="graph"):
    return {"n": n, "bins": bins, "input": "u32", "counter": "u64",
            "distribution": distribution, "order": order, "cache": cache,
            "launch": launch, "warmup_ms": PROTOCOL["warmup_ms"]}


def workloads():
    values = [workload((1 << 24) + 17, bins) for bins in (24577, 32768, 1048576)]
    values += [workload(1 << 28, bins) for bins in (24577, 1048576)]
    values += [workload((1 << 20) + 17, bins, "hot99", order)
               for bins in (24577, 32768, 1048576) for order in ("shuffled", "sorted")]
    values += [workload(1 << 24, bins, cache=cache, launch=launch)
               for bins in (24577, 1048576) for cache, launch in (("cold", "graph"), ("warm", "stream"))]
    if len(values) != 15:
        raise ValueError("extension matrix must contain exactly 15 workloads")
    return values


def source_selection(bins, binary_hash, allow_pending):
    path = SOURCE_ROOT / f"cases/n{1 << 24}-b{bins}/selection.json"
    if not path.is_file() and allow_pending:
        return {"pending_selection": str(path.resolve())}
    plan = read_json(path)
    if (plan.get("schema") != 1 or plan.get("kind") != "shared_overflow_experiment_plan"
            or plan.get("workload") != workload(1 << 24, bins)
            or plan.get("binary_sha256") != binary_hash or plan.get("production_change") is not False):
        raise ValueError(f"source selection identity differs: {path}")
    comparator_record = plan["comparator_provenance"]
    check_file(comparator_record)
    controls = read_json(Path(comparator_record["path"]))
    if (controls.get("schema") != 1 or controls.get("case") != plan["case"]
            or controls.get("workload") != plan["workload"]):
        raise ValueError(f"source comparator identity differs: {path}")
    prior_source = controls["frozen"]["source"]
    check_file(prior_source)
    for role in ("native", "narrow"):
        if controls["frozen"][role + "_variant"] != plan[ROLES[role] + "_variant"]:
            raise ValueError(f"source {role} comparator differs: {path}")
    configs = {}
    for role, source_role in ROLES.items():
        config = plan[source_role]
        if (set(config) != set(autotune.CONFIG) or variant(config) != plan[source_role + "_variant"]
                or config["algorithm"] not in autotune.CUSTOM_ALGORITHMS
                or config["clear_policy"] != "kernel"):
            raise ValueError(f"invalid {role} source configuration: {path}")
        configs[role] = dict(config)
    if (configs["overflow"]["algorithm"] != "shared_overflow"
            or configs["native"]["algorithm"] not in ("global", "warp")
            or configs["native"]["local_counter"] != "native"
            or configs["narrow"]["algorithm"] not in ("global", "warp")
            or configs["narrow"]["local_counter"] != "u32"
            or variant(configs["default"]) != "global:2:192:native:kernel"):
        raise ValueError(f"source policy roles differ: {path}")
    evidence = [file_record(path), comparator_record, prior_source] + plan["validation_csvs"]
    for item in evidence:
        check_file(item)
    return {"selection": file_record(path), "comparators": comparator_record,
            "evidence": evidence, "original_roles": configs}


def warp_control(bins, local, clear):
    return {"algorithm": "warp", "tuning": 4, "threads": 128, "items": 8,
            "replicas": 4, "blocks": 96, "scratch_bytes": bins * 4 if local == "u32" else 0,
            "local_counter": local, "load_policy": "scalar", "shared_limit": 48 * 1024,
            "clear_policy": clear}


def declared_cases(binary_hash, allow_pending):
    sources = {bins: source_selection(bins, binary_hash, allow_pending)
               for bins in (24577, 32768, 1048576)}
    result = []
    for shape in workloads():
        name = f"n{shape['n']}-b{shape['bins']}-{shape['distribution']}-{shape['order']}-{shape['cache']}-{shape['launch']}"
        source = sources[shape["bins"]]
        case = {"name": name, "workload": shape, "source": source}
        if "pending_selection" not in source:
            clear = "runtime" if shape["launch"] == "stream" else "kernel"
            configs = {role: dict(config, clear_policy=clear)
                       for role, config in source["original_roles"].items()}
            if shape["distribution"] == "hot99":
                configs.update({"warp_native": warp_control(shape["bins"], "native", clear),
                                "warp_narrow": warp_control(shape["bins"], "u32", clear)})
            candidates = {}
            for config in configs.values():
                label = variant(config)
                if label in candidates and candidates[label] != config:
                    raise ValueError("duplicate variant has inconsistent configuration metadata")
                candidates[label] = config
            case.update({"roles": {role: variant(config) for role, config in configs.items()},
                         "configs": candidates, "variants": list(candidates)})
        result.append(case)
    if len({case["name"] for case in result}) != 15:
        raise ValueError("duplicate extension workload names")
    return result


def manifest(executable, allow_pending=False):
    source_manifest = SOURCE_ROOT / "manifest.json"
    parent = read_json(source_manifest)
    old_seeds = {parent["seeds"]["search"], *parent["seeds"]["validation"], *parent["seeds"]["confirmation"]}
    if old_seeds.intersection(SEEDS) or len(set(SEEDS)) != 2:
        raise ValueError("extension seeds must be distinct and fresh")
    binary = file_record(executable)
    if parent["binary"] != binary:
        raise ValueError("extension executable differs from the source selection campaign")
    cases = declared_cases(binary["sha256"], allow_pending)
    return {"schema": 1, "kind": "frozen_large_bin_policy_extensions", "binary": binary,
            "runner": file_record(Path(__file__)), "recorder": file_record(RECORDER_PATH),
            "parser": file_record(PARSER_PATH), "source_campaign": file_record(source_manifest),
            "sources": [file_record(ROOT / path) for path in SOURCE_PATHS],
            "protocol": PROTOCOL, "seeds": SEEDS, "cases": cases,
            "ready": all("variants" in case for case in cases), "limits": LIMITATIONS,
            "selection_rule": "Freeze same-bin chosen overall, best overflow, prior native, prior narrow, "
                              "and explicit global:2:192:native before execution; add fixed warp:4:96 "
                              "native/u32 controls only for hot99. Deduplicate exact variants. No retuning.",
            "clear_adaptation": "Change only clear_policy to runtime for stream or kernel for graph."}


def write_once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"refusing to rewrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def assert_frozen(record):
    items = [record[key] for key in ("binary", "runner", "recorder", "parser", "source_campaign")]
    items += record["sources"]
    for case in record["cases"]:
        items += case["source"]["evidence"]
    unique = {item["path"]: item for item in items}
    for item in unique.values():
        check_file(item)


def command(record, case, seed):
    return [record["binary"]["path"]] + autotune.workload_args(case["workload"], seed) + [
        "--samples", str(PROTOCOL["samples"]), "--batch", str(PROTOCOL["batch"]),
        "--variants", ",".join(case["variants"])]


def checked_rows(stem, case, seed, state, audit):
    record, environment = state["manifest"], state["environment"]
    assert_frozen(record)
    cmd = command(record, case, seed)
    paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise ValueError(f"incomplete invocation artifacts: {stem}")
    elif audit:
        raise ValueError(f"missing invocation for CPU audit: {stem}")
    else:
        recorder.run(stem, cmd)
    metadata = read_json(paths[2])
    binary = record["binary"]
    if (metadata.get("command") != cmd or metadata.get("exit_code") != 0
            or metadata.get("binary") != binary["path"] or metadata.get("executable") != binary["path"]
            or metadata.get("binary_sha256") != binary["sha256"]
            or metadata.get("executable_sha256") != binary["sha256"]
            or metadata.get("executables_unchanged") is not True
            or metadata.get("gpus_before") != environment["gpus"]
            or metadata.get("gpus_after") != environment["gpus"]):
        raise ValueError(f"invocation command, executable, GPU identity or status differs: {stem}")
    seconds = metadata.get("seconds")
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds < 0
            or any(not isinstance(metadata.get(key), str) or not metadata[key]
                   for key in ("telemetry_before", "telemetry_after"))):
        raise ValueError(f"missing execution duration or telemetry: {stem}")
    _, rows = autotune.parse_csv(paths[0].read_text())
    if sorted(variant(row) for row in rows) != sorted(case["variants"]):
        raise ValueError(f"missing, duplicate or unexpected candidates: {stem}")
    cache = case["workload"]["cache"]
    for row in rows:
        autotune.verify_row(row, case["workload"], state["benchmark_environments"].get(cache), seed,
                            PROTOCOL["samples"], PROTOCOL["batch"], case["configs"][variant(row)])
        observed = autotune.extract(row, autotune.ENVIRONMENT)
        state["benchmark_environments"][cache] = observed
        stable = {key: value for key, value in observed.items() if key != "eviction_bytes"}
        if state["stable_environment"] is not None and state["stable_environment"] != stable:
            raise ValueError(f"benchmark environment changed across cache modes: {stem}")
        state["stable_environment"] = stable
        if row["gpu"] not in {gpu["name"] for gpu in environment["gpus"]}:
            raise ValueError(f"CSV GPU name differs from recorded GPU identity: {stem}")
        raw = sorted(row["raw_samples"])
        derived = {"median_us": raw[math.ceil(.5 * len(raw)) - 1],
                   "p95_us": raw[math.ceil(.95 * len(raw)) - 1],
                   "min_us": raw[0], "max_us": raw[-1],
                   "input_gb_s": row["n"] * 4 / (row["median_us"] * 1000)}
        if any(not math.isclose(row[key], value, rel_tol=1e-6, abs_tol=1e-6)
               for key, value in derived.items()):
            raise ValueError(f"CSV summary differs from raw timing samples: {stem}")
    assert_frozen(record)
    return rows, [file_record(path) for path in paths]


def execute_case(base, case, state, audit):
    folder = base / "cases" / case["name"]
    # This role/configuration artifact is written before either extension seed.
    frozen = {"schema": 1, "case": case["name"], "workload": case["workload"],
              "source": case["source"], "roles": case["roles"],
              "configs": case["configs"], "variants": case["variants"], "retuned": False}
    if audit:
        if read_json(folder / "frozen.json") != frozen:
            raise ValueError(f"frozen case selection differs: {folder}")
    else:
        write_once(folder / "frozen.json", frozen)
    measurements = []
    for seed in SEEDS:
        rows, artifacts = checked_rows(folder / f"confirmation-s{seed}", case, seed, state, audit)
        by_variant = {variant(row): row for row in rows}
        role_times = {role: by_variant[label]["median_us"] for role, label in case["roles"].items()}
        ratios = {denominator: {numerator: value / role_times[denominator]
                               for numerator, value in role_times.items()}
                  for denominator in ("chosen", "overflow")}
        measurements.append({"seed": seed, "role_median_us": role_times,
                             "ratios_by_denominator": ratios,
                             "candidates": [{"variant": variant(row), "config": autotune.extract(row, autotune.CONFIG),
                                              "median_us": row["median_us"], "p95_us": row["p95_us"],
                                              "min_us": row["min_us"], "max_us": row["max_us"],
                                              "input_gb_s": row["input_gb_s"], "raw_samples_us": row["raw_samples"]}
                                             for row in rows], "artifacts": artifacts})
    summary = {"schema": 1, "case": case["name"], "workload": case["workload"],
               "frozen": file_record(folder / "frozen.json"), "roles": case["roles"],
               "measurements": measurements, "retuned": False,
               "summary": {role: {"min": min(item["role_median_us"][role] for item in measurements),
                                  "median": statistics.median(item["role_median_us"][role] for item in measurements),
                                  "max": max(item["role_median_us"][role] for item in measurements)}
                           for role in case["roles"]}}
    if audit:
        if read_json(folder / "confirmation.json") != summary:
            raise ValueError(f"confirmation summary differs from raw measurements: {folder}")
    else:
        write_once(folder / "confirmation.json", summary)
    print(("AUDITED" if audit else "CONFIRMED"), case["name"], flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--output-root", type=Path, default=BASE / "extension-evaluation")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    options = parser.parse_args(argv)
    record = manifest(options.exe.resolve(), allow_pending=options.dry_run)
    if options.dry_run:
        print(json.dumps(record, indent=2, allow_nan=False))
        return 0
    if not record["ready"]:
        raise ValueError("all source selections must be complete before execution")
    base = options.output_root.resolve()
    if options.audit:
        if read_json(base / "manifest.json") != record:
            raise ValueError("recorded campaign manifest differs from the fixed declaration")
        environment = read_json(base / "environment.json")
    else:
        if not (base / "environment.json").is_file() and (base / "cases").exists():
            if any(path.is_file() for path in (base / "cases").rglob("*")):
                raise ValueError("cannot adopt measurements without recorded environment metadata")
        write_once(base / "manifest.json", record)
        recorder.EXE = Path(record["binary"]["path"])
        environment = recorder.ensure_environment(base)
    if (environment.get("schema") != 1 or environment.get("binary") != record["binary"]["path"]
            or environment.get("binary_sha256") != record["binary"]["sha256"]
            or not environment.get("gpus") or not environment.get("session_started_utc")):
        raise ValueError("recorded execution environment differs or is incomplete")
    state = {"manifest": record, "environment": environment,
             "benchmark_environments": {}, "stable_environment": None}
    cases = [execute_case(base, case, state, options.audit) for case in record["cases"]]
    assert_frozen(record)
    report = {"schema": 1, "kind": "audited_frozen_large_bin_policy_extensions", "status": "complete",
              "manifest": file_record(base / "manifest.json"),
              "environment": file_record(base / "environment.json"), "cases": cases,
              "case_count": len(cases), "invocation_count": len(cases) * len(SEEDS),
              "measurement_count": sum(len(item["candidates"]) for case in cases for item in case["measurements"]),
              "raw_sample_count": sum(len(row["raw_samples_us"]) for case in cases
                                      for item in case["measurements"] for row in item["candidates"]),
              "retuned": False, "production_change": False, "limits": LIMITATIONS}
    write_once(base / "extension-analysis.json", report)
    print(json.dumps({key: report[key] for key in
                     ("status", "case_count", "invocation_count", "measurement_count", "raw_sample_count")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, autotune.TuningError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
