#!/usr/bin/env python3
"""Tune a custom histogram plan; NVIDIA implementations are benchmark references only."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys


ALGORITHMS = {"cub", "global", "warp", "shared", "shared_rle", "shared_warp",
              "shared_partial", "shared_overflow", "bitplane", "nvidia_sample256"}
BASELINES = {"cub", "nvidia_sample256"}
CUSTOM_ALGORITHMS = ALGORITHMS - BASELINES
REFERENCE_MARGIN = 1.05
ENVIRONMENT = ("gpu", "sm", "driver_api", "runtime", "cub_version", "eviction_bytes", "timing_protocol")
WORKLOAD = ("n", "bins", "input", "counter", "distribution", "order", "cache", "launch", "warmup_ms")
CONFIG = ("algorithm", "tuning", "threads", "items", "replicas", "blocks", "scratch_bytes", "local_counter", "load_policy", "shared_limit", "clear_policy")
INTEGER_COLUMNS = ("n", "bins", "seed", "tuning", "threads", "items", "replicas", "blocks",
                   "scratch_bytes", "samples", "batch", "eviction_bytes", "timing_protocol", "warmup_ms", "shared_limit")
FLOAT_COLUMNS = ("median_us", "min_us", "p95_us", "max_us", "input_gb_s")
REQUIRED_COLUMNS = set(ENVIRONMENT + WORKLOAD + CONFIG + INTEGER_COLUMNS + FLOAT_COLUMNS
                       + ("sample_us",))


class TuningError(Exception):
    pass


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(row: dict, names: tuple[str, ...]) -> dict:
    return {name: row[name] for name in names}


def config_key(row: dict) -> tuple:
    return row["algorithm"], row["tuning"], row["blocks"], row["local_counter"], row["clear_policy"]


def family_key(row: dict) -> tuple:
    # Narrowed global counters add a separate counting/widening path. Preserve
    # its validation opportunity even if native policies fill the fastest slots.
    return row["algorithm"], row["local_counter"]


def parse_csv(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not REQUIRED_COLUMNS.issubset(fields):
        raise TuningError("benchmark emitted a missing, duplicate, or invalid CSV header")
    rows = []
    for line, original in enumerate(reader, start=2):
        if None in original or any(value is None for value in original.values()):
            raise TuningError(f"malformed benchmark CSV row {line}")
        row = dict(original)
        try:
            for key in INTEGER_COLUMNS:
                row[key] = int(row[key])
            for key in FLOAT_COLUMNS:
                row[key] = float(row[key])
            row["raw_samples"] = [float(value) for value in row["sample_us"].split(";")]
        except (TypeError, ValueError) as error:
            raise TuningError(f"invalid number in benchmark CSV row {line}") from error
        times = [row[key] for key in FLOAT_COLUMNS if key != "input_gb_s"] + row["raw_samples"]
        if (any(not math.isfinite(value) or value <= 0 for value in times)
                or not math.isfinite(row["input_gb_s"]) or row["input_gb_s"] < 0):
            raise TuningError(f"nonpositive or nonfinite benchmark timing in row {line}")
        if (row["samples"] <= 0 or row["batch"] <= 0
                or len(row["raw_samples"]) != row["samples"]):
            raise TuningError(f"incomplete benchmark samples in row {line}")
        if not row["min_us"] <= row["median_us"] <= row["p95_us"] <= row["max_us"]:
            raise TuningError(f"inconsistent benchmark timing summary in row {line}")
        # CUB owns dispatch: zero threads/items/replicas means not applicable,
        # while its requested tuning/blocks are retained only for replay.
        minimum_policy = 0 if row["algorithm"] in BASELINES else 1
        if (row["n"] < 0 or row["bins"] < 1 or row["scratch_bytes"] < 0
                or row["blocks"] < 1 or row["threads"] < minimum_policy
                or row["items"] < minimum_policy or row["replicas"] < minimum_policy
                or row["algorithm"] not in ALGORITHMS
                or row["tuning"] < 0 or row["local_counter"] not in ("native", "u32")
                or row["cache"] not in ("warm", "cold") or row["launch"] not in ("stream", "graph")
                or row["eviction_bytes"] < 0 or row["warmup_ms"] < 0 or row["shared_limit"] < 0
                or row["clear_policy"] not in ("runtime", "kernel")
                or row["timing_protocol"] != 3
                or row["load_policy"] not in ("scalar", "full_tile", "vector4", "reference")):
            raise TuningError(f"invalid benchmark configuration in row {line}")
        if any(not row[key] for key in ENVIRONMENT if key != "eviction_bytes"):
            raise TuningError(f"missing environment metadata in row {line}")
        if (row["cache"] == "cold") != (row["eviction_bytes"] > 0):
            raise TuningError("cache mode and eviction byte count disagree")
        if row["algorithm"] in ("global", "warp") and row["local_counter"] == "u32":
            expected_scratch = row["bins"] * 4 if row["n"] else 0
            if (row["counter"] != "u64" or row["n"] > (1 << 32) - 1
                    or row["load_policy"] != "scalar" or row["scratch_bytes"] != expected_scratch):
                raise TuningError("narrowed global counters require u64 output, n <= UINT32_MAX, "
                                  "a scalar policy, and one u32 scratch counter per bin for nonempty input")
        row["csv_row"] = original
        rows.append(row)
    if not rows:
        raise TuningError("benchmark emitted no measurements")
    return fields, rows


def workload_args(workload: dict, seed: int) -> list[str]:
    result = []
    for key in WORKLOAD:
        result.extend(("--" + key.replace("_", "-"), str(workload[key])))
    return result + ["--seed", str(seed)]


def candidate_args(config: dict) -> list[str]:
    return ["--algorithm", config["algorithm"], "--tuning", str(config["tuning"]),
            "--blocks", str(config["blocks"]), "--local-counter", config["local_counter"], "--clear", config["clear_policy"]]


def invoke(command: list[str], timeout: float) -> str:
    print("Running: " + shlex.join(command), file=sys.stderr, flush=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                check=False)
    except subprocess.TimeoutExpired as error:
        raise TuningError(f"benchmark exceeded {timeout:g} seconds") from error
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise TuningError(f"benchmark exited with status {result.returncode}")
    return result.stdout


def verify_row(row: dict, workload: dict, environment: dict | None, seed: int,
               samples: int, batch: int, config: dict | None = None) -> None:
    if extract(row, WORKLOAD) != workload:
        raise TuningError("benchmark workload metadata differs from the requested workload")
    if environment is not None and extract(row, ENVIRONMENT) != environment:
        raise TuningError("GPU name/SM, CUDA driver API/runtime, CUB, or eviction metadata changed")
    if (row["seed"], row["samples"], row["batch"]) != (seed, samples, batch):
        raise TuningError("benchmark seed or measurement protocol differs from the request")
    if config is not None and extract(row, CONFIG) != config:
        raise TuningError("benchmark configuration differs from the selected configuration")


def save_validation(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row["csv_row"] for row in rows)


def choose_custom(evaluations: list[dict]) -> dict:
    candidates = [item for item in evaluations if item["config"]["algorithm"] in CUSTOM_ALGORITHMS]
    if not candidates:
        raise TuningError("no custom finalist is available; NVIDIA algorithms are benchmark references only")
    return max(candidates, key=lambda item: item["holdout_reference_speedup"])


def tune(args: argparse.Namespace, executable: Path) -> None:
    workload = {key: getattr(args, key) for key in WORKLOAD}
    if args.n < 0 or args.bins < 1 or (args.input == "u8" and args.bins > 256):
        raise TuningError("workload requires n >= 0, bins >= 1, and u8 bins <= 256")
    if args.counter == "u32" and args.n > 0xffffffff:
        raise TuningError("u32 counters require n <= UINT32_MAX")
    if args.warmup_ms < 0:
        raise TuningError("warmup-ms must be nonnegative")
    if args.batch < 1 or args.search_samples < 3 or args.validation_samples < 3:
        raise TuningError("batch must be positive and sampling rounds must be at least three")
    seeds = args.validation_seeds
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or args.seed in seeds:
        raise TuningError("use at least two distinct validation seeds, separate from the search seed")
    if args.seed < 0 or any(seed < 0 for seed in seeds):
        raise TuningError("seeds must be nonnegative")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    search_path = output.with_name(output.stem + ".search.csv")
    validation_path = output.with_name(output.stem + ".validation.csv")
    # Refuse accidental overwrite of evidence or a saved plan from another run.
    if any(path.exists() for path in (output, search_path, validation_path)):
        raise TuningError("output plan or associated CSV already exists; choose a new --output path")

    build_hash = fingerprint(executable)
    search_command = [str(executable)] + workload_args(workload, args.seed)
    search_command += ["--clear", args.clear]
    search_command += ["--sweep", "--samples", str(args.search_samples), "--batch", str(args.batch)]
    search_text = invoke(search_command, args.timeout)
    search_path.write_text(search_text, encoding="utf-8")
    fields, search_rows = parse_csv(search_text)
    environment = extract(search_rows[0], ENVIRONMENT)
    for row in search_rows:
        verify_row(row, workload, environment, args.seed, args.search_samples, args.batch)
    if not any(row["algorithm"] in CUSTOM_ALGORITHMS for row in search_rows):
        raise TuningError("search did not provide a custom candidate; NVIDIA algorithms are benchmark references only")
    cub_rows = [row for row in search_rows if row["algorithm"] == "cub"]
    if not cub_rows:
        raise TuningError("search did not provide the required CUB baseline")
    baseline = min(cub_rows, key=lambda row: row["median_us"])
    references = [baseline]
    for algorithm in sorted(BASELINES - {"cub"}):
        matching = [row for row in search_rows if row["algorithm"] == algorithm]
        if matching:
            references.append(min(matching, key=lambda row: row["median_us"]))
    finalists = list(references)
    seen = {config_key(row) for row in references}
    for row in sorted(search_rows, key=lambda row: row["median_us"]):
        if row["algorithm"] not in BASELINES and config_key(row) not in seen:
            finalists.append(row)
            seen.add(config_key(row))
            if len(finalists) == len(references) + 4:
                break
    # A short search can otherwise fill all slots with near-duplicate policies
    # from one family and omit a more stable family. Preserve a representative
    # of every measured algorithm/local-counter path in addition to the four fastest rows.
    represented = {family_key(row) for row in finalists}
    for row in sorted(search_rows, key=lambda row: row["median_us"]):
        if family_key(row) not in represented:
            finalists.append(row)
            seen.add(config_key(row))
            represented.add(family_key(row))
    print(f"Validating {len(finalists)} finalists on {len(seeds)} independent seeds.",
          file=sys.stderr, flush=True)
    jobs = list(seeds)
    order_seed = args.seed ^ 0x475055
    randomizer = random.Random(order_seed)
    randomizer.shuffle(jobs)
    validation_rows = []
    validation_commands = []
    measurements = {}
    for seed in jobs:
        variants = list(finalists)
        randomizer.shuffle(variants)
        variant_list = ",".join(f"{row['algorithm']}:{row['tuning']}:{row['blocks']}:{row['local_counter']}:{row['clear_policy']}"
                                for row in variants)
        command = [str(executable)] + workload_args(workload, seed)
        command += ["--variants", variant_list, "--samples", str(args.validation_samples), "--batch", str(args.batch)]
        validation_fields, rows = parse_csv(invoke(command, args.timeout))
        expected = {config_key(candidate): extract(candidate, CONFIG) for candidate in finalists}
        if (validation_fields != fields or len(rows) != len(expected)
                or {config_key(row) for row in rows} != set(expected)):
            raise TuningError("validation must return exactly the selected variants and CSV schema")
        for row in rows:
            key = config_key(row)
            verify_row(row, workload, environment, seed, args.validation_samples, args.batch, expected[key])
            validation_rows.append(row)
            measurements[(key, seed)] = row
        validation_commands.append(command)
        save_validation(validation_path, fields, validation_rows)

    if fingerprint(executable) != build_hash:
        raise TuningError("benchmark executable changed during tuning; discard this run")
    evaluations = []
    for candidate in finalists:
        per_seed = []
        for seed in seeds:
            measured = measurements[(config_key(candidate), seed)]
            cub = measurements[(config_key(baseline), seed)]
            best_reference = min((measurements[(config_key(reference), seed)] for reference in references),
                                 key=lambda row: row["median_us"])
            per_seed.append({"seed": seed, "median_us": measured["median_us"],
                             "cub_median_us": cub["median_us"],
                             "speedup": cub["median_us"] / measured["median_us"],
                             "reference": best_reference["algorithm"],
                             "reference_median_us": best_reference["median_us"],
                             "reference_speedup": best_reference["median_us"] / measured["median_us"]})
        speedup = statistics.median(item["speedup"] for item in per_seed)
        evaluations.append({"config": extract(candidate, CONFIG),
                            "holdout_median_speedup": speedup,
                            "holdout_reference_speedup": statistics.median(item["reference_speedup"] for item in per_seed),
                            "per_seed": per_seed})
    winner = max(evaluations, key=lambda item: item["holdout_median_speedup"])
    chosen = choose_custom(evaluations)
    reference_ratios = [seed["reference_speedup"] for seed in chosen["per_seed"]]
    reference_comparison = {
        "wins": sum(ratio > 1.0 for ratio in reference_ratios),
        "ties": sum(ratio == 1.0 for ratio in reference_ratios),
        "losses": sum(ratio < 1.0 for ratio in reference_ratios),
        "seeds_meeting_margin": sum(ratio >= REFERENCE_MARGIN for ratio in reference_ratios),
        "all_seeds_meet_margin": all(ratio >= REFERENCE_MARGIN for ratio in reference_ratios),
        "minimum_speedup": min(reference_ratios),
        "maximum_speedup": max(reference_ratios),
    }
    launch = ([str(executable)] + workload_args(workload, seeds[0])
              + candidate_args(chosen["config"]) + ["--samples", str(args.validation_samples), "--batch", str(args.batch)])
    plan = {
        "schema": 5,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment,
        "build": {"executable": str(executable), "sha256": build_hash},
        "workload": workload,
        "scope": "Measured for this GPU, binary, size, bin count, types, distribution, order, cache and launch mode. "
                 "Validation seeds selected the custom finalist; NVIDIA implementations are benchmark references only. "
                 "A selected custom implementation may be slower than a reference; these are not universal performance claims.",
        "chosen": chosen["config"],
        "holdout_median_speedup": chosen["holdout_median_speedup"],
        "holdout_reference_speedup": chosen["holdout_reference_speedup"],
        "selection": {"custom_only": True,
                      "minimum_speedup_over_reference": REFERENCE_MARGIN,
                      "threshold_is_selection_gate": False,
                      "shortlist": "four fastest custom configurations plus best representative of each algorithm/local-counter family",
                      "gate": "custom implementation required; reference performance does not determine eligibility",
                      "ranking": "median of per-seed reference-normalized median-time speedups",
                      "references": [reference["algorithm"] for reference in references],
                      "reference_comparison": reference_comparison,
                      "retained_reference": False,
                      "retained_cub": False,
                      "best_finalist_median_speedup": winner["holdout_median_speedup"]},
        "finalists": evaluations,
        "search": {"seed": args.seed, "samples": args.search_samples, "batch": args.batch, "csv": str(search_path),
                   "csv_sha256": fingerprint(search_path), "command": search_command,
                   "measurements": len(search_rows)},
        "validation": {"seeds": seeds, "samples": args.validation_samples, "batch": args.batch,
                       "invocation_shuffle_seed": order_seed, "csv": str(validation_path),
                       "csv_sha256": fingerprint(validation_path), "commands": validation_commands},
        "launch_command": launch,
    }
    with output.open("x", encoding="utf-8") as destination:
        json.dump(plan, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"Saved {output}; selected {chosen['config']['algorithm']} "
          f"({chosen['holdout_reference_speedup']:.3f}x versus the fastest NVIDIA histogram reference "
          f"on validation medians; wins/ties/losses "
          f"{reference_comparison['wins']}/{reference_comparison['ties']}/{reference_comparison['losses']}; "
          f"{reference_comparison['seeds_meeting_margin']}/{len(seeds)} seeds meet the diagnostic "
          f"{REFERENCE_MARGIN:.2f}x margin).", file=sys.stderr)
    print("Launch: " + shlex.join(launch), file=sys.stderr)
    print(json.dumps({"plan": str(output), "chosen": chosen["config"],
                      "holdout_median_speedup": chosen["holdout_median_speedup"],
                      "holdout_reference_speedup": chosen["holdout_reference_speedup"],
                      "reference_comparison": reference_comparison}, allow_nan=False))


def replay(args: argparse.Namespace, plan: dict, executable: Path) -> None:
    if plan.get("schema") != 5:
        raise TuningError("production replay requires schema 5 custom-only plans; retune, or use matching "
                          "archived tools and binaries to reproduce historical benchmarks")
    if plan["chosen"]["algorithm"] not in CUSTOM_ALGORITHMS:
        raise TuningError("production plan must select a custom histogram; NVIDIA algorithms are benchmark references only")
    if fingerprint(executable) != plan["build"]["sha256"]:
        raise TuningError("benchmark executable SHA256 differs from the saved plan; retune")
    seed = plan["validation"]["seeds"][0]
    samples, batch = plan["validation"]["samples"], plan["validation"]["batch"]
    command = ([str(executable)] + workload_args(plan["workload"], seed)
               + candidate_args(plan["chosen"]) + ["--samples", str(samples), "--batch", str(batch)])
    text = invoke(command, args.timeout)
    _, rows = parse_csv(text)
    if len(rows) != 1:
        raise TuningError("replay must return exactly one benchmark row")
    verify_row(rows[0], plan["workload"], plan["environment"], seed, samples, batch, plan["chosen"])
    if fingerprint(executable) != plan["build"]["sha256"]:
        raise TuningError("benchmark executable changed during replay")
    print("Replay metadata verified; current performance is reported below.", file=sys.stderr)
    print(text, end="" if text.endswith("\n") else "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", help="benchmark executable (default: build/histogram_bench; "
                        "saved executable when replaying)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", help="new JSON plan path; sibling search/validation CSVs are saved")
    action.add_argument("--replay", help="verify and rerun a saved JSON plan")
    parser.add_argument("--n", type=int, default=1 << 24)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--input", choices=("u8", "u32"), default="u32")
    parser.add_argument("--counter", choices=("u32", "u64"), default="u32")
    parser.add_argument("--distribution", choices=("uniform", "single", "two", "hot90", "hot99"),
                        default="uniform")
    parser.add_argument("--order", choices=("shuffled", "sorted", "roundrobin"), default="shuffled")
    parser.add_argument("--cache", choices=("warm", "cold"), default="warm")
    parser.add_argument("--launch", choices=("stream", "graph"), default="stream")
    parser.add_argument("--warmup-ms", type=int, default=0,
                        help="untimed premeasurement warmup duration in milliseconds")
    parser.add_argument("--clear", choices=("auto", "runtime", "kernel"), default="auto",
                        help="custom output clearing; auto uses runtime for streams and kernel for graphs")
    parser.add_argument("--batch", type=int, default=32,
                        help="identical operations per sample in search, validation and replay")
    parser.add_argument("--search-samples", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=11)
    parser.add_argument("--seed", type=int, default=12345, help="search seed")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[67890, 24680])
    parser.add_argument("--timeout", type=float, default=3600,
                        help="maximum seconds per benchmark invocation (default: 3600)")
    args = parser.parse_args()
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise TuningError("timeout must be finite and positive")
        plan = None
        if args.replay:
            plan = json.loads(Path(args.replay).expanduser().read_text(encoding="utf-8"))
        default_exe = plan["build"]["executable"] if plan is not None else "build/histogram_bench"
        executable = Path(args.exe or default_exe).expanduser().resolve()
        if not executable.is_file():
            raise TuningError(f"benchmark executable does not exist: {executable}")
        if plan is not None:
            replay(args, plan, executable)
        else:
            tune(args, executable)
        return 0
    except (TuningError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"autotune: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
