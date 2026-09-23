#!/usr/bin/env python3
"""Recorded larger-workload search; only the root agent runs GPU work."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
EXE = ROOT / "build/custom-only/histogram_bench"
EXPECTED_SHA = "d7568b5c93b80fbe39002c82866d08f2c75683b92ae99869d920c9896c481de7"
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
spec = importlib.util.spec_from_file_location("scaling_recorder", RECORDER_PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
sys.path.insert(0, str(ROOT / "tools"))
import autotune

PROTOCOL = {"timing_protocol": 3, "batch": 4, "warmup_ms": 200,
            "default_samples": 3, "search_samples": 5,
            "validation_samples": 11, "confirmation_samples": 21}
SEEDS = {"search": 104729, "validation": [130363, 155921],
         "confirmation": [196613, 262147]}
GRIDS = [24, 48, 96, 192, 384, 768, 1536]
REFERENCE = "cub:2:192:native:kernel"


def file_record(path):
    return {"path": str(path.resolve()), "sha256": recorder.sha256(path)}


def variant(row):
    return ":".join(str(row[k]) for k in ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def candidates(bins):
    values = [f"{a}:{p}:{g}:native:kernel" for a in ("global", "warp")
              for p in (0, 1, 2, 3) for g in GRIDS]
    if bins <= 24576:
        values += [f"{a}:{p}:{g}:u32:kernel"
                   for a in ("shared", "shared_rle", "shared_warp", "shared_partial")
                   for p in (14, 15) for g in GRIDS]
    return values + [REFERENCE]


def cases():
    pairs = [(1 << power, bins) for power in range(24, 29)
             for bins in (16384, 24576, 32768, 65536)]
    pairs += [(1 << 24, bins) for bins in (24575, 24577, 131072, 262144, 1048576)]
    pairs += [(1 << power, bins) for power in (29, 30) for bins in (16384, 65536)]
    return [{"name": f"n{n}-b{bins}",
             "workload": {"n": n, "bins": bins, "input": "u32", "counter": "u64",
                          "distribution": "uniform", "order": "shuffled", "cache": "warm",
                          "launch": "graph", "warmup_ms": PROTOCOL["warmup_ms"]},
             "search_variants": candidates(bins)} for n, bins in pairs]


def manifest():
    binary = file_record(EXE)
    if binary["sha256"] != EXPECTED_SHA:
        raise ValueError("preserved benchmark hash differs from the declared binary")
    return {"schema": 1, "kind": "larger_histogram_workload_search",
            "binary": binary, "runner": file_record(Path(__file__)),
            "recorder": file_record(RECORDER_PATH), "parser": file_record(ROOT / "tools/autotune.py"),
            "protocol": PROTOCOL, "seeds": SEEDS, "grids": GRIDS, "cases": cases(),
            "selection": "Four fastest custom search candidates, best of every other custom family, "
                         "resolved production default and reference are validated together. Choose custom "
                         "by median reference-normalized validation speedup, tie-break by median latency "
                         "then variant string. Freeze before fresh confirmation seeds.",
            "limits": ["Uniform shuffled u32 inputs and u64 outputs, warm graph execution only.",
                       "Existing compile-time kernel catalog; grids extend to 1536 blocks.",
                       "Batch four differs from earlier batch-32 campaigns; compare within this campaign.",
                       "Clocks are unlocked; two held-out seeds do not establish universal optimality.",
                       "No production defaults are changed by this runner."]}


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def command(case, seed, samples, variants=None):
    result = [str(EXE)] + autotune.workload_args(case["workload"], seed)
    result += ["--samples", str(samples), "--batch", str(PROTOCOL["batch"])]
    return result + (["--algorithm", "auto"] if variants is None else ["--variants", ",".join(variants)])


def run_or_read(stem, cmd):
    if stem.with_suffix(".csv").exists():
        record = json.loads(stem.with_suffix(".command.json").read_text())
        if (record["command"] != cmd or record["exit_code"] != 0
                or record["binary_sha256"] != EXPECTED_SHA or not record["executables_unchanged"]
                or record["gpus_before"] != recorder.ACTIVE_ENVIRONMENT["gpus"]
                or record["gpus_after"] != recorder.ACTIVE_ENVIRONMENT["gpus"]):
            raise ValueError(f"cannot resume incompatible invocation {stem}")
    else:
        recorder.run(stem, cmd)
    _, rows = autotune.parse_csv(stem.with_suffix(".csv").read_text())
    return rows


def checked_rows(stem, case, seed, samples, variants=None):
    rows = run_or_read(stem, command(case, seed, samples, variants))
    for row in rows:
        autotune.verify_row(row, case["workload"], None, seed, samples, PROTOCOL["batch"])
    if variants is None:
        if len(rows) != 1 or rows[0]["algorithm"] in autotune.BASELINES:
            raise ValueError("automatic production selection must return one custom variant")
    elif sorted(variant(row) for row in rows) != sorted(variants):
        raise ValueError(f"variant set differs for {stem}")
    return rows


def execute_case(base, case):
    case_dir = base / "cases" / case["name"]
    default = checked_rows(case_dir / "default", case, SEEDS["search"], PROTOCOL["default_samples"])[0]
    default_variant = variant(default)
    search_variants = list(dict.fromkeys(case["search_variants"] + [default_variant]))
    search = checked_rows(case_dir / "search", case, SEEDS["search"], PROTOCOL["search_samples"], search_variants)
    ranked = sorted((row for row in search if row["algorithm"] not in autotune.BASELINES),
                    key=lambda row: (row["median_us"], variant(row)))
    finalists = ranked[:4]
    represented = {row["algorithm"] for row in finalists}
    for row in ranked:
        if row["algorithm"] not in represented:
            finalists.append(row)
            represented.add(row["algorithm"])
    finalist_variants = list(dict.fromkeys([variant(row) for row in finalists] + [default_variant, REFERENCE]))
    write_once(case_dir / "finalists.json", {"schema": 1, "case": case["name"],
                "search_csv": file_record(case_dir / "search.csv"),
                "default_csv": file_record(case_dir / "default.csv"),
                "default": autotune.extract(default, autotune.CONFIG),
                "variants": finalist_variants})
    validation = []
    for seed in SEEDS["validation"]:
        validation += checked_rows(case_dir / f"validation-s{seed}", case, seed,
                                   PROTOCOL["validation_samples"], finalist_variants)
    scores = []
    for candidate in finalist_variants:
        if candidate == REFERENCE:
            continue
        own = [row for row in validation if variant(row) == candidate]
        ratios = []
        for row in own:
            ref = next(r for r in validation if r["seed"] == row["seed"] and r["algorithm"] == "cub")
            ratios.append(ref["median_us"] / row["median_us"])
        scores.append({"variant": candidate, "config": autotune.extract(own[0], autotune.CONFIG),
                       "validation_reference_speedups": ratios,
                       "median_validation_reference_speedup": statistics.median(ratios),
                       "median_validation_us": statistics.median(row["median_us"] for row in own)})
    selected = min(scores, key=lambda r: (-r["median_validation_reference_speedup"],
                                          r["median_validation_us"], r["variant"]))
    plan = {"schema": 1, "kind": "scaling_experiment_plan", "case": case["name"],
            "workload": case["workload"], "binary_sha256": EXPECTED_SHA,
            "default": autotune.extract(default, autotune.CONFIG), "chosen": selected["config"],
            "chosen_variant": selected["variant"], "evaluations": scores,
            "validation_csvs": [file_record(case_dir / f"validation-s{s}.csv") for s in SEEDS["validation"]],
            "production_change": False}
    write_once(case_dir / "selection.json", plan)
    confirm_variants = list(dict.fromkeys([selected["variant"], default_variant, REFERENCE]))
    for seed in SEEDS["confirmation"]:
        checked_rows(case_dir / f"confirmation-s{seed}", case, seed,
                     PROTOCOL["confirmation_samples"], confirm_variants)
    print("CONFIRMED", case["name"], selected["variant"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    record = manifest()
    selected = set(options.cases or [case["name"] for case in record["cases"]])
    known = {case["name"] for case in record["cases"]}
    if not selected <= known:
        raise ValueError("unknown cases: " + str(selected - known))
    if options.dry_run:
        print(json.dumps(record, indent=2))
        return
    base = options.output_root.resolve()
    recorder.EXE = EXE
    recorder.ensure_environment(base)
    write_once(base / "manifest.json", record)
    for case in record["cases"]:
        if case["name"] in selected:
            execute_case(base, case)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, autotune.TuningError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
