#!/usr/bin/env python3
"""Record the fixed five-shape shared-overflow evaluation; root owns GPU execution.

--dry-run only reads and hashes local files and prints the proposed manifest.
All real measurements run serially through the existing run_round recorder.
Comparators are frozen before search. Confirmation always includes best overflow.
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
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
PARSER_PATH = ROOT / "tools/autotune.py"
spec = importlib.util.spec_from_file_location("overflow_recorder", RECORDER_PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
sys.path.insert(0, str(ROOT / "tools"))
import autotune

PROTOCOL = {"timing_protocol": 3, "batch": 4, "warmup_ms": 200,
            "default_samples": 3, "search_samples": 5,
            "validation_samples": 11, "confirmation_samples": 21}
SEEDS = {"search": 1500450271, "validation": [1500450283, 1500450307],
         "confirmation": [1500450311, 1500450323]}
GRIDS = [24, 48, 96, 192, 384, 768, 1536]
REFERENCE = "cub:2:192:native:kernel"
SOURCE_PATHS = [
    "CMakeLists.txt", "include/gh/histogram.hpp", "src/config.cpp", "src/defaults.cpp",
    "src/histogram.cu", "src/global_narrow.cu", "src/shared_overflow.cu", "src/bitplane.cu",
    "bench/benchmark.cpp", "bench/references.hpp", "bench/references.cu",
    "bench/sample256.cu", "bench/cache_flush.hpp", "bench/cache_flush.cu", "support/common.hpp",
]


def file_record(path):
    return {"path": str(path.resolve()), "sha256": recorder.sha256(path)}


def variant(row):
    return ":".join(str(row[key]) for key in
                    ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def family(row):
    return row["algorithm"], row["local_counter"]


def comparators(workload):
    bins = workload["bins"]
    if bins == 32768:
        source = ROOT / "results/a5000-scaling/complete-catalog/cases/n16777216-b32768/selection.json"
        prior = json.loads(source.read_text())
        native = "global:2:1536:native:kernel"
        if prior["workload"] != workload or prior["chosen_variant"] != native:
            raise ValueError("32,768-bin scaling control differs from its declared source")
        return {"kind": "explicit_controls_without_narrow_tuning", "source": file_record(source),
                "source_binary_sha256": prior["binary_sha256"], "native_variant": native,
                "narrow_variant": "global:2:192:u32:kernel",
                "extra_variants": ["global:2:192:native:kernel"],
                "note": "Native grid1536 comes from the prior scaling selection. Native and u32 grid192 "
                        "are explicit controls; no prior narrow search was run for 32,768 bins."}
    source = BASE / f"narrow-evaluation/cases/n16777216-b{bins}/selection.json"
    prior = json.loads(source.read_text())
    if prior["kind"] != "narrow_global_experiment_plan" or prior["workload"] != workload:
        raise ValueError("prior narrow comparison workload differs")
    chosen = {}
    for local in ("native", "u32"):
        candidates = [row for row in prior["evaluations"]
                      if row["config"]["local_counter"] == local
                      and row["config"]["algorithm"] in ("global", "warp")]
        if not candidates:
            raise ValueError(f"no prior {local} comparator for bins={bins}")
        chosen[local] = min(candidates, key=lambda row: (row["median_validation_us"],
                            -row["median_validation_reference_speedup"], row["variant"]))
    return {"kind": "frozen_prior_narrow_validation", "source": file_record(source),
            "source_binary_sha256": prior["binary_sha256"],
            "native_variant": chosen["native"]["variant"], "narrow_variant": chosen["u32"]["variant"],
            "extra_variants": [], "native_evaluation": chosen["native"], "narrow_evaluation": chosen["u32"],
            "rule": "Minimum prior validation median latency within each local-counter width; ties use "
                    "higher median reference-normalized speedup, then variant. Fixed before overflow search."}


def cases():
    overflow = [f"shared_overflow:{policy}:{grid}:u32:kernel"
                for policy in (14, 15) for grid in GRIDS]
    result = []
    for bins in (24577, 32768, 65536, 262144, 1048576):
        workload = {"n": 1 << 24, "bins": bins, "input": "u32", "counter": "u64",
                    "distribution": "uniform", "order": "shuffled", "cache": "warm",
                    "launch": "graph", "warmup_ms": PROTOCOL["warmup_ms"]}
        controls = comparators(workload)
        variants = list(dict.fromkeys(overflow + [controls["native_variant"], controls["narrow_variant"]]
                                     + controls["extra_variants"] + [REFERENCE]))
        result.append({"name": f"n{1 << 24}-b{bins}", "workload": workload,
                       "comparators": controls, "search_variants": variants})
    return result


def manifest(executable):
    return {"schema": 1, "kind": "shared_overflow_five_shape_evaluation",
            "binary": file_record(executable), "runner": file_record(Path(__file__)),
            "recorder": file_record(RECORDER_PATH), "parser": file_record(PARSER_PATH),
            "sources": [file_record(ROOT / path) for path in SOURCE_PATHS],
            "protocol": PROTOCOL, "seeds": SEEDS, "grids": GRIDS, "cases": cases(),
            "selection": "Validate the top four custom search policies, the best of every remaining "
                         "(algorithm, local_counter) family, the resolved default and the reference. "
                         "Choose custom by median paired reference/custom validation ratio, then median "
                         "custom latency and variant. Freeze before fresh confirmation seeds.",
            "overflow_selection": "Independently choose best validated shared_overflow with the same "
                                  "reference-normalized rule; always include it in confirmation even when it loses.",
            "confirmation": "Measure best overflow, chosen overall, frozen native, frozen narrow, resolved "
                            "default and NVIDIA reference together on two fresh seeds; deduplicate variants.",
            "limits": ["Five fixed N=2^24 shapes; uniform shuffled u32 input, u64 output, warm graph only.",
                       "Four comparator pairs come from frozen prior validation; 32,768-bin narrow is an untuned control.",
                       "NVIDIA histogram is a benchmark reference and cannot become the chosen policy.",
                       "No production default changes; selection is specific to this experiment.",
                       "Clocks are unlocked; two fresh seeds do not establish universal optimality."]}


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def assert_frozen(record):
    comparison_sources = [case["comparators"]["source"] for case in record["cases"]]
    for item in [record[key] for key in ("binary", "runner", "recorder", "parser")] + record["sources"] + comparison_sources:
        if recorder.sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("campaign executable or source changed: " + item["path"])


def command(record, case, seed, samples, variants=None):
    result = [record["binary"]["path"]] + autotune.workload_args(case["workload"], seed)
    result += ["--samples", str(samples), "--batch", str(PROTOCOL["batch"])]
    return result + (["--algorithm", "auto"] if variants is None
                     else ["--variants", ",".join(variants)])


def checked_rows(stem, case, seed, samples, variants, state):
    record = state["manifest"]
    assert_frozen(record)
    cmd = command(record, case, seed, samples, variants)
    paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise ValueError(f"incomplete invocation artifacts: {stem}")
    else:
        recorder.run(stem, cmd)
    metadata = json.loads(paths[2].read_text())
    binary = record["binary"]
    if (metadata["command"] != cmd or metadata["exit_code"] != 0
            or metadata["binary"] != binary["path"] or metadata["executable"] != binary["path"]
            or metadata["binary_sha256"] != binary["sha256"]
            or metadata["executable_sha256"] != binary["sha256"]
            or metadata["executables_unchanged"] is not True
            or metadata["gpus_before"] != recorder.ACTIVE_ENVIRONMENT["gpus"]
            or metadata["gpus_after"] != recorder.ACTIVE_ENVIRONMENT["gpus"]):
        raise ValueError(f"invocation identity or execution status differs: {stem}")
    assert_frozen(record)
    _, rows = autotune.parse_csv(paths[0].read_text())
    for row in rows:
        autotune.verify_row(row, case["workload"], state["benchmark_environment"], seed,
                            samples, PROTOCOL["batch"])
        state["benchmark_environment"] = autotune.extract(row, autotune.ENVIRONMENT)
        if row["gpu"] not in {gpu["name"] for gpu in recorder.ACTIVE_ENVIRONMENT["gpus"]}:
            raise ValueError(f"CSV GPU identity differs: {stem}")
        raw = sorted(row["raw_samples"])
        derived = {"median_us": raw[math.ceil(.5 * len(raw)) - 1],
                   "p95_us": raw[math.ceil(.95 * len(raw)) - 1],
                   "min_us": raw[0], "max_us": raw[-1],
                   "input_gb_s": row["n"] * 4 / (row["median_us"] * 1000)}
        if any(not math.isclose(row[key], value, rel_tol=1e-6, abs_tol=1e-6)
               for key, value in derived.items()):
            raise ValueError(f"CSV summary differs from raw samples: {stem}")
        key = (case["name"], variant(row))
        config = autotune.extract(row, autotune.CONFIG)
        if key in state["configs"] and state["configs"][key] != config:
            raise ValueError(f"configuration metadata changed: {stem}")
        state["configs"][key] = config
    if variants is None:
        if len(rows) != 1 or rows[0]["algorithm"] not in autotune.CUSTOM_ALGORITHMS:
            raise ValueError("automatic default must resolve to one custom policy")
    elif len(variants) != len(set(variants)) or sorted(variant(row) for row in rows) != sorted(variants):
        raise ValueError(f"candidate set differs: {stem}")
    return rows


def winner_key(score):
    return (-score["median_validation_reference_speedup"], score["median_validation_us"], score["variant"])


def execute_case(base, case, state):
    folder = base / "cases" / case["name"]
    controls = case["comparators"]
    # This artifact and the whole-campaign manifest precede all search timing.
    write_once(folder / "comparators.json", {"schema": 1, "case": case["name"],
                "workload": case["workload"], "frozen": controls})
    default = checked_rows(folder / "default", case, SEEDS["search"], PROTOCOL["default_samples"], None, state)[0]
    default_variant = variant(default)
    search_variants = list(dict.fromkeys(case["search_variants"] + [default_variant]))
    search = checked_rows(folder / "search", case, SEEDS["search"], PROTOCOL["search_samples"], search_variants, state)
    ranked = sorted((row for row in search if row["algorithm"] in autotune.CUSTOM_ALGORITHMS),
                    key=lambda row: (row["median_us"], variant(row)))
    finalists = ranked[:4]
    represented = {family(row) for row in finalists}
    for row in ranked:
        if family(row) not in represented:
            finalists.append(row)
            represented.add(family(row))
    finalist_variants = list(dict.fromkeys([variant(row) for row in finalists] + [default_variant, REFERENCE]))
    write_once(folder / "finalists.json", {"schema": 1, "case": case["name"],
               "search_csv": file_record(folder / "search.csv"), "default_csv": file_record(folder / "default.csv"),
               "default": autotune.extract(default, autotune.CONFIG), "variants": finalist_variants})
    validations = [checked_rows(folder / f"validation-s{seed}", case, seed,
                               PROTOCOL["validation_samples"], finalist_variants, state)
                   for seed in SEEDS["validation"]]
    scores = []
    for candidate in finalist_variants:
        if candidate == REFERENCE:
            continue
        own = [next(row for row in rows if variant(row) == candidate) for rows in validations]
        references = [next(row for row in rows if variant(row) == REFERENCE) for rows in validations]
        ratios = [reference["median_us"] / row["median_us"] for reference, row in zip(references, own)]
        scores.append({"variant": candidate, "config": autotune.extract(own[0], autotune.CONFIG),
                       "validation_reference_speedups": ratios,
                       "median_validation_reference_speedup": statistics.median(ratios),
                       "median_validation_us": statistics.median(row["median_us"] for row in own)})
    chosen = min(scores, key=winner_key)
    best_overflow = min((score for score in scores if score["config"]["algorithm"] == "shared_overflow"),
                       key=winner_key)
    search_by_variant = {variant(row): row for row in search}
    native_variant, narrow_variant = controls["native_variant"], controls["narrow_variant"]
    native = autotune.extract(search_by_variant[native_variant], autotune.CONFIG)
    narrow = autotune.extract(search_by_variant[narrow_variant], autotune.CONFIG)
    if chosen["config"]["algorithm"] not in autotune.CUSTOM_ALGORITHMS:
        raise ValueError("chosen policy must be custom")
    confirm_variants = list(dict.fromkeys([best_overflow["variant"], chosen["variant"],
                                          native_variant, narrow_variant, default_variant, REFERENCE]))
    plan = {"schema": 1, "kind": "shared_overflow_experiment_plan", "case": case["name"],
            "workload": case["workload"], "binary_sha256": state["manifest"]["binary"]["sha256"],
            "default": autotune.extract(default, autotune.CONFIG), "default_variant": default_variant,
            "chosen": chosen["config"], "chosen_variant": chosen["variant"],
            "best_overflow": best_overflow["config"], "best_overflow_variant": best_overflow["variant"],
            "native_comparator": native, "native_comparator_variant": native_variant,
            "narrow_comparator": narrow, "narrow_comparator_variant": narrow_variant,
            "comparator_provenance": file_record(folder / "comparators.json"),
            "evaluations": scores, "confirmation_variants": confirm_variants,
            "validation_csvs": [file_record(folder / f"validation-s{seed}.csv") for seed in SEEDS["validation"]],
            "production_change": False}
    write_once(folder / "selection.json", plan)
    confirmations = []
    for seed in SEEDS["confirmation"]:
        stem = folder / f"confirmation-s{seed}"
        rows = checked_rows(stem, case, seed, PROTOCOL["confirmation_samples"], confirm_variants, state)
        by_variant = {variant(row): row for row in rows}
        ours = by_variant[chosen["variant"]]
        overflow = by_variant[best_overflow["variant"]]
        comparison = {"seed": seed, "chosen_median_us": ours["median_us"],
                      "best_overflow_median_us": overflow["median_us"],
                      "chosen_over_overflow": ours["median_us"] / overflow["median_us"]}
        for name, label in (("native", native_variant), ("narrow", narrow_variant),
                            ("default", default_variant), ("reference", REFERENCE)):
            comparison[name + "_median_us"] = by_variant[label]["median_us"]
            comparison[name + "_over_chosen"] = by_variant[label]["median_us"] / ours["median_us"]
            comparison[name + "_over_overflow"] = by_variant[label]["median_us"] / overflow["median_us"]
        comparison["artifacts"] = [file_record(stem.with_suffix(suffix)) for suffix in (".csv", ".log", ".command.json")]
        confirmations.append(comparison)
    write_once(folder / "confirmation.json", {"schema": 1, "case": case["name"],
               "selection": file_record(folder / "selection.json"), "comparisons": confirmations})
    print("CONFIRMED", case["name"], "chosen", chosen["variant"],
          "best_overflow", best_overflow["variant"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--output-root", type=Path, default=BASE / "overflow-evaluation")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    record = manifest(options.exe.resolve())
    known = {case["name"] for case in record["cases"]}
    selected = options.cases or [case["name"] for case in record["cases"]]
    if len(selected) != len(set(selected)) or not set(selected) <= known:
        raise ValueError("unknown or duplicate case selection")
    if options.dry_run:
        print(json.dumps(record, indent=2, allow_nan=False))
        return 0
    base = options.output_root.resolve()
    if not (base / "environment.json").is_file() and (base / "cases").exists():
        if any(path.is_file() for path in (base / "cases").rglob("*")):
            raise ValueError("cannot adopt existing measurements without environment metadata")
    write_once(base / "manifest.json", record)
    recorder.EXE = Path(record["binary"]["path"])
    environment = recorder.ensure_environment(base)
    if environment["binary_sha256"] != record["binary"]["sha256"]:
        raise ValueError("environment executable differs from frozen manifest")
    state = {"manifest": record, "benchmark_environment": None, "configs": {}}
    for case in record["cases"]:
        if case["name"] in selected:
            execute_case(base, case, state)
    assert_frozen(record)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, autotune.TuningError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
