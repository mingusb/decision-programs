#!/usr/bin/env python3
"""Validate the recorded reference stage and compare every frozen custom case."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("preservation_audit", BASE / "analyze_validation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
tune = audit.tune

LIMITATIONS = [
    "Speedup is the lowest applicable NVIDIA reference median divided by the frozen custom median from the same invocation. Values above 1 favor the custom implementation.",
    "Both recorded seeds and every applicable reference are retained. The best reference is selected separately for each seed; changes in its identity are flagged.",
    "These are finite observations of 13 predeclared workloads on one GPU/driver configuration, not statistical proof of superiority or a universal fastest-histogram claim.",
    "Custom configurations were frozen before these measurements. The byte case uses its predeclared custom configuration because its historical plan selected a reference.",
    "Recorded clear_policy is the requested custom initialization strategy. NVIDIA references retain their own initialization, as recorded in the benchmark logs.",
    "All variants use timing protocol 3, 21 samples, batch 32, and 200 ms requested warmup. Stream and graph timing and cold and warm cache conditions remain distinct workloads.",
    "CUB workspace sizes are recorded and checked for positivity and consistency between seeds; this CPU-only audit does not repeat a CUDA workspace query.",
]


def analyze(directory: Path) -> dict:
    directory = directory.expanduser().resolve()
    manifest_path = directory / "validation-manifest.json"
    environment_path = directory / "environment.json"
    manifest, environment = audit.read_json(manifest_path), audit.read_json(environment_path)
    audit.validate_manifest(manifest, environment)
    binary = manifest["binaries"]["current"]
    expected_artifacts, cases, common_environment = set(), [], None
    for case in manifest["cases"]:
        workload = case["workload"]
        algorithms = ["cub"]
        if (workload["input"] == "u8" and workload["counter"] == "u32"
                and workload["bins"] == 256 and workload["n"] % 4 == 0):
            algorithms.append("nvidia_sample256")
        labels = [case["variant"]] + [f"{name}:2:192:native:{case['config']['clear_policy']}" for name in algorithms]
        invocations, case_environment, reference_configs = [], None, None
        for round_info in manifest["rounds"].values():
            seed = round_info["seed"]
            stem = directory / "references" / f"{case['name']}-s{seed}"
            paths = {name: stem.with_suffix(suffix) for name, suffix in
                     (("csv", ".csv"), ("command_json", ".command.json"), ("log", ".log"))}
            for path in paths.values():
                if not path.is_file():
                    raise ValueError(f"missing reference artifact: {path}")
                expected_artifacts.add(path.resolve())
            metadata = audit.read_json(paths["command_json"])
            if (metadata.get("exit_code") != 0 or metadata.get("executables_unchanged") is not True
                    or metadata.get("binary_sha256") != binary["sha256"]
                    or metadata.get("executable_sha256") != binary["sha256"]
                    or metadata.get("binary") != binary["path"] or metadata.get("executable") != binary["path"]):
                raise ValueError(f"{paths['command_json']}: failed command or binary identity mismatch")
            if (metadata.get("gpus_before") != environment["gpus"]
                    or metadata.get("gpus_after") != environment["gpus"]):
                raise ValueError(f"{paths['command_json']}: GPU/driver identity changed")
            seconds = metadata.get("seconds")
            if (not isinstance(seconds, (float, int)) or isinstance(seconds, bool)
                    or not math.isfinite(seconds) or seconds < 0
                    or "telemetry_before" not in metadata or "telemetry_after" not in metadata):
                raise ValueError(f"{paths['command_json']}: missing execution metadata")
            expected_flags = {"--" + key.replace("_", "-"): str(value) for key, value in workload.items()}
            expected_flags.update({"--variants": ",".join(labels), "--seed": str(seed), "--samples": "21", "--batch": "32"})
            if audit.command_flags(metadata.get("command"), binary["path"]) != expected_flags:
                raise ValueError(f"{paths['command_json']}: reference command differs from the declared stage")
            _, rows = tune.parse_csv(paths["csv"].read_text(encoding="utf-8"))
            if len(rows) != len(labels) or {audit.variant(row) for row in rows} != set(labels):
                raise ValueError(f"{paths['csv']}: missing, duplicate, or unexpected custom/reference variant")
            records, measured_reference_configs = [], {}
            for row in rows:
                tune.verify_row(row, workload, None, seed, 21, 32,
                                case["config"] if row["algorithm"] == case["config"]["algorithm"] else None)
                measured_environment = tune.extract(row, tune.ENVIRONMENT)
                common = {key: value for key, value in measured_environment.items() if key != "eviction_bytes"}
                if row["gpu"] not in {gpu["name"] for gpu in environment["gpus"]}:
                    raise ValueError(f"{paths['csv']}: unexpected GPU name")
                if common_environment is not None and common != common_environment:
                    raise ValueError(f"{paths['csv']}: benchmark environment changed across measurements")
                if case_environment is not None and measured_environment != case_environment:
                    raise ValueError(f"{paths['csv']}: cache/environment changed within a case")
                common_environment, case_environment = common, measured_environment
                config = tune.extract(row, tune.CONFIG)
                if row["algorithm"] in algorithms:
                    expected_config = dict(algorithm=row["algorithm"], tuning=2, threads=0, items=0, replicas=0,
                                           blocks=192, scratch_bytes=row["scratch_bytes"], local_counter="native",
                                           load_policy="reference", shared_limit=0, clear_policy=case["config"]["clear_policy"])
                    if config != expected_config or config["scratch_bytes"] <= 0:
                        raise ValueError(f"{paths['csv']}: invalid reference configuration metadata")
                    if row["algorithm"] == "nvidia_sample256" and config["scratch_bytes"] != 245760:
                        raise ValueError(f"{paths['csv']}: NVIDIA sample workspace size differs from its fixed policy")
                    measured_reference_configs[row["algorithm"]] = config
                samples = sorted(row["raw_samples"])
                expected = {"median_us": samples[math.ceil(0.5 * len(samples)) - 1],
                            "p95_us": samples[math.ceil(0.95 * len(samples)) - 1], "min_us": samples[0], "max_us": samples[-1],
                            "input_gb_s": row["n"] * (1 if row["input"] == "u8" else 4) / (row["median_us"] * 1000)}
                for name, value in expected.items():
                    if not math.isclose(row[name], value, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError(f"{paths['csv']}: {name} differs from raw samples/workload")
                records.append({"config": config, "median_us": row["median_us"],
                                "min_us": row["min_us"], "p95_us": row["p95_us"], "max_us": row["max_us"],
                                "raw_samples_us": row["raw_samples"], "csv_metadata": row["csv_row"]})
            if reference_configs is not None and reference_configs != measured_reference_configs:
                raise ValueError(f"{paths['csv']}: reference configurations differ between seeds")
            reference_configs = measured_reference_configs
            custom = next(row for row in records if row["config"]["algorithm"] == case["config"]["algorithm"])
            references = [row for row in records if row["config"]["algorithm"] in algorithms]
            best = min(references, key=lambda row: row["median_us"])
            ratio = best["median_us"] / custom["median_us"]
            invocations.append({"seed": seed, "best_reference": best["config"]["algorithm"],
                                "custom_median_us": custom["median_us"], "best_reference_median_us": best["median_us"],
                                "custom_speedup": ratio, "custom_faster": ratio > 1,
                                "all_reference_speedups": {row["config"]["algorithm"]: row["median_us"] / custom["median_us"]
                                                           for row in references},
                                "measurements": records, "recorded": metadata,
                                "log_text": paths["log"].read_text(encoding="utf-8"),
                                "artifacts": {name: {"path": str(path), "sha256": audit.sha256(path)} for name, path in paths.items()}})
        ratios = [item["custom_speedup"] for item in invocations]
        winner_names = sorted({item["best_reference"] for item in invocations})
        cases.append({"case": case["name"], "workload": workload, "config": case["config"], "variant": case["variant"],
                      "provenance": case["provenance"], "benchmark_environment": case_environment,
                      "applicable_references": algorithms, "invocations": invocations,
                      "summary": {"seed_count": len(invocations), "minimum_custom_speedup": min(ratios),
                                  "maximum_custom_speedup": max(ratios), "median_custom_speedup": statistics.median(ratios),
                                  "custom_faster_seed_count": sum(ratio > 1 for ratio in ratios),
                                  "custom_slower_seed_count": sum(ratio < 1 for ratio in ratios),
                                  "faster_in_every_seed": all(ratio > 1 for ratio in ratios),
                                  "comparison_changes_direction": min(ratios) < 1 < max(ratios),
                                  "best_reference_algorithms": winner_names, "best_reference_changes": len(winner_names) > 1}})
    actual_artifacts = {path.resolve() for path in (directory / "references").iterdir() if path.is_file()}
    if actual_artifacts != expected_artifacts:
        raise ValueError("reference directory contains undeclared artifacts: " + ", ".join(map(str, actual_artifacts - expected_artifacts)))
    return {"schema": 1, "kind": "frozen_custom_vs_all_applicable_references", "status": "complete",
            "manifest": {"path": str(manifest_path), "sha256": audit.sha256(manifest_path), "recorded": manifest},
            "environment": {"path": str(environment_path), "sha256": audit.sha256(environment_path), "recorded": environment},
            "common_benchmark_environment": common_environment,
            "case_count": len(cases), "invocation_count": sum(len(case["invocations"]) for case in cases),
            "measurement_count": sum(len(item["measurements"]) for case in cases for item in case["invocations"]),
            "cases_faster_in_every_seed": sum(case["summary"]["faster_in_every_seed"] for case in cases),
            "cases": cases, "limitations": LIMITATIONS}


def markdown(report: dict) -> str:
    lines = ["# Frozen custom histogram versus NVIDIA references", "",
             f"Validated all {report['invocation_count']} invocations and {report['measurement_count']} measurements across "
             f"{report['case_count']} cases. The custom implementation was faster than the best applicable reference "
             f"on both recorded seeds in {report['cases_faster_in_every_seed']} cases.", "",
             "Speedup = best reference median / custom median. Values above 1 favor the custom implementation. "
             "Each range includes both seeds, 424242 and 987654.", "",
             "| Case | Best reference(s) | Seed 424242 speedup | Seed 987654 speedup | Speedup range | Direction changes |",
             "|---|---|---:|---:|---|---|"]
    for case in report["cases"]:
        summary = case["summary"]
        by_seed = {item["seed"]: item for item in case["invocations"]}
        lines.append(f"| {case['case']} | {', '.join(summary['best_reference_algorithms'])} | "
                     f"{by_seed[424242]['custom_speedup']:.6f}× | {by_seed[987654]['custom_speedup']:.6f}× | "
                     f"{summary['minimum_custom_speedup']:.6f}–{summary['maximum_custom_speedup']:.6f}× | "
                     f"{'yes' if summary['comparison_changes_direction'] else 'no'} |")
    lines += ["", "For cachedbyte, the fastest reference changes from nvidia_sample256 at seed 424242 to CUB at seed 987654.",
              "", "## Every recorded median", "",
              "| Case | Seed | Algorithm | Median (µs) | Reference/custom speedup |",
              "|---|---:|---|---:|---:|"]
    for case in report["cases"]:
        for item in case["invocations"]:
            for row in item["measurements"]:
                algorithm = row["config"]["algorithm"]
                ratio = item["all_reference_speedups"].get(algorithm)
                lines.append(f"| {case['case']} | {item['seed']} | {algorithm} | {row['median_us']:.6f} | "
                             + (f"{ratio:.6f}× |" if ratio is not None else "custom |"))
    lines += ["", "## Interpretation and validation", ""] + ["- " + text for text in LIMITATIONS]
    lines += ["", "The JSON report retains every raw sample, CSV row, command record, telemetry record, full log, and artifact hash. "
              "All expected files, executable identities, GPU/driver metadata, variants, workload fields, seeds, protocol fields, "
              "and CSV summaries were checked. No GPU queries or benchmark execution were performed by the analyzer.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE)
    options = parser.parse_args(argv)
    try:
        result = analyze(options.output_root)
        directory = options.output_root.expanduser().resolve()
        (directory / "reference-comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (directory / "reference-comparison.md").write_text(markdown(result), encoding="utf-8")
        print(f"Validated {result['case_count']} cases, {result['invocation_count']} invocations, {result['measurement_count']} measurements.")
        for case in result["cases"]:
            summary = case["summary"]
            print(f"{case['case']}: {summary['minimum_custom_speedup']:.6f}–{summary['maximum_custom_speedup']:.6f}x; "
                  f"best reference {','.join(summary['best_reference_algorithms'])}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, tune.TuningError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
