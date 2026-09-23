#!/usr/bin/env python3
"""Audit recorded histogram scaling experiments without querying or using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune as tune

_spec = importlib.util.spec_from_file_location("scaling_runner", BASE / "run_scaling.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

LIMITATIONS = [
    "Search, validation, and confirmation are separate stages. The chosen custom configuration is frozen using validation seeds before the two fresh confirmation seeds are inspected.",
    "Confirmation ratios compare candidates measured in the same invocation: reference/custom and production-default/custom values above one favor the chosen custom configuration.",
    "Every candidate and raw timing sample is retained. A best search result is a result within the declared candidate set, not a claim of global optimality.",
    "The automatic-default invocation uses three samples on the search seed to identify its concrete configuration; performance comparisons use its fresh same-invocation confirmation measurements.",
    "When the selected configuration equals the production default, both labels refer to the same measured CSV row. Their ratio of one is an identity, not an independent performance-equivalence measurement.",
    "Uniform shuffled u32 inputs, u64 counts, warm cache, and graph execution are the only workloads covered. These observations do not establish behavior for other distributions, types, launch modes, or devices.",
    "Warm cache means no explicit eviction between operations; it does not mean these larger input arrays fit in the GPU cache. Input GB/s counts input bytes only, not output or internal memory traffic.",
    "This campaign uses timing protocol 3, batch 4, and 200 ms requested warmup. Earlier batch-32 campaign latencies are not treated as matched comparisons.",
    "Clocks are unlocked. Before/after telemetry does not capture every timed batch. Two confirmation seeds do not establish statistical significance or universal superiority.",
    "Requested clear_policy describes custom initialization. The NVIDIA histogram retains its own initialization; its scratch metadata is checked for within-case consistency without repeating a GPU workspace query.",
    "No production defaults are changed by this analysis or by the recorded experiment selection.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def variant(config: dict) -> str:
    return ":".join(str(config[key]) for key in
                    ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def command_flags(command: list[str], binary: str) -> dict[str, str]:
    if (not isinstance(command, list) or not command or command[0] != binary
            or len(command) % 2 != 1 or any(not isinstance(item, str) for item in command)):
        raise ValueError("benchmark command must directly invoke the declared executable with option/value pairs")
    flags = {}
    for key, value in zip(command[1::2], command[2::2]):
        if not key.startswith("--") or key in flags:
            raise ValueError("invalid or duplicate benchmark command option: " + key)
        flags[key] = value
    return flags


def audit_invocation(stem: Path, command: list[str], binary: dict, environment: dict,
                     workload: dict, seed: int, protocol: dict,
                     expected_variants: list[str] | None) -> dict:
    paths = {name: stem.with_suffix(extension) for name, extension in
             (("csv", ".csv"), ("command_json", ".command.json"), ("log", ".log"))}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing measurement artifact: {path}")
    metadata = read_json(paths["command_json"])
    if (metadata.get("exit_code") != 0 or metadata.get("executables_unchanged") is not True
            or metadata.get("binary_sha256") != binary["sha256"]
            or metadata.get("executable_sha256") != binary["sha256"]
            or metadata.get("binary") != binary["path"]
            or metadata.get("executable") != binary["path"]
            or metadata.get("command") != command):
        raise ValueError(f"{paths['command_json']}: command, executable identity, or execution status mismatch")
    if (metadata.get("gpus_before") != environment["gpus"]
            or metadata.get("gpus_after") != environment["gpus"]):
        raise ValueError(f"{paths['command_json']}: GPU/driver identity differs from the recorded session")
    seconds = metadata.get("seconds")
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds < 0
            or "telemetry_before" not in metadata or "telemetry_after" not in metadata):
        raise ValueError(f"{paths['command_json']}: missing or invalid execution metadata")
    flags = command_flags(command, binary["path"])
    requested = {"--" + key.replace("_", "-"): str(value) for key, value in workload.items()}
    requested.update({"--seed": str(seed), "--samples": str(protocol["samples"]),
                      "--batch": str(protocol["batch"])})
    if expected_variants is None:
        requested["--algorithm"] = "auto"
    else:
        if not expected_variants or len(expected_variants) != len(set(expected_variants)):
            raise ValueError("expected variant list is empty or contains duplicates")
        requested["--variants"] = ",".join(expected_variants)
    if flags != requested:
        raise ValueError(f"{paths['command_json']}: command differs from workload, seed, protocol, or candidate declaration")
    _, rows = tune.parse_csv(paths["csv"].read_text(encoding="utf-8"))
    labels = [variant(row) for row in rows]
    if expected_variants is None:
        if len(rows) != 1 or rows[0]["algorithm"] not in tune.CUSTOM_ALGORITHMS:
            raise ValueError(f"{paths['csv']}: automatic default must resolve to exactly one custom configuration")
    elif len(labels) != len(expected_variants) or set(labels) != set(expected_variants):
        raise ValueError(f"{paths['csv']}: missing, duplicate, or undeclared candidate measurements")
    records, common = [], None
    for row in rows:
        tune.verify_row(row, workload, common, seed, protocol["samples"], protocol["batch"])
        if row["timing_protocol"] != 3 or protocol["timing_protocol"] != 3:
            raise ValueError(f"{paths['csv']}: expected timing protocol 3")
        common = tune.extract(row, tune.ENVIRONMENT)
        if row["gpu"] not in {gpu["name"] for gpu in environment["gpus"]}:
            raise ValueError(f"{paths['csv']}: CSV GPU name differs from the recorded session")
        samples = sorted(row["raw_samples"])
        expected = {"median_us": samples[math.ceil(.5 * len(samples)) - 1],
                    "p95_us": samples[math.ceil(.95 * len(samples)) - 1],
                    "min_us": samples[0], "max_us": samples[-1],
                    "input_gb_s": row["n"] * (1 if row["input"] == "u8" else 4) / (row["median_us"] * 1000)}
        for key, value in expected.items():
            if not math.isclose(row[key], value, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{paths['csv']}: {key} differs from raw samples/workload")
        records.append({"variant": variant(row), "config": tune.extract(row, tune.CONFIG),
                        **{key: row[key] for key in tune.FLOAT_COLUMNS},
                        "raw_samples_us": row["raw_samples"], "csv_metadata": row["csv_row"]})
    return {"seed": seed, "measurements": records, "benchmark_environment": common,
            "recorded": metadata, "log_text": paths["log"].read_text(encoding="utf-8"),
            "artifacts": {name: file_record(path) for name, path in paths.items()}}


def validate_manifest(manifest: dict, environment: dict) -> None:
    if (manifest.get("schema") != 1 or manifest.get("kind") != "larger_histogram_workload_search"
            or manifest.get("protocol") != runner.PROTOCOL or manifest.get("seeds") != runner.SEEDS
            or manifest.get("grids") != runner.GRIDS or manifest.get("cases") != runner.cases()):
        raise ValueError("manifest differs from the declared scaling experiment matrix or protocol")
    binary = manifest.get("binary", {})
    if binary != {"path": str(runner.EXE.resolve()), "sha256": runner.EXPECTED_SHA}:
        raise ValueError("manifest binary differs from the preserved campaign executable")
    for key, path in (("runner", BASE / "run_scaling.py"), ("parser", ROOT / "tools/autotune.py"),
                      ("recorder", runner.RECORDER_PATH)):
        if manifest.get(key) != file_record(path):
            raise ValueError(f"{key} source differs from the version frozen by the manifest")
    if (environment.get("schema") != 1 or not environment.get("session_started_utc")
            or not isinstance(environment.get("uname"), dict)
            or environment.get("binary") != binary["path"]
            or environment.get("binary_sha256") != binary["sha256"]):
        raise ValueError("session environment differs from the campaign binary or lacks required metadata")
    gpus = environment.get("gpus")
    if (not isinstance(gpus, list) or not gpus or any(not isinstance(gpu, dict)
            or not gpu.get("name") or not gpu.get("driver_version") or "uuid" not in gpu for gpu in gpus)):
        raise ValueError("session environment lacks GPU/driver identity")
    seeds = [manifest["seeds"]["search"], *manifest["seeds"]["validation"], *manifest["seeds"]["confirmation"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("search, validation, and confirmation seeds must be distinct")


class PendingCase(Exception):
    def __init__(self, record: dict, missing: list[Path]):
        self.record = record
        self.missing = missing


def analyze_case(directory: Path, case: dict, manifest: dict, environment: dict) -> dict:
    folder = directory / "cases" / case["name"]
    record = {"case": case["name"], "workload": case["workload"], "status": "pending", "stages": {}}
    expected_artifacts, configurations, case_environment = set(), {}, None

    def require(paths: list[Path]) -> None:
        expected_artifacts.update(path.resolve() for path in paths)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            record["existing_artifacts"] = [file_record(path) for path in sorted(folder.glob("*")) if path.is_file()]
            raise PendingCase(record, missing)

    def invoke(name: str, seed: int, samples: int, variants: list[str] | None) -> dict:
        nonlocal case_environment
        stem = folder / name
        require([stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")])
        protocol = {"samples": samples, "batch": manifest["protocol"]["batch"], "timing_protocol": 3}
        result = audit_invocation(stem, runner.command(case, seed, samples, variants),
                                  manifest["binary"], environment, case["workload"], seed, protocol, variants)
        if case_environment is not None and case_environment != result["benchmark_environment"]:
            raise ValueError(f"{stem}: benchmark environment changed within the case")
        case_environment = result["benchmark_environment"]
        for row in result["measurements"]:
            label, config = row["variant"], row["config"]
            if label in configurations and configurations[label] != config:
                raise ValueError(f"{stem}: configuration metadata changed for {label}")
            configurations[label] = config
            if config["algorithm"] in tune.BASELINES:
                expected = dict(algorithm="cub", tuning=2, threads=0, items=0, replicas=0,
                                blocks=192, scratch_bytes=config["scratch_bytes"], local_counter="native",
                                load_policy="reference", shared_limit=0, clear_policy="kernel")
                if config != expected or config["scratch_bytes"] <= 0:
                    raise ValueError(f"{stem}: invalid NVIDIA histogram reference metadata")
        record["stages"][name] = result
        return result

    p, seeds = manifest["protocol"], manifest["seeds"]
    default = invoke("default", seeds["search"], p["default_samples"], None)["measurements"][0]
    default_variant = default["variant"]
    record.update(default=default["config"], default_variant=default_variant)
    search_variants = list(dict.fromkeys(case["search_variants"] + [default_variant]))
    search = invoke("search", seeds["search"], p["search_samples"], search_variants)
    ranked = sorted((row for row in search["measurements"] if row["config"]["algorithm"] in tune.CUSTOM_ALGORITHMS),
                    key=lambda row: (row["median_us"], row["variant"]))
    finalists = ranked[:4]
    represented = {row["config"]["algorithm"] for row in finalists}
    for row in ranked:
        if row["config"]["algorithm"] not in represented:
            finalists.append(row)
            represented.add(row["config"]["algorithm"])
    finalist_variants = list(dict.fromkeys([row["variant"] for row in finalists] + [default_variant, runner.REFERENCE]))
    finalists_path = folder / "finalists.json"
    require([finalists_path])
    finalist_manifest = read_json(finalists_path)
    expected_finalists = {"schema": 1, "case": case["name"], "search_csv": file_record(folder / "search.csv"),
                          "default_csv": file_record(folder / "default.csv"), "default": default["config"],
                          "variants": finalist_variants}
    if finalist_manifest != expected_finalists:
        raise ValueError(f"{finalists_path}: finalists differ from the frozen search selection rule or source hashes")
    record["finalists"] = {**file_record(finalists_path), "recorded": finalist_manifest}
    record["search_fastest_custom"] = ranked[0]
    validations = [invoke(f"validation-s{seed}", seed, p["validation_samples"], finalist_variants)
                   for seed in seeds["validation"]]
    evaluations = []
    for candidate in finalist_variants:
        if candidate == runner.REFERENCE:
            continue
        own = [next(row for row in item["measurements"] if row["variant"] == candidate) for item in validations]
        references = [next(row for row in item["measurements"] if row["variant"] == runner.REFERENCE) for item in validations]
        ratios = [ref["median_us"] / row["median_us"] for ref, row in zip(references, own)]
        evaluations.append({"variant": candidate, "config": own[0]["config"],
                            "validation_reference_speedups": ratios,
                            "median_validation_reference_speedup": statistics.median(ratios),
                            "median_validation_us": statistics.median(row["median_us"] for row in own)})
    chosen = min(evaluations, key=lambda row: (-row["median_validation_reference_speedup"],
                                              row["median_validation_us"], row["variant"]))
    if chosen["config"]["algorithm"] not in tune.CUSTOM_ALGORITHMS:
        raise ValueError("scaling selection must choose a custom implementation")
    selection_path = folder / "selection.json"
    require([selection_path])
    selection = read_json(selection_path)
    expected_selection = {"schema": 1, "kind": "scaling_experiment_plan", "case": case["name"],
                          "workload": case["workload"], "binary_sha256": manifest["binary"]["sha256"],
                          "default": default["config"], "chosen": chosen["config"],
                          "chosen_variant": chosen["variant"], "evaluations": evaluations,
                          "validation_csvs": [file_record(folder / f"validation-s{seed}.csv") for seed in seeds["validation"]],
                          "production_change": False}
    if selection != expected_selection:
        raise ValueError(f"{selection_path}: chosen configuration, scores, source hashes, or frozen selection rule differs")
    record["selection"] = {**file_record(selection_path), "recorded": selection}
    record.update(chosen=chosen["config"], chosen_variant=chosen["variant"])
    confirm_variants = list(dict.fromkeys([chosen["variant"], default_variant, runner.REFERENCE]))
    confirmations = []
    for seed in seeds["confirmation"]:
        item = invoke(f"confirmation-s{seed}", seed, p["confirmation_samples"], confirm_variants)
        rows = {row["variant"]: row for row in item["measurements"]}
        custom, reference, own_default = rows[chosen["variant"]], rows[runner.REFERENCE], rows[default_variant]
        confirmations.append({"seed": seed, "chosen_median_us": custom["median_us"],
                              "reference_median_us": reference["median_us"], "default_median_us": own_default["median_us"],
                              "reference_over_chosen": reference["median_us"] / custom["median_us"],
                              "default_over_chosen": own_default["median_us"] / custom["median_us"],
                              "chosen_input_gb_s": custom["input_gb_s"],
                              "faster_than_reference": custom["median_us"] < reference["median_us"],
                              "faster_than_default": custom["median_us"] < own_default["median_us"]})
    actual_artifacts = {path.resolve() for path in folder.iterdir() if path.is_file()}
    if actual_artifacts != expected_artifacts:
        raise ValueError(f"{folder}: unexpected case artifacts: {sorted(map(str, actual_artifacts - expected_artifacts))}")
    record.update(status="complete", confirmations=confirmations, benchmark_environment=case_environment)
    record["summary"] = {
        "confirmation_count": len(confirmations),
        "search_selected_differ": ranked[0]["variant"] != chosen["variant"],
        "selected_is_default": chosen["variant"] == default_variant,
        "reference_win_count": sum(item["faster_than_reference"] for item in confirmations),
        "default_win_count": sum(item["faster_than_default"] for item in confirmations),
    }
    for field in ("reference_over_chosen", "default_over_chosen", "chosen_median_us", "chosen_input_gb_s"):
        values = [item[field] for item in confirmations]
        record["summary"][field] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
    return record


def analyze(directory: Path, names: list[str] | None = None, available_only: bool = False) -> dict:
    directory = directory.expanduser().resolve()
    manifest_path, environment_path = directory / "manifest.json", directory / "environment.json"
    manifest, environment = read_json(manifest_path), read_json(environment_path)
    validate_manifest(manifest, environment)
    known = {case["name"] for case in manifest["cases"]}
    if names is not None and (not names or len(names) != len(set(names)) or not set(names) <= known):
        raise ValueError("case selection contains duplicate or unknown names")
    actual_dirs = {path.name for path in (directory / "cases").iterdir() if path.is_dir()} if (directory / "cases").exists() else set()
    if actual_dirs - known:
        raise ValueError("undeclared case directories: " + ", ".join(sorted(actual_dirs - known)))
    records, common = [], None
    for case in manifest["cases"]:
        if names is not None and case["name"] not in names:
            continue
        try:
            record = analyze_case(directory, case, manifest, environment)
        except PendingCase as pending:
            if not available_only:
                raise ValueError("missing case artifacts: " + ", ".join(map(str, pending.missing))) from pending
            record = pending.record
            record["missing_artifacts"] = list(map(str, pending.missing))
        for item in record["stages"].values():
            measured = item["benchmark_environment"]
            if common is not None and measured != common:
                raise ValueError("benchmark environment differs across scaling cases")
            common = measured
        records.append(record)
    completed = [case for case in records if case["status"] == "complete"]
    invocations = [item for case in records for item in case["stages"].values()]
    return {"schema": 1, "kind": "audited_scaling_search_and_heldout_confirmation",
            "status": "complete" if len(completed) == len(records) else "partial",
            "available_only": available_only, "selected_cases": names,
            "manifest": {**file_record(manifest_path), "recorded": manifest},
            "environment": {**file_record(environment_path), "recorded": environment},
            "common_benchmark_environment": common, "case_count": len(records),
            "completed_case_count": len(completed), "invocation_count": len(invocations),
            "measurement_count": sum(len(item["measurements"]) for item in invocations),
            "raw_sample_count": sum(len(row["raw_samples_us"]) for item in invocations for row in item["measurements"]),
            "cases": records, "limitations": LIMITATIONS}


def markdown(report: dict) -> str:
    lines = ["# Larger histogram scaling experiments", "",
             f"Status: **{report['status']}**. Audited {report['completed_case_count']}/{report['case_count']} requested cases, "
             f"{report['invocation_count']} invocations, and {report['measurement_count']} candidate measurements.", "",
             "Speedups below use fresh confirmation seeds and candidates measured in the same invocation. "
             "Reference/custom and default/custom ratios above 1 favor the selected custom configuration. "
             "The existing production default is resolved and measured separately for each workload, then included in confirmation.", "",
             "| Inputs | Bins | Chosen algorithm:policy:grid:local:clear | NVIDIA/custom range | Default/custom range | Custom µs range |",
             "|---:|---:|---|---:|---:|---:|"]
    for case in report["cases"]:
        w = case["workload"]
        if case["status"] != "complete":
            lines.append(f"| {w['n']:,} | {w['bins']:,} | pending | — | — | — |")
            continue
        s = case["summary"]
        ranges = [s[field] for field in ("reference_over_chosen", "default_over_chosen", "chosen_median_us")]
        lines.append(f"| {w['n']:,} | {w['bins']:,} | {case['chosen_variant']} | "
                     + " | ".join(f"{r['min']:.4f}–{r['max']:.4f}" for r in ranges) + " |")
    lines += ["", "## Search and selection", "",
              "Search ranks the declared custom candidates by median latency. Validation includes the four fastest, "
              "the fastest remaining candidate from every other custom algorithm family, the production default, and the NVIDIA histogram. "
              "Selection maximizes median speedup over the NVIDIA histogram across the two validation seeds; median latency and variant "
              "string break ties. Confirmation uses two further seeds after this choice is frozen.", "",
              "| Case | Fastest search custom | Validation-selected custom | Production default |", "|---|---|---|---|"]
    for case in report["cases"]:
        if case["status"] == "complete":
            lines.append(f"| {case['case']} | {case['search_fastest_custom']['variant']} | {case['chosen_variant']} | {case['default_variant']} |")
    lines += ["", "## Every confirmation comparison", "",
              "| Case | Seed | Chosen µs | NVIDIA µs | Default µs | NVIDIA/custom | Default/custom |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        for item in case.get("confirmations", []):
            lines.append(f"| {case['case']} | {item['seed']} | {item['chosen_median_us']:.6f} | {item['reference_median_us']:.6f} | "
                         f"{item['default_median_us']:.6f} | {item['reference_over_chosen']:.6f} | {item['default_over_chosen']:.6f} |")
    lines += ["", "## Interpretation and evidence", ""] + ["- " + limit for limit in LIMITATIONS]
    lines += ["", "The JSON report preserves every audited CSV row, raw timing sample, command, telemetry record, log, "
              "selection score, and artifact hash. The audit checks exact commands and candidate coverage, binary/GPU identity, "
              "workload/protocol consistency, raw-derived summaries, source hashes, and both deterministic selection stages. "
              "No GPU queries or execution are performed by this analyzer.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--available-only", action="store_true", help="explicitly permit and report unfinished cases")
    parser.add_argument("--output-prefix", type=Path, help="defaults to OUTPUT_ROOT/scaling-analysis")
    options = parser.parse_args(argv)
    try:
        report = analyze(options.output_root, options.cases, options.available_only)
        prefix = options.output_prefix or options.output_root / "scaling-analysis"
        prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
        print(f"Audited {report['completed_case_count']}/{report['case_count']} cases ({report['status']}), "
              f"{report['invocation_count']} invocations, {report['measurement_count']} measurements; wrote {prefix}.json/.md")
        return 0
    except (ValueError, KeyError, TypeError, OSError, tune.TuningError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
