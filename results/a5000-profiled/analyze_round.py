#!/usr/bin/env python3
"""Audit frozen-plan confirmations from run_round.py without executing GPU work."""

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

_legacy_tuners = {}
ARCHIVED_PARSERS = {3: ROOT / "build" / "profiled" / "autotune.py",
                   4: ROOT / "build" / "profiled-clear-policy" / "autotune.py"}


def parser_for_schema(schema: int):
    if schema == 5:
        return tune
    if schema not in ARCHIVED_PARSERS:
        raise ValueError(f"unsupported plan schema: {schema}")
    if schema not in _legacy_tuners:
        spec = importlib.util.spec_from_file_location(f"profiled_schema{schema}_autotune", ARCHIVED_PARSERS[schema])
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        _legacy_tuners[schema] = parser
    return _legacy_tuners[schema]

CASES = ("small8", "smallbyte", "cachedbyte", "byte", "hot99", "sortedhot99", "single",
         "large4096", "large4096-u64", "large8192-u64", "large16384-u64", "cold4096-u64")
RUN_SEEDS = {1: 424242, 2: 424242, 3: 987654}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_label(config: dict) -> str:
    label = f"{config['algorithm']}:{config['tuning']}:{config['blocks']}:{config['local_counter']}"
    return label + ":" + config["clear_policy"] if "clear_policy" in config else label


def config(row: dict) -> dict:
    parser = parser_for_schema(4 if "clear_policy" in row else 3)
    return parser.extract(row, parser.CONFIG)


def read_rows(path: Path, parser=None) -> list[dict]:
    parser = parser or tune
    try:
        _, rows = parser.parse_csv(path.read_text(encoding="utf-8"))
    except parser.TuningError as error:
        raise ValueError(f"{path}: {error}") from error
    keys = [parser.config_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate configuration")
    for row in rows:
        raw = sorted(row["raw_samples"])
        expected = {
            "median_us": raw[math.ceil(0.5 * len(raw)) - 1],
            "p95_us": raw[math.ceil(0.95 * len(raw)) - 1],
            "min_us": raw[0], "max_us": raw[-1],
        }
        for name, measured in expected.items():
            if not math.isclose(row[name], measured, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{path}: {name} disagrees with raw samples")
    return rows


def timing(row: dict) -> dict:
    raw = row["raw_samples"]
    median = row["median_us"]
    return {
        "config": config(row), "median_us": median, "minimum_us": min(raw),
        "p95_us": row["p95_us"], "maximum_us": max(raw),
        "max_over_median": max(raw) / median,
        "samples_over_twice_median": sum(value > 2 * median for value in raw),
        "sample_count": len(raw), "raw_samples_us": raw,
    }


def command_option(command: list[str], option: str) -> str:
    if command.count(option) != 1:
        raise ValueError(f"confirmation command needs exactly one {option}")
    index = command.index(option)
    if index + 1 == len(command):
        raise ValueError(f"missing command value for {option}")
    return command[index + 1]


def analyze_case(directory: Path, case: str, available_only: bool) -> dict:
    provenance_path = directory / "reconfirmation" / f"{case}.json"
    provenance = None
    if (directory / "reconfirmation").exists():
        if not provenance_path.exists():
            if available_only:
                return {"case": case, "status": "pending_provenance", "missing": [str(provenance_path)]}
            raise ValueError(f"missing reconfirmation provenance: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (provenance.get("schema") != 1 or provenance.get("kind") != "frozen_prior_selection"
                or provenance.get("case") != case or provenance.get("clear_policy") != "kernel"):
            raise ValueError(f"{provenance_path}: invalid reconfirmation provenance")
        plan_path = Path(provenance["prior_plan"])
        expected_plan_path = Path(provenance["source_root"]) / "plans" / f"{case}.json"
        if plan_path.resolve() != expected_plan_path.resolve():
            raise ValueError(f"{provenance_path}: prior plan path differs from its declared source directory")
        if sha256(plan_path) != provenance["prior_plan_sha256"]:
            raise ValueError(f"{plan_path}: SHA256 differs from reconfirmation provenance")
    else:
        plan_path = directory / "plans" / f"{case}.json"
    if not plan_path.exists():
        if available_only:
            return {"case": case, "status": "pending_plan", "missing": [str(plan_path)]}
        raise ValueError(f"missing plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_tune = parser_for_schema(plan.get("schema"))
    if plan["schema"] == 5 and plan["chosen"]["algorithm"] in tune.BASELINES:
        raise ValueError(f"{plan_path}: schema5 plans must select a custom histogram, not a NVIDIA reference")
    measurement_schema = provenance.get("measurement_schema", 4) if provenance else plan["schema"]
    measurement_tune = parser_for_schema(measurement_schema)
    search_path = plan_path.with_name(case + ".search.csv")
    if provenance:
        if (provenance["prior_plan_schema"] != plan["schema"]
                or provenance["prior_binary_sha256"] != plan["build"]["sha256"]
                or Path(provenance["prior_search_csv"]).resolve() != search_path.resolve()
                or provenance["prior_search_csv_sha256"] != sha256(search_path)):
            raise ValueError(f"{provenance_path}: prior plan/search metadata mismatch")
    if sha256(search_path) != plan["search"]["csv_sha256"]:
        raise ValueError(f"{search_path}: SHA256 differs from plan")
    search = read_rows(search_path, plan_tune)
    if len(search) != plan["search"]["measurements"]:
        raise ValueError(f"{search_path}: search candidate count differs from plan")
    for row in search:
        try:
            plan_tune.verify_row(row, plan["workload"], plan["environment"],
                                 plan["search"]["seed"], plan["search"]["samples"], plan["search"]["batch"])
        except plan_tune.TuningError as error:
            raise ValueError(f"{search_path}: {error}") from error
    search_by_key = {plan_tune.config_key(row): row for row in search}
    prior_chosen_key = plan_tune.config_key(plan["chosen"])
    if prior_chosen_key not in search_by_key or config(search_by_key[prior_chosen_key]) != plan["chosen"]:
        raise ValueError(f"{plan_path}: chosen configuration not present in recorded search")
    scalar_candidates = [row for row in search
                         if row["tuning"] < 6 and row["algorithm"] not in tune.BASELINES]
    if not scalar_candidates:
        raise ValueError(f"{search_path}: missing preserved tuning0..5 scalar control")
    scalar = min(scalar_candidates, key=lambda row: row["median_us"])
    if scalar["load_policy"] != "scalar":
        raise ValueError(f"{search_path}: selected tuning<6 control is not scalar")
    references = [row for row in search if row["algorithm"] in tune.BASELINES]
    if not any(row["algorithm"] == "cub" for row in references):
        raise ValueError(f"{search_path}: required CUB reference missing")
    if {row["algorithm"] for row in references} != set(plan["selection"]["references"]):
        raise ValueError(f"{plan_path}: reference list differs from recorded search")
    def measured_config(row):
        value = config(row)
        return dict(value, clear_policy=provenance["clear_policy"]) if provenance else value

    chosen_config = measured_config(plan["chosen"])
    scalar_config = measured_config(scalar)
    chosen_key = measurement_tune.config_key(chosen_config)
    scalar_key = measurement_tune.config_key(scalar_config)
    expected = {measurement_tune.config_key(measured_config(row)): measured_config(row)
                for row in [search_by_key[prior_chosen_key], scalar] + references}
    expected_labels = {config_label(item) for item in expected.values()}
    if provenance and (len(provenance["expected_variants"]) != len(expected_labels)
                       or set(provenance["expected_variants"]) != expected_labels):
        raise ValueError(f"{provenance_path}: expected variants differ from frozen prior choices")
    used_seeds = {plan["search"]["seed"], *plan["validation"]["seeds"]}
    if used_seeds.intersection(RUN_SEEDS.values()):
        raise ValueError(f"{plan_path}: confirmation seed overlaps selection seeds")
    result = {
        "case": case, "status": "complete", "workload": plan["workload"],
        "environment": plan["environment"],
        "binary_sha256": provenance["binary_sha256"] if provenance else plan["build"]["sha256"],
        "selection_plan_schema": plan["schema"], "measurement_schema": measurement_schema,
        "plan": str(plan_path.resolve()), "plan_sha256": sha256(plan_path),
        "search_csv": str(search_path.resolve()), "search_csv_sha256": sha256(search_path),
        "search_candidate_count": len(search), "validation_finalist_count": len(plan["finalists"]),
        "chosen": chosen_config, "chosen_is_reference": plan["chosen"]["algorithm"] in tune.BASELINES,
        "scalar_control": scalar_config, "scalar_search_median_us": scalar["median_us"],
        "references": [measured_config(row) for row in references],
        "selection_reference_ratio": plan["holdout_reference_speedup"],
        "expected_confirmation_variants": sorted(expected_labels),
        "runs": [], "missing": [],
    }
    if provenance:
        result["reconfirmation_provenance"] = provenance
        result["reconfirmation_provenance_json"] = str(provenance_path.resolve())
        result["reconfirmation_provenance_sha256"] = sha256(provenance_path)
    for repeat, seed in RUN_SEEDS.items():
        path = directory / "confirmation" / f"{case}-r{repeat}.csv"
        command_path = path.with_suffix(".command.json")
        missing = [str(item) for item in (path, command_path) if not item.exists()]
        if missing:
            if not available_only:
                raise ValueError("missing confirmation artifacts: " + ", ".join(missing))
            result["missing"].extend(missing)
            result["status"] = "partial_confirmation"
            continue
        recorded = json.loads(command_path.read_text(encoding="utf-8"))
        if recorded["exit_code"] != 0 or recorded["binary_sha256"] != result["binary_sha256"]:
            raise ValueError(f"{command_path}: failure or binary mismatch")
        if recorded.get("executables_unchanged") is False:
            raise ValueError(f"{command_path}: executable changed during command")
        if ("gpus_before" in recorded and recorded.get("gpus_after") != recorded["gpus_before"]):
            raise ValueError(f"{command_path}: GPU/driver environment changed during command")
        command = recorded["command"]
        variants = command_option(command, "--variants").split(",")
        if len(variants) != len(set(variants)) or set(variants) != expected_labels:
            raise ValueError(f"{command_path}: variants must be exactly chosen, best scalar and all references")
        samples = int(command_option(command, "--samples"))
        batch = int(command_option(command, "--batch"))
        if (int(command_option(command, "--seed")) != seed or samples != 21
                or batch != 32 or batch != plan["validation"]["batch"]):
            raise ValueError(f"{command_path}: expected recorded round seed,21 samples,batch32 matching validation")
        rows = read_rows(path, measurement_tune)
        if {measurement_tune.config_key(row) for row in rows} != set(expected):
            raise ValueError(f"{path}: CSV candidates differ from required confirmation variants")
        for row in rows:
            try:
                measurement_tune.verify_row(row, plan["workload"], plan["environment"], seed, samples, batch,
                                            expected[measurement_tune.config_key(row)])
            except measurement_tune.TuningError as error:
                raise ValueError(f"{path}: {error}") from error
        by_key = {measurement_tune.config_key(row): row for row in rows}
        chosen = by_key[chosen_key]
        scalar_measured = by_key[scalar_key]
        reference = min((row for row in rows if row["algorithm"] in tune.BASELINES),
                        key=lambda row: row["median_us"])
        result["runs"].append({
            "repeat": repeat, "seed": seed, "samples": samples, "batch": batch,
            "csv": str(path.resolve()), "csv_sha256": sha256(path),
            "command_json": str(command_path.resolve()), "command_sha256": sha256(command_path),
            "variants_verified": True,
            "chosen": timing(chosen), "scalar_control": timing(scalar_measured),
            "strongest_reference": timing(reference),
            "strongest_reference_over_chosen": reference["median_us"] / chosen["median_us"],
            "scalar_control_over_chosen": scalar_measured["median_us"] / chosen["median_us"],
            "measurements": [timing(row) for row in rows],
            "recorded_seconds": recorded.get("seconds"),
            "telemetry_before": recorded.get("telemetry_before"),
            "telemetry_after": recorded.get("telemetry_after"),
        })
    runs = result["runs"]
    if runs:
        reference_ratios = [run["strongest_reference_over_chosen"] for run in runs]
        scalar_ratios = [run["scalar_control_over_chosen"] for run in runs]
        measurements = [item for run in runs for item in run["measurements"]]
        chosen_measurements = [run["chosen"] for run in runs]
        result["confirmation_summary"] = {
            "available_runs": len(runs), "complete_three_run_confirmation": len(runs) == 3,
            "median_reference_ratio": statistics.median(reference_ratios),
            "minimum_reference_ratio": min(reference_ratios),
            "median_scalar_ratio": statistics.median(scalar_ratios),
            "minimum_scalar_ratio": min(scalar_ratios),
            "all_three_reference_ratios_at_least_1_05": len(runs) == 3 and min(reference_ratios) >= 1.05,
            "chosen_samples": sum(item["sample_count"] for item in chosen_measurements),
            "chosen_samples_over_twice_median": sum(item["samples_over_twice_median"] for item in chosen_measurements),
            "chosen_largest_max_over_median": max(item["max_over_median"] for item in chosen_measurements),
            "all_variant_samples": sum(item["sample_count"] for item in measurements),
            "all_variant_samples_over_twice_median": sum(item["samples_over_twice_median"] for item in measurements),
        }
        repeats = {run["repeat"]: run for run in runs}
        if 1 in repeats and 2 in repeats:
            times = [repeats[number]["chosen"]["median_us"] for number in (1, 2)]
            ratios = [repeats[number]["strongest_reference_over_chosen"] for number in (1, 2)]
            result["same_seed_repeat"] = {
                "seed": RUN_SEEDS[1], "chosen_median_us": times,
                "chosen_max_over_min_median": max(times) / min(times),
                "reference_ratios": ratios, "reference_ratio_max_over_min": max(ratios) / min(ratios),
            }
    return result


def markdown(report: dict) -> str:
    completed = report["completed_cases"]
    lines = [
        f"Frozen-plan confirmation: **{completed}/{len(report['cases'])} cases complete**. "
        "Search and validation selected each configuration before these runs. "
        "Confirmations evaluate that frozen choice; they do not independently choose a winner.", "",
        "Each confirmation contains the chosen configuration, the fastest preserved scalar custom configuration "
        "from the same search (tuning<6), and every recorded reference, deduplicated when configurations coincide. "
        "The analyzer verifies CSV configuration sets, command variants, workload metadata and recorded binary hashes.", "",
        "R1/R2 use seed424242 in two separate invocations; R3 uses seed987654. "
        "Both seeds are excluded from this plan's search and validation. "
        "Every invocation has21 samples and batch32. Ratios are reference-time/chosen-time or scalar-time/chosen-time; "
        "values above1 favor the frozen choice. The strongest reference is recomputed within each invocation. "
        "Scalar comparisons use the current binary and timing protocol, not historical before timings.", "",
        "| Case | Chosen algorithm:tuning:blocks:local[:clear] | Selection search configs | Strongest reference / chosen, R1 / R2 / R3 | Scalar / chosen, R1 / R2 / R3 | Chosen >2× samples; largest max/median |",
        "|---|---|---:|---|---|---|",
    ]
    if report.get("prior_selection_reconfirmation"):
        lines[2:2] = [
            "These are reconfirmations of configurations selected by prior plans on a different recorded binary. "
            "The new binary uses explicit kernel clearing for these graph measurements. "
            "No new search or validation selection was performed; the search counts and selection ratios belong "
            "to the prior plans. The JSON preserves prior-plan, prior-binary, prior-search, new-binary and "
            "provenance hashes. Configuration matching permits only the declared clear-policy field change.", "",
        ]
    for case in report["cases"]:
        if "chosen" not in case:
            lines.append(f"| {case['case']} | pending plan | — | — | — | — |")
            continue
        by_repeat = {run["repeat"]: run for run in case["runs"]}
        reference_cells, scalar_cells = [], []
        for repeat in RUN_SEEDS:
            run = by_repeat.get(repeat)
            reference_cells.append("pending" if run is None else
                                   f"{run['strongest_reference']['config']['algorithm']} {run['strongest_reference_over_chosen']:.3f}×")
            scalar_cells.append("pending" if run is None else f"{run['scalar_control_over_chosen']:.3f}×")
        summary = case.get("confirmation_summary")
        outliers = "pending" if summary is None else (
            f"{summary['chosen_samples_over_twice_median']}/{summary['chosen_samples']}; "
            f"{summary['chosen_largest_max_over_median']:.2f}×")
        lines.append(f"| {case['case']} | {config_label(case['chosen'])} | {case['search_candidate_count']} | "
                     + " / ".join(reference_cells) + " | " + " / ".join(scalar_cells) + f" | {outliers} |")
    lines += ["", "| Case | Chosen load / shared-memory limit | Preserved scalar control | R1/R2 chosen median spread (max/min) |",
              "|---|---|---|---:|"]
    for case in report["cases"]:
        if "chosen" not in case:
            continue
        chosen = case["chosen"]
        repeat = case.get("same_seed_repeat")
        spread = "pending" if repeat is None else f"{repeat['chosen_max_over_min_median']:.3f}×"
        limit = "N/A" if chosen["algorithm"] in tune.BASELINES else f"{chosen['shared_limit'] / 1024:g}KiB"
        lines.append(f"| {case['case']} | {chosen['load_policy']} / {limit} | "
                     f"{config_label(case['scalar_control'])} | {spread} |")
    pending = [case["case"] for case in report["cases"] if case["status"] != "complete"]
    if pending:
        lines += ["", "This is a partial report. Pending cases: " + ", ".join(pending) + "."]
    lines += ["", "Ratios and raw-sample excursions describe these recorded runs. They do not establish universal "
              "performance or a precise tail distribution. Two repeats of one seed provide limited repeatability evidence. "
              "Reference choice may differ across runs; the JSON records all candidates, per-run medians and samples. "
              "No samples are discarded. Telemetry snapshots are preserved without inferring a clock/power cause.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--directory", dest="directory", type=Path, default=BASE,
                        help="session directory; --directory remains a compatibility alias")
    parser.add_argument("--available-only", action="store_true", help="emit an explicitly partial report")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output-prefix", type=Path, help="default: DIRECTORY/final-comparison")
    args = parser.parse_args()
    try:
        if len(set(args.cases)) != len(args.cases):
            raise ValueError("duplicate --cases entries")
        reconfirmation = (args.directory / "reconfirmation").exists()
        if reconfirmation and (args.directory / "plans").exists():
            raise ValueError("cannot mix new-selection plans and prior-selection reconfirmation in one directory")
        schemas = set()
        sources = set()
        for case in CASES:
            path = args.directory / ("reconfirmation" if reconfirmation else "plans") / f"{case}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            schemas.add(record.get("prior_plan_schema") if reconfirmation else record.get("schema"))
            if reconfirmation:
                sources.add(str(Path(record["source_root"]).resolve()))
        if len(schemas) > 1 or schemas - {3, 4, 5}:
            raise ValueError("directory must contain a single supported selection-plan schema (3, 4 or 5)")
        if len(sources) > 1:
            raise ValueError("reconfirmation directory must use one prior selection directory")
        cases = [analyze_case(args.directory, case, args.available_only) for case in args.cases]
        environment_path = args.directory / "environment.json"
        if reconfirmation and not environment_path.exists():
            raise ValueError("reconfirmation requires the new binary's session environment.json")
        environment = None
        if environment_path.exists():
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            if environment.get("schema") != 1 or not environment.get("session_started_utc"):
                raise ValueError(f"{environment_path}: invalid session environment metadata")
            for case in cases:
                if "binary_sha256" in case and case["binary_sha256"] != environment["binary_sha256"]:
                    raise ValueError(f"{case['case']}: plan binary hash differs from session environment")
                for run in case.get("runs", []):
                    recorded = json.loads(Path(run["command_json"]).read_text(encoding="utf-8"))
                    if ("gpus_before" in recorded and recorded["gpus_before"] != environment["gpus"]):
                        raise ValueError(f"{run['command_json']}: GPU/driver differs from session environment")
        report = {
            "schema": 1,
            "interpretation": "Confirmations evaluate configurations frozen by earlier search/validation; "
                              "confirmation measurements do not select new custom winners.",
            "prior_selection_reconfirmation": reconfirmation,
            "available_only": args.available_only,
            "completed_cases": sum(case["status"] == "complete" for case in cases),
            "cases": cases,
        }
        used_schemas = schemas | {case["measurement_schema"] for case in cases if "measurement_schema" in case}
        for schema in used_schemas & ARCHIVED_PARSERS.keys():
            parser_path = ARCHIVED_PARSERS[schema]
            report[f"schema{schema}_parser"] = {"path": str(parser_path), "sha256": sha256(parser_path)}
        if environment is not None:
            report["session_environment"] = environment
            report["session_environment_json"] = str(environment_path.resolve())
            report["session_environment_sha256"] = sha256(environment_path)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        rendered = markdown(report)
        prefix = (args.output_prefix or args.directory / "final-comparison").resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_name(prefix.name + ".json")
        md_path = prefix.with_name(prefix.name + ".md")
        json_path.write_text(encoded, encoding="utf-8")
        md_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {md_path} and {json_path}; {report['completed_cases']}/{len(cases)} complete cases.")
    except (OSError, ValueError, KeyError, TypeError, tune.TuningError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
