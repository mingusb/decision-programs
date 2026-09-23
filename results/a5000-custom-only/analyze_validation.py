#!/usr/bin/env python3
"""Audit temporally paired old/current preservation measurements without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune as tune

PROTOCOL = {"samples": 21, "batch": 32, "warmup_ms": 200, "timing_protocol": 3}
ROUNDS = {
    "1": {"seed": 424242, "order": ["old", "current", "current", "old"]},
    "2": {"seed": 987654, "order": ["current", "old", "old", "current"]},
}
LIMITATIONS = [
    "Ratios are current median / old median; values above 1 mean the current invocation was slower.",
    "Each ratio pairs adjacent invocations in the specified ABBA/BAAB order. Individual timing samples are not paired.",
    "The >=1.05 flag marks a suspicious regression for investigation. Smaller slowdowns remain reported and are not declared acceptable.",
    "These measurements do not establish statistical significance, prove zero regressions, or cover workloads absent from this manifest.",
    "Executable identities are validated from recorded hashes and paths; the analyzer does not execute or require the historical binaries.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def variant(config: dict) -> str:
    return ":".join(str(config[key]) for key in
                    ("algorithm", "tuning", "blocks", "local_counter", "clear_policy"))


def validate_manifest(manifest: dict, environment: dict) -> None:
    if manifest.get("schema") != 1 or manifest.get("protocol") != PROTOCOL:
        raise ValueError("unsupported validation manifest schema or measurement protocol")
    if manifest.get("rounds") != ROUNDS:
        raise ValueError("manifest must specify seed424242 ABBA and seed987654 BAAB rounds")
    binaries = manifest.get("binaries", {})
    if set(binaries) != {"old", "current"}:
        raise ValueError("manifest must identify exactly old and current binaries")
    for role, binary in binaries.items():
        if (not isinstance(binary, dict) or not isinstance(binary.get("path"), str)
                or not Path(binary["path"]).is_absolute()
                or not re.fullmatch(r"[0-9a-f]{64}", binary.get("sha256", ""))):
            raise ValueError(f"invalid {role} binary identity")
    if (environment.get("schema") != 1 or not environment.get("session_started_utc")
            or not isinstance(environment.get("uname"), dict)
            or environment.get("binary_sha256") != binaries["current"]["sha256"]
            or environment.get("binary") != binaries["current"]["path"]):
        raise ValueError("environment identity differs from the manifest current binary")
    gpus = environment.get("gpus")
    if (not isinstance(gpus, list) or not gpus
            or any(not isinstance(gpu, dict) or not gpu.get("name")
                   or not gpu.get("driver_version") or "uuid" not in gpu for gpu in gpus)):
        raise ValueError("missing recorded GPU/driver identity")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest contains no cases")
    names = []
    for case in cases:
        name = case.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid manifest case name")
        names.append(name)
        config, workload = case.get("config", {}), case.get("workload", {})
        if set(config) != set(tune.CONFIG) or set(workload) != set(tune.WORKLOAD):
            raise ValueError(f"{name}: manifest must contain exact complete CONFIG and WORKLOAD metadata")
        if config["algorithm"] not in tune.CUSTOM_ALGORITHMS:
            raise ValueError(f"{name}: preservation cases must use custom implementations")
        if config["clear_policy"] not in ("runtime", "kernel"):
            raise ValueError(f"{name}: unresolved clear policy")
        if case.get("variant") != variant(config) or "provenance" not in case:
            raise ValueError(f"{name}: invalid variant or missing selection provenance")
        if workload["warmup_ms"] != PROTOCOL["warmup_ms"]:
            raise ValueError(f"{name}: warmup differs from the validation protocol")
    if len(set(names)) != len(names):
        raise ValueError("duplicate manifest case name")


def command_flags(command: list[str], expected_binary: str) -> dict[str, str]:
    if (not isinstance(command, list) or not command
            or any(not isinstance(part, str) for part in command)
            or command[0] != expected_binary or len(command) % 2 != 1):
        raise ValueError("recorded command must directly invoke the expected binary with option/value pairs")
    flags = {}
    for index in range(1, len(command), 2):
        key, value = command[index:index + 2]
        if not key.startswith("--") or key in flags:
            raise ValueError("invalid or duplicate recorded command option: " + key)
        flags[key] = value
    return flags


def read_invocation(directory: Path, case: dict, number: int, position: int,
                    manifest: dict, environment: dict, available_only: bool) -> dict:
    round_info = manifest["rounds"][str(number)]
    role = round_info["order"][position - 1]
    stem = directory / "measurements" / f"round{number}" / f"{case['name']}-p{position}-{role}"
    paths = {name: stem.with_suffix(suffix) for name, suffix in
             (("csv", ".csv"), ("command_json", ".command.json"), ("log", ".log"))}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    identity = {"round": number, "seed": round_info["seed"], "position": position, "role": role}
    if missing:
        if not available_only:
            raise ValueError("missing measurement artifacts: " + ", ".join(missing))
        return dict(identity, status="pending", missing=missing,
                    existing_artifacts={name: {"path": str(path), "sha256": sha256(path)}
                                        for name, path in paths.items() if path.is_file()})
    recorded = read_json(paths["command_json"])
    binary = manifest["binaries"][role]
    if (recorded.get("exit_code") != 0 or recorded.get("executables_unchanged") is not True
            or recorded.get("binary_sha256") != binary["sha256"]
            or recorded.get("executable_sha256") != binary["sha256"]
            or recorded.get("binary") != binary["path"]
            or recorded.get("executable") != binary["path"]):
        raise ValueError(f"{paths['command_json']}: failed command or binary identity mismatch")
    if (recorded.get("gpus_before") != environment["gpus"]
            or recorded.get("gpus_after") != environment["gpus"]):
        raise ValueError(f"{paths['command_json']}: GPU/driver metadata differs from the session environment")
    seconds = recorded.get("seconds")
    if (not isinstance(seconds, (float, int)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds < 0
            or "telemetry_before" not in recorded or "telemetry_after" not in recorded):
        raise ValueError(f"{paths['command_json']}: incomplete invocation metadata")
    expected_flags = {"--" + key.replace("_", "-"): str(value)
                      for key, value in case["workload"].items()}
    expected_flags.update({"--seed": str(round_info["seed"]), "--variants": case["variant"],
                           "--samples": str(PROTOCOL["samples"]), "--batch": str(PROTOCOL["batch"])})
    flags = command_flags(recorded.get("command"), binary["path"])
    if "--clear" in flags:
        expected_flags["--clear"] = case["config"]["clear_policy"]
    if flags != expected_flags:
        raise ValueError(f"{paths['command_json']}: command differs from the exact manifest workload/variant/protocol")
    try:
        _, rows = tune.parse_csv(paths["csv"].read_text(encoding="utf-8"))
        if len(rows) != 1:
            raise ValueError(f"{paths['csv']}: expected exactly one custom measurement")
        row = rows[0]
        tune.verify_row(row, case["workload"], None, round_info["seed"],
                        PROTOCOL["samples"], PROTOCOL["batch"], case["config"])
    except tune.TuningError as error:
        raise ValueError(f"{paths['csv']}: {error}") from error
    if row["timing_protocol"] != PROTOCOL["timing_protocol"]:
        raise ValueError(f"{paths['csv']}: incorrect timing protocol")
    if row["gpu"] not in {gpu["name"] for gpu in environment["gpus"]}:
        raise ValueError(f"{paths['csv']}: GPU name differs from recorded device identities")
    samples = sorted(row["raw_samples"])
    expected_summary = {
        "median_us": samples[math.ceil(0.5 * len(samples)) - 1],
        "p95_us": samples[math.ceil(0.95 * len(samples)) - 1],
        "min_us": samples[0], "max_us": samples[-1],
    }
    input_bytes = row["n"] * (1 if row["input"] == "u8" else 4)
    expected_summary["input_gb_s"] = input_bytes / (row["median_us"] * 1000)
    for field, expected in expected_summary.items():
        if not math.isclose(row[field], expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{paths['csv']}: {field} disagrees with raw samples/workload")
    return dict(identity, status="complete", median_us=row["median_us"],
                raw_samples_us=row["raw_samples"], csv_metadata=row["csv_row"],
                benchmark_environment=tune.extract(row, tune.ENVIRONMENT),
                recorded=recorded,
                artifacts={name: {"path": str(path), "sha256": sha256(path)}
                           for name, path in paths.items()})


def summarize(ratios: list[float]) -> dict | None:
    if not ratios:
        return None
    return {
        "pair_count": len(ratios), "current_over_old_ratios": ratios,
        "median_current_over_old": statistics.median(ratios),
        "minimum_current_over_old": min(ratios), "maximum_current_over_old": max(ratios),
        "current_slower_pair_count": sum(value > 1 for value in ratios),
        "suspicious_regression_pair_count": sum(value >= 1.05 for value in ratios),
        "suspicious_regression": any(value >= 1.05 for value in ratios),
    }


def analyze(directory: Path, rounds: list[int], names: list[str] | None = None,
            available_only: bool = False) -> dict:
    directory = directory.expanduser().resolve()
    manifest_path, environment_path = directory / "validation-manifest.json", directory / "environment.json"
    manifest, environment = read_json(manifest_path), read_json(environment_path)
    validate_manifest(manifest, environment)
    if not rounds or len(set(rounds)) != len(rounds) or set(rounds) - {1, 2}:
        raise ValueError("select each requested round once, from 1 and 2")
    available_names = {case["name"] for case in manifest["cases"]}
    if names is not None and (not names or len(set(names)) != len(names) or set(names) - available_names):
        raise ValueError("case selection contains duplicate or unknown names")
    chosen = [case for case in manifest["cases"] if names is None or case["name"] in names]
    results, stable_environment = [], None
    for case in chosen:
        invocations, pairs, case_environment = [], [], None
        for number in rounds:
            group = [read_invocation(directory, case, number, position, manifest, environment, available_only)
                     for position in range(1, 5)]
            invocations.extend(group)
            for item in group:
                if item["status"] != "complete":
                    continue
                measured_environment = item["benchmark_environment"]
                # Cold and warm cases legitimately differ in eviction byte count.
                common = {key: value for key, value in measured_environment.items() if key != "eviction_bytes"}
                if stable_environment is not None and common != stable_environment:
                    raise ValueError(f"{case['name']}: benchmark environment changed across measurements")
                if case_environment is not None and measured_environment != case_environment:
                    raise ValueError(f"{case['name']}: benchmark environment changed across matched invocations")
                stable_environment, case_environment = common, measured_environment
            for left, right in ((group[0], group[1]), (group[2], group[3])):
                if left["status"] != "complete" or right["status"] != "complete":
                    continue
                by_role = {left["role"]: left, right["role"]: right}
                old, current = by_role["old"], by_role["current"]
                ratio = current["median_us"] / old["median_us"]
                pairs.append({"round": number, "seed": left["seed"],
                              "positions": [left["position"], right["position"]],
                              "order": [left["role"], right["role"]],
                              "old_median_us": old["median_us"], "current_median_us": current["median_us"],
                              "current_over_old": ratio, "current_slower": ratio > 1,
                              "suspicious_regression": ratio >= 1.05})
        complete = all(item["status"] == "complete" for item in invocations)
        results.append({"case": case["name"], "status": "complete" if complete else "partial",
                        "workload": case["workload"], "config": case["config"], "variant": case["variant"],
                        "provenance": case["provenance"], "benchmark_environment": case_environment,
                        "invocations": invocations, "pairs": pairs,
                        "summary": summarize([pair["current_over_old"] for pair in pairs])})
    return {"schema": 1, "kind": "matched_binary_preservation", "output_root": str(directory),
            "rounds": rounds, "available_only": available_only,
            "status": "complete" if all(case["status"] == "complete" for case in results) else "partial",
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path), "recorded": manifest},
            "environment": {"path": str(environment_path), "sha256": sha256(environment_path), "recorded": environment},
            "common_benchmark_environment": stable_environment, "cases": results,
            "limitations": LIMITATIONS}


def markdown(report: dict) -> str:
    lines = ["# Matched binary preservation measurements", "",
             f"Status: **{report['status']}** for requested rounds {', '.join(map(str, report['rounds']))}.", "",
             "Each ratio is current median / old median. Values above 1 indicate a slower current invocation.", "",
             "| Case | Status | Pairs | Median ratio | Min ratio | Max ratio | Slower pairs | >=1.05 pairs |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        summary = case["summary"]
        if summary is None:
            lines.append(f"| {case['case']} | {case['status']} | 0 | — | — | — | — | — |")
        else:
            lines.append(f"| {case['case']} | {case['status']} | {summary['pair_count']} | "
                         f"{summary['median_current_over_old']:.6f} | {summary['minimum_current_over_old']:.6f} | "
                         f"{summary['maximum_current_over_old']:.6f} | {summary['current_slower_pair_count']} | "
                         f"{summary['suspicious_regression_pair_count']} |")
    lines += ["", "## Adjacent invocation pairs", "",
              "| Case | Round | Positions | Order | Old median (µs) | Current median (µs) | Ratio | >=1.05 flag |",
              "|---|---:|---|---|---:|---:|---:|---|"]
    for case in report["cases"]:
        for pair in case["pairs"]:
            lines.append(f"| {case['case']} | {pair['round']} | {'/'.join(map(str, pair['positions']))} | "
                         f"{'→'.join(pair['order'])} | {pair['old_median_us']:.6f} | "
                         f"{pair['current_median_us']:.6f} | {pair['current_over_old']:.6f} | "
                         f"{'yes' if pair['suspicious_regression'] else 'no'} |")
    missing = [path for case in report["cases"] for item in case["invocations"] for path in item.get("missing", [])]
    if missing:
        lines += ["", "## Missing artifacts", ""] + [f"- `{path}`" for path in missing]
    lines += ["", "## Interpretation and provenance", ""] + ["- " + text for text in LIMITATIONS]
    lines += ["", "All raw samples, complete invocation metadata, and artifact SHA256 hashes are retained in the JSON report.",
              "CSV summary values and bandwidth were checked against raw samples and workload size. Exact commands, variants, "
              "binary identities, GPU/driver metadata, and benchmark environments were validated.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--rounds", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--available-only", action="store_true", help="explicitly allow missing invocations and report partial coverage")
    parser.add_argument("--cases", nargs="+", help="optional manifest case names")
    parser.add_argument("--output-prefix", type=Path, help="output filename prefix; defaults to OUTPUT_ROOT/paired-comparison")
    options = parser.parse_args(argv)
    try:
        report = analyze(options.output_root, options.rounds, options.cases, options.available_only)
        prefix = options.output_prefix or options.output_root / "paired-comparison"
        prefix = prefix.expanduser().resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        markdown_path.write_text(markdown(report), encoding="utf-8")
        print(f"Validated {len(report['cases'])} cases ({report['status']}); wrote {json_path} and {markdown_path}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
