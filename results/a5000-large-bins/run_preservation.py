#!/usr/bin/env python3
"""Record matched native-path preservation measurements; only root runs GPU stages."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD_EXE = ROOT / "build/custom-only/histogram_bench"
DEFAULT_NEW_EXE = ROOT / "build/large-bins/histogram_bench"
OLD_SHA256 = "d7568b5c93b80fbe39002c82866d08f2c75683b92ae99869d920c9896c481de7"
SOURCE_MANIFEST = ROOT / "results/a5000-custom-only/validation-manifest.json"
SCALING_ROOT = ROOT / "results/a5000-scaling/complete-catalog"
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
AUDITOR_PATH = ROOT / "results/a5000-custom-only/analyze_validation.py"
PROTOCOL = {"samples": 21, "batch": 32, "warmup_ms": 200, "timing_protocol": 3}
ROUNDS = {
    "1": {"seed": 2026092211, "order": ["old", "current", "current", "old"]},
    "2": {"seed": 2026092212, "order": ["current", "old", "old", "current"]},
}
EXPECTED_PRIOR_NAMES = {"small8", "smallbyte", "cachedbyte", "byte", "hot99", "sortedhot99",
                        "single", "large4096", "large4096-u64", "large8192-u64", "large16384-u64",
                        "cold4096-u64", "stream4096"}
NATIVE_CONTROLS = [
    ("native-boundary", 24577, "global:0:192:native:kernel"),
    ("native-medium", 65536, "global:1:384:native:kernel"),
    ("native-million", 1048576, "warp:4:96:native:kernel"),
]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = import_file("large_bin_preservation_recorder", RECORDER_PATH)
audit = import_file("large_bin_preservation_auditor", AUDITOR_PATH)
# Reuse the strict CSV/metadata auditor with this campaign's fresh declared seeds.
# This changes only the imported module instance, never the historical helper file.
audit.ROUNDS = ROUNDS
tune = audit.tune


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": recorder.sha256(path)}


def build_cases() -> list[dict]:
    source = audit.read_json(SOURCE_MANIFEST)
    if (source.get("schema") != 1 or source.get("protocol") != PROTOCOL
            or source.get("binaries", {}).get("current", {}).get("sha256") != OLD_SHA256):
        raise ValueError("prior preservation manifest does not identify the expected protocol and archived binary")
    names = [case["name"] for case in source["cases"]]
    if len(names) != 13 or set(names) != EXPECTED_PRIOR_NAMES:
        raise ValueError("prior preservation manifest does not contain exactly the required 13 cases")
    cases = copy.deepcopy(source["cases"])
    for case in cases:
        if case["variant"] != audit.variant(case["config"]) or case["config"]["algorithm"] not in tune.CUSTOM_ALGORITHMS:
            raise ValueError("invalid frozen custom configuration in prior preservation manifest")
        case["provenance"] = {
            "source_manifest": file_record(SOURCE_MANIFEST),
            "source_case": case["name"],
            "selection": "Exact previously recorded custom configuration; no retuning or automatic-default promotion.",
            "prior_provenance": case["provenance"],
        }
    for name, bins, expected_variant in NATIVE_CONTROLS:
        source_case = f"n{1 << 24}-b{bins}"
        selection_path = SCALING_ROOT / "cases" / source_case / "selection.json"
        selection = audit.read_json(selection_path)
        config = selection["chosen"]
        workload = selection["workload"]
        expected_workload = dict(n=1 << 24, bins=bins, input="u32", counter="u64",
                                 distribution="uniform", order="shuffled", cache="warm",
                                 launch="graph", warmup_ms=PROTOCOL["warmup_ms"])
        if (selection.get("schema") != 1 or selection.get("binary_sha256") != OLD_SHA256
                or workload != expected_workload or audit.variant(config) != expected_variant
                or selection.get("chosen_variant") != expected_variant or config["local_counter"] != "native"):
            raise ValueError(f"{selection_path}: native control differs from its declared historical selection")
        cases.append({"name": name, "workload": copy.deepcopy(workload), "config": copy.deepcopy(config),
                      "variant": expected_variant, "provenance": {
                          "source_selection": file_record(selection_path), "source_case": source_case,
                          "selection": "Frozen native global/warp winner from prior scaling; batch 32 is used in both binaries here."}})
    return cases


def build_manifest(new_exe: Path) -> dict:
    old, current = file_record(OLD_EXE), file_record(new_exe)
    if old["sha256"] != OLD_SHA256:
        raise ValueError("archived custom-only benchmark differs from the declared baseline hash")
    return {"schema": 1, "kind": "large_bin_native_path_preservation",
            "binaries": {"old": old, "current": current}, "protocol": PROTOCOL, "rounds": ROUNDS,
            "cases": build_cases(), "runner": file_record(Path(__file__)),
            "recorder": file_record(RECORDER_PATH), "auditor": file_record(AUDITOR_PATH),
            "parser": file_record(ROOT / "tools/autotune.py"), "production_default_promotion": False,
            "interpretation": "Adjacent old/current invocation medians compare identical frozen custom configurations. "
                              "Current/old ratios above one indicate slower current execution; all outcomes are retained.",
            "limits": [
                "The 13 original cases reproduce their recorded explicit configurations, not a new resolution of automatic defaults.",
                "Three native global/warp controls use identical batch-32 settings in both binaries; historical batch-4 latencies are not matched comparisons.",
                "Both seeds are fresh for this preservation campaign. Each case runs ABBA then BAAB; clocks remain unlocked.",
                "This is bounded preservation evidence, not proof of zero regressions or universal performance equivalence.",
                "The narrowed backend is evaluated separately. This runner makes no production-default promotion.",
            ]}


def command_for(binary: str, case: dict, seed: int) -> list[str]:
    return ([binary] + tune.workload_args(case["workload"], seed)
            + ["--variants", case["variant"], "--samples", str(PROTOCOL["samples"]),
               "--batch", str(PROTOCOL["batch"])])


def jobs_for(manifest: dict, stage: str, names: list[str] | None, directory: Path) -> list[tuple]:
    selected = set(names or [case["name"] for case in manifest["cases"]])
    known = {case["name"] for case in manifest["cases"]}
    if not selected <= known or (names is not None and len(names) != len(selected)):
        raise ValueError("case selection contains duplicate or unknown names")
    jobs = []
    for case in manifest["cases"]:
        if case["name"] not in selected:
            continue
        for number, round_info in ROUNDS.items():
            if stage != "all" and stage != "round" + number:
                continue
            for position, role in enumerate(round_info["order"], 1):
                stem = directory / "measurements" / ("round" + number) / f"{case['name']}-p{position}-{role}"
                jobs.append((stem, command_for(manifest["binaries"][role]["path"], case, round_info["seed"]), role))
    return jobs


def analyze(directory: Path, names: list[str] | None, rounds: list[int], available_only: bool) -> dict:
    manifest = audit.read_json(directory / "validation-manifest.json")
    if (manifest.get("kind") != "large_bin_native_path_preservation"
            or manifest.get("production_default_promotion") is not False
            or manifest.get("binaries", {}).get("old", {}).get("sha256") != OLD_SHA256
            or manifest.get("cases") != build_cases()):
        raise ValueError("preservation manifest differs from the declared frozen scope")
    report = audit.analyze(directory, rounds, names, available_only)
    # Reject unrecognized measurements instead of silently omitting evidence.
    expected = {stem.with_suffix(suffix).resolve()
                for stem, _, _ in jobs_for(manifest, "all", None, directory)
                for suffix in (".csv", ".log", ".command.json")}
    actual = {path.resolve() for path in (directory / "measurements").rglob("*") if path.is_file()}
    if actual - expected:
        raise ValueError("undeclared preservation artifacts: " + ", ".join(map(str, sorted(actual - expected))))
    report["production_default_promotion"] = False
    report["limitations"] = list(report["limitations"]) + manifest["limits"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("round1", "round2", "all", "analyze"))
    parser.add_argument("--output-root", type=Path, default=BASE / "preservation")
    parser.add_argument("--new-exe", type=Path, default=DEFAULT_NEW_EXE,
                        help="new benchmark binary; use a new output root when changing binaries")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--rounds", type=int, nargs="+", choices=(1, 2), default=[1, 2], help="analysis only")
    parser.add_argument("--available-only", action="store_true", help="analysis only: report missing invocations explicitly")
    parser.add_argument("--dry-run", action="store_true", help="print frozen manifest and planned jobs without querying or using the GPU")
    options = parser.parse_args(argv)
    directory = options.output_root.expanduser().resolve()
    if options.stage == "analyze":
        if options.dry_run:
            parser.error("--dry-run is for execution stages; analyze is already CPU-only")
        report = analyze(directory, options.cases, options.rounds, options.available_only)
        prefix = directory / "paired-comparison"
        prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        text = audit.markdown(report)
        text += "\n## Frozen scope\n\n" + "\n".join("- " + item for item in report["manifest"]["recorded"]["limits"]) + "\n"
        prefix.with_suffix(".md").write_text(text)
        print(f"Audited {len(report['cases'])} cases ({report['status']}); wrote {prefix}.json/.md")
        return 0
    if options.available_only or options.rounds != [1, 2]:
        parser.error("--available-only and --rounds are analysis options")
    manifest = build_manifest(options.new_exe.expanduser().resolve())
    jobs = jobs_for(manifest, options.stage, options.cases, directory)
    if options.dry_run:
        print(json.dumps({"manifest": manifest, "planned_invocations": len(jobs),
                          "jobs": [{"stem": str(stem), "role": role, "command": command}
                                   for stem, command, role in jobs]}, indent=2))
        return 0
    manifest_path = directory / "validation-manifest.json"
    if manifest_path.exists() and audit.read_json(manifest_path) != manifest:
        raise ValueError("existing manifest differs; use a new --output-root")
    for stem, _, _ in jobs:
        for suffix in (".csv", ".log", ".command.json"):
            if stem.with_suffix(suffix).exists():
                raise ValueError("refusing to overwrite artifact: " + str(stem.with_suffix(suffix)))
    if not (directory / "environment.json").exists() and (manifest_path.exists() or (directory / "measurements").exists()):
        raise ValueError("cannot adopt existing measurements without their original environment.json")
    recorder.EXE = Path(manifest["binaries"]["current"]["path"])
    recorder.ensure_environment(directory)
    if not manifest_path.exists():
        with manifest_path.open("x") as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write("\n")
    for stem, command, role in jobs:
        if recorder.sha256(Path(command[0])) != manifest["binaries"][role]["sha256"]:
            raise ValueError(f"{role} benchmark binary changed after manifest freeze")
        recorder.run(stem, command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tune.TuningError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
