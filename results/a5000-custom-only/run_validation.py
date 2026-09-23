#!/usr/bin/env python3
"""Run serialized, matched archived/current histogram comparisons; root owns GPU execution."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RECORDER_PATH = ROOT / "results/a5000-profiled/run_round.py"
spec = importlib.util.spec_from_file_location("histogram_validation_recorder", RECORDER_PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)
sys.path.insert(0, str(ROOT / "tools"))
import autotune

OLD_EXE = ROOT / "build/defaults/histogram_bench"
CURRENT_EXE = ROOT / "build/histogram_bench"
OLD_SHA256 = "a91544fe771a19175b05611c51b1b08549314dcee418d624eeba58ab5444ce6d"
SOURCE_ROOT = ROOT / "results/a5000-profiled/resumed-2026-09-21"
STREAM_PLAN = ROOT / "results/a5000-defaults/stream-4096.json"
PROTOCOL = {"samples": 21, "batch": 32, "warmup_ms": 200, "timing_protocol": 3}
ROUNDS = {
    "1": {"seed": 424242, "order": ["old", "current", "current", "old"]},
    "2": {"seed": 987654, "order": ["current", "old", "old", "current"]},
}
CASE_NAMES = [case[0] for case in recorder.CASES] + ["stream4096"]


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": recorder.sha256(path)}


def config_from_row(row: dict, clear: str) -> dict:
    config = {key: row[key] for key in autotune.CONFIG if key != "clear_policy"}
    for key in set(config) & set(autotune.INTEGER_COLUMNS):
        config[key] = int(config[key])
    config["clear_policy"] = clear
    if config["algorithm"] in autotune.BASELINES:
        raise ValueError("matched comparisons must use a custom histogram")
    return config


def build_cases() -> list[dict]:
    """Freeze metadata from prior evidence without executing any CUDA operation."""
    cases = []
    for name, n, bins, input_type, counter, distribution, order, cache in recorder.CASES:
        path = SOURCE_ROOT / "plans" / f"{name}.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        workload = dict(n=n, bins=bins, input=input_type, counter=counter,
                        distribution=distribution, order=order, cache=cache,
                        launch="graph", warmup_ms=PROTOCOL["warmup_ms"])
        if {key: value for key, value in plan["workload"].items() if key != "warmup_ms"} != {
                key: value for key, value in workload.items() if key != "warmup_ms"}:
            raise ValueError(f"{path}: workload differs from the declared comparison case")
        provenance = {"prior_plan": file_record(path), "prior_binary_sha256": plan["build"]["sha256"],
                      "selection": "prior frozen custom choice; no new selection"}
        chosen = plan["chosen"]
        if name == "byte":
            search_path = path.with_suffix(".search.csv")
            if recorder.sha256(search_path) != plan["search"]["csv_sha256"]:
                raise ValueError(f"{search_path}: prior search hash mismatch")
            with search_path.open(newline="", encoding="utf-8") as source:
                matches = [row for row in csv.DictReader(source)
                           if (row["algorithm"], row["tuning"], row["blocks"], row["local_counter"])
                           == ("shared", "10", "96", "native")]
            if len(matches) != 1:
                raise ValueError("byte comparison requires exactly one shared:10:96:native source row")
            chosen = matches[0]
            provenance.update(prior_search=file_record(search_path),
                              selection="predeclared custom shared:10:96:native; prior plan chose a reference")
        config = config_from_row(chosen, "kernel")
        if name in ("sortedhot99", "single") and recorder.key(config) != "shared:6:192:native:kernel":
            raise ValueError(f"{name}: required original frozen policy changed")
        cases.append(dict(name=name, workload=workload, config=config,
                          variant=recorder.key(config), provenance=provenance))

    plan = json.loads(STREAM_PLAN.read_text(encoding="utf-8"))
    config = config_from_row(plan["chosen"], "runtime")
    if recorder.key(config) != "shared:10:48:native:runtime":
        raise ValueError("stream comparison requires shared:10:48:native:runtime")
    workload = dict(plan["workload"], warmup_ms=PROTOCOL["warmup_ms"])
    expected = dict(n=1 << 20, bins=4096, input="u32", counter="u32", distribution="uniform",
                    order="shuffled", cache="warm", launch="stream", warmup_ms=200)
    if workload != expected:
        raise ValueError("stream plan workload differs from the declared comparison")
    cases.append(dict(name="stream4096", workload=workload, config=config,
                      variant=recorder.key(config), provenance={"prior_plan": file_record(STREAM_PLAN),
                          "prior_binary_sha256": plan["build"]["sha256"],
                          "selection": "prior frozen stream custom choice; no new selection"}))
    return cases


def build_manifest() -> dict:
    old = file_record(OLD_EXE)
    if old["sha256"] != OLD_SHA256:
        raise ValueError("archived defaults benchmark SHA256 differs from the designated baseline")
    return {
        "schema": 1,
        "interpretation": "Temporally paired old/current complete-operation measurements of identical "
                          "predeclared custom configurations. Current/old ratios above one mean slower current execution; "
                          "these finite observations do not prove absence of regressions.",
        "binaries": {"old": old, "current": file_record(CURRENT_EXE)},
        "protocol": PROTOCOL, "rounds": ROUNDS, "cases": build_cases(),
        "recorder": file_record(RECORDER_PATH),
        "runner": file_record(Path(__file__)),
        "reference_stage": "Optional separate current-binary measurements; excluded from matched preservation ratios.",
    }


def command_for(executable: str, case: dict, seed: int, variants: str | None = None) -> list[str]:
    command = [executable]
    for key in autotune.WORKLOAD:
        command.extend(("--" + key.replace("_", "-"), str(case["workload"][key])))
    return command + ["--variants", variants or case["variant"], "--seed", str(seed),
                      "--samples", str(PROTOCOL["samples"]), "--batch", str(PROTOCOL["batch"])]


def jobs_for(manifest: dict, stage: str, selected: set[str], output_root: Path) -> list[tuple]:
    jobs = []
    for case in manifest["cases"]:
        if selected and case["name"] not in selected:
            continue
        if stage == "references":
            variants = [case["variant"], "cub:2:192:native:" + case["config"]["clear_policy"]]
            workload = case["workload"]
            if (workload["input"] == "u8" and workload["counter"] == "u32"
                    and workload["bins"] == 256 and workload["n"] % 4 == 0):
                variants.append("nvidia_sample256:2:192:native:" + case["config"]["clear_policy"])
            for round_info in ROUNDS.values():
                stem = output_root / "references" / f"{case['name']}-s{round_info['seed']}"
                command = command_for(manifest["binaries"]["current"]["path"], case,
                                      round_info["seed"], ",".join(variants))
                jobs.append((stem, command, "current"))
            continue
        for number, round_info in ROUNDS.items():
            if stage != "all" and stage != "round" + number:
                continue
            for position, role in enumerate(round_info["order"], 1):
                stem = output_root / "measurements" / ("round" + number) / f"{case['name']}-p{position}-{role}"
                command = command_for(manifest["binaries"][role]["path"], case, round_info["seed"])
                jobs.append((stem, command, role))
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("round1", "round2", "all", "references"))
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES)
    options = parser.parse_args(argv)
    selected = set(options.cases or [])
    if options.cases and len(selected) != len(options.cases):
        parser.error("duplicate --cases entries")
    output_root = options.output_root.resolve()
    manifest = build_manifest()
    manifest_path = output_root / "validation-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing validation manifest differs; use a new --output-root")
    jobs = jobs_for(manifest, options.stage, selected, output_root)
    for stem, _, _ in jobs:
        for suffix in (".csv", ".command.json", ".log"):
            path = stem.with_suffix(suffix)
            if path.exists():
                raise ValueError(f"refusing to overwrite existing artifact: {path}")
    if not (output_root / "environment.json").exists():
        if manifest_path.exists() or any((output_root / folder).exists() for folder in ("measurements", "references")):
            raise ValueError("cannot adopt comparison artifacts without their original environment.json")
    recorder.EXE = CURRENT_EXE
    recorder.ensure_environment(output_root)
    if not manifest_path.exists():
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
    for stem, command, role in jobs:
        if recorder.sha256(Path(command[0])) != manifest["binaries"][role]["sha256"]:
            raise ValueError(f"{role} executable changed since the manifest was frozen")
        recorder.run(stem, command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
