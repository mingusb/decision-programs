#!/usr/bin/env python3
"""Frozen ABBA/BAAB follow-up: 2 MiB/two scans versus 1 MiB/four scans."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_window_experiment as shared


def declaration():
    value = json.loads(json.dumps(shared.manifest(ROOT / "build/window-experiment/histogram_bench", "window")))
    value["stage"] = "window_size_followup"
    value["sources"].append(shared.record(__file__))
    value["parent_campaign"] = shared.record(Path(__file__).parent / "analysis.json")
    value["seeds"] = [2026092291, 2026092292]
    value["protocol"] = "Eight processes, 2MiB/1MiB/1MiB/2MiB at first seed, reversed at second. Existing narrow kernel measured in every invocation. Twenty-one randomized rounds, batch4, 200ms warmup. Descriptive normalized paired comparison only."
    value["jobs"] = []
    for seed, pattern in zip(value["seeds"], ("ABBA", "BAAB")):
        for position, letter in enumerate(pattern):
            window = 524288 if letter == "A" else 262144
            job = dict(case=dict(id=0 if letter == "A" else 1, workload=shared.workload(),
                       window=window, variants=["global:0:48:u32:kernel", "global_window:0:48:u32:kernel"],
                       samples=21, batch=4), seed=seed)
            job["stem"] = f"s{seed}-{pattern}-p{position}-w{window}"
            value["jobs"].append(job)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    options = parser.parse_args()
    base = Path(__file__).parent / "window-sizes"
    value = declaration()
    if options.audit:
        if shared.read(base / "manifest.json") != value: raise ValueError("manifest changed")
        environment = shared.read(base / "environment.json")
    else:
        shared.write_once(base / "manifest.json", value)
        spec = importlib.util.spec_from_file_location("rec", shared.RECORDER)
        rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)
        rec.EXE = Path(value["binary"]["path"])
        environment = rec.ensure_environment(base)
    results = []
    for job in value["jobs"]:
        shared.frozen(value)
        stem = base / "measurements" / job["stem"]
        if not options.audit and not stem.with_suffix(".receipt.json").exists():
            rec.run(stem, shared.command(value["binary"]["path"], job))
            shared.write_once(stem.with_suffix(".receipt.json"), dict(manifest=shared.record(base / "manifest.json"),
                environment=shared.record(base / "environment.json"), artifacts=[shared.record(stem.with_suffix(suffix))
                for suffix in (".csv", ".log", ".command.json")]))
        results.append(shared.audit_job(base, value, environment, job))
    expected = {job["stem"] + suffix for job in value["jobs"]
                for suffix in (".csv", ".log", ".command.json", ".receipt.json")}
    if {p.name for p in (base / "measurements").iterdir()} != expected: raise ValueError("artifact set changed")
    if any(r["hardware"] != results[0]["hardware"] for r in results): raise ValueError("hardware changed")
    shared.frozen(value)
    shared.write_once(base / "analysis.json", dict(schema=1, complete=True, invocations=len(results),
        manifest=shared.record(base / "manifest.json"), results=results, limitations=value["limits"],
        production_default_promotion=False))
    print("Audited eight window-size follow-up processes.")


if __name__ == "__main__": main()
