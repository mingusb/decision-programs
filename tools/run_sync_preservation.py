#!/usr/bin/env python3
"""Bounded synchronization/batch AA screen; --dry-run and --audit are CPU only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_paired_preservation as legacy
import analyze_paired_preservation as shared_audit

DEFAULT_EXE = ROOT / "build/preservation-sync/histogram_preservation"
DEFAULT_OUTPUT = ROOT / "results/a5000-sync-preservation-screen"
BATCHES = (32, 64, 128, 256)
MODES = ("position", "quartet")
COMPARISONS = ("old-old", "new-new")
REPETITIONS = (0, 1)
SCHEDULE_SEED = 2026092301
DATA_SEED = 2026092302
ORDER_SEEDS = (2026092303, 2026092304)
PROTOCOL = {"quartets": 32, "warmup_ms": 200}
FIELDS = shared_audit.FIELDS + ["sync_mode", "snapshot_storage", "timing_pair", "quartet_host_us", "position_host_scope"]
GROUP_FIELDS = ("case", "comparison", "batch", "sync_mode")
file_record, read_json, write_new = legacy.file_record, legacy.read_json, legacy.write_new


def schedule() -> list[dict]:
    blocks = [(case, comparison, batch, repetition) for case in legacy.CASES
              for comparison in COMPARISONS for batch in BATCHES for repetition in REPETITIONS]
    rng = random.Random(SCHEDULE_SEED)
    rng.shuffle(blocks)
    jobs = []
    for case, comparison, batch, repetition in blocks:
        # Each matched pair is adjacent; mode order reverses on repetition two.
        initial = (list(legacy.CASES).index(case) + COMPARISONS.index(comparison) + BATCHES.index(batch)) % 2
        modes = MODES if (initial + repetition) % 2 == 0 else MODES[::-1]
        for mode in modes:
            index = len(jobs)
            jobs.append({"index": index, "case": case, "comparison": comparison, "batch": batch,
                         "sync_mode": mode, "repetition": repetition, "data_seed": DATA_SEED,
                         "order_seed": ORDER_SEEDS[repetition],
                         "patterns": legacy.pattern_order(ORDER_SEEDS[repetition], PROTOCOL["quartets"]),
                         "stem": f"{index:02d}-{case}-{comparison}-b{batch}-{mode}-r{repetition}"})
    return jobs


def command_for(exe: str, job: dict) -> list[str]:
    return [exe, "--case", job["case"], "--comparison", job["comparison"],
            "--seed", str(job["data_seed"]), "--order-seed", str(job["order_seed"]),
            "--quartets", str(PROTOCOL["quartets"]), "--batch", str(job["batch"]),
            "--warmup-ms", str(PROTOCOL["warmup_ms"]), "--sync-mode", job["sync_mode"]]


def build_manifest(exe: Path, provenance: Path) -> dict:
    base = legacy.build_manifest(exe, provenance)
    jobs = schedule()
    for job in jobs:
        job["command"] = command_for(str(exe.resolve()), job)
    return {"schema": 1, "kind": "matched_synchronization_aa_screen", "binary": base["binary"],
            "protocol": PROTOCOL, "batches": list(BATCHES), "modes": list(MODES),
            "comparisons": list(COMPARISONS), "process_repetitions_per_stratum": len(REPETITIONS),
            "cases": legacy.CASES, "schedule_seed": SCHEDULE_SEED, "jobs": jobs,
            **{key: base[key] for key in ("sources", "build_artifacts", "source_provenance", "source_provenance_content", "recorder")},
            "runner": file_record(Path(__file__)),
            "dependencies": [file_record(Path(legacy.__file__)), file_record(Path(shared_audit.__file__))],
            "production_default_promotion": False,
            "metric_scopes": {
                "event_us": "CUDA events around complete histogram batch, divided by batch; excludes pinned snapshot copy",
                "submit_us": "host submission of timed batch only, divided by batch; excludes snapshot submission",
                "total_host_us": "position mode only: timed batch enqueue through pinned snapshot completion, divided by batch; blank in quartet mode",
                "quartet_host_us": "one host interval from before first untimed launch to synchronization completing fourth pinned copy, in microseconds per quartet; repeated on four rows, never four independent samples",
                "timing_pair": "A first/second occurrence use IDs 0/1, B first/second use 2/3; four unique event pairs and graph instances per quartet",
            },
            "limitations": [
                "AA screening only; no old/new performance conclusion, equivalence test, significance claim, or default promotion.",
                "Each case/comparison/batch/mode has two process repetitions. Never pool workloads, AA bindings, batches, or modes as replications of one effect.",
                "Pairs of modes run in adjacent distinct processes with reversed mode order on repetition two; GPU clocks remain unlocked.",
                "Both matched modes use four pinned snapshots and four occurrence-specific timing pairs/graphs. Compared with legacy, both change graph count and snapshot storage.",
                "Every quartet position is validated against CPU counts and canaries; no timing-position or outlier deletion.",
                "The position-mode quartet host interval includes CPU checks/event reads between positions. Both modes exclude validation after the fourth snapshot completion boundary.",
                "Each graph position has one preceding untimed graph batch; stream positions have one untimed operation. Host completion includes waiting for this work.",
                "Quartet mode cannot observe independent per-position host completion timestamps. It deliberately leaves total_host_us empty.",
                "Absolute latency and AA ratio variation are descriptive screens. Two repetitions cannot establish a precision plateau or zero loss.",
            ]}


def positive(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timing must be positive and finite: " + name)
    return value


def parse_rows(path: Path, job: dict, manifest: dict) -> list[dict]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV header differs from matched synchronization schema")
        rows = list(reader)
    count = manifest["protocol"]["quartets"] * 4
    if len(rows) != count:
        raise ValueError(f"expected {count} timing positions, found {len(rows)}")
    patterns = legacy.pattern_order(job["order_seed"], manifest["protocol"]["quartets"])
    if job["patterns"] != patterns:
        raise ValueError("frozen pattern order differs from independent shuffle")
    slot_a, slot_b = job["comparison"].split("-")
    fixed = {"case": job["case"], "comparison": job["comparison"], "slot_a": slot_a, "slot_b": slot_b,
             "data_seed": job["data_seed"], "order_seed": job["order_seed"], "batch": job["batch"],
             "sync_mode": job["sync_mode"], "snapshot_storage": "pinned_four_snapshots",
             "warmup_ms": manifest["protocol"]["warmup_ms"], **manifest["cases"][job["case"]]}
    addresses, environment = None, None
    for index, row in enumerate(rows):
        if (set(row) != set(FIELDS) or any(value is None for value in row.values()) or
                any(value == "" for field, value in row.items() if field != "total_host_us")):
            raise ValueError("incomplete or extra CSV values")
        for field in (*shared_audit.INT_FIELDS, "timing_pair"):
            row[field] = int(row[field])
        for field in ("event_us", "submit_us", "quartet_host_us"):
            row[field] = positive(row[field], field)
        if job["sync_mode"] == "quartet":
            if row["total_host_us"] != "" or row["position_host_scope"] != "unobserved":
                raise ValueError("quartet mode must leave per-position completion unobserved")
            row["total_host_us"] = None
        else:
            row["total_host_us"] = positive(row["total_host_us"], "total_host_us")
            if (row["position_host_scope"] != "timed_enqueue_through_snapshot_completion_per_operation" or
                    row["total_host_us"] + 1e-9 < row["submit_us"]):
                raise ValueError("invalid position host completion scope or interval")
        if any(row[field] != wanted for field, wanted in fixed.items()):
            raise ValueError("workload/protocol differs from frozen job")
        quartet, position = divmod(index, 4)
        pattern = patterns[quartet]
        slot = pattern[position]
        backend = slot_a if slot == "A" else slot_b
        pair = (0 if slot == "A" else 2) + pattern[:position].count(slot)
        if (row["quartet"], row["position"], row["pattern"], row["slot"], row["backend"], row["timing_pair"]) != (
                quartet, position, pattern, slot, backend, pair):
            raise ValueError("balanced execution order, backend, or unique timing pair differs")
        current_addresses = {}
        for field in shared_audit.ADDRESS_FIELDS:
            if not row[field].startswith("0x") or int(row[field], 16) <= 0:
                raise ValueError("missing valid address: " + field)
            current_addresses[field] = int(row[field], 16)
        if current_addresses["old_launch_address"] == current_addresses["new_launch_address"]:
            raise ValueError("old and new global backend functions have the same address")
        if addresses is not None and current_addresses != addresses:
            raise ValueError("backend/context/stream address changed within process")
        addresses = current_addresses
        current_environment = {field: row[field] for field in shared_audit.ENV_FIELDS}
        if not row["gpu_uuid"].startswith("GPU-") or row["driver"] <= 0 or row["runtime"] <= 0:
            raise ValueError("missing GPU/API identity")
        if environment is not None and current_environment != environment:
            raise ValueError("GPU/API identity changed within process")
        environment = current_environment
    for start in range(0, count, 4):
        group = rows[start:start + 4]
        if len({row["quartet_host_us"] for row in group}) != 1:
            raise ValueError("quartet host interval must be identical across its four rows")
        host = group[0]["quartet_host_us"]
        if host + 1e-6 < sum(row["submit_us"] for row in group) * job["batch"]:
            raise ValueError("quartet host interval is shorter than enclosed submissions")
        if job["sync_mode"] == "position" and host + 1e-6 < sum(row["total_host_us"] for row in group) * job["batch"]:
            raise ValueError("quartet host interval is shorter than enclosed position completions")
    return rows


def validate_job(output: Path, manifest: dict, job: dict) -> dict:
    stem = output / "measurements" / job["stem"]
    receipt_path = stem.with_suffix(".receipt.json")
    receipt = read_json(receipt_path)
    if set(receipt) != {"schema", "index", "manifest", "environment", "artifacts"} or receipt["schema"] != 1 or receipt["index"] != job["index"]:
        raise ValueError("malformed invocation receipt")
    shared_audit.validate_file(receipt["manifest"], output / "manifest.json")
    shared_audit.validate_file(receipt["environment"], output / "environment.json")
    paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
    if len(receipt["artifacts"]) != len(paths):
        raise ValueError("missing artifact hashes")
    for record, path in zip(receipt["artifacts"], paths, strict=True):
        shared_audit.validate_file(record, path)
    metadata = read_json(stem.with_suffix(".command.json"))
    binary = manifest["binary"]
    command = command_for(binary["path"], job)
    if job["command"] != command or metadata.get("command") != command:
        raise ValueError("exact command differs from frozen manifest")
    if metadata.get("exit_code") != 0 or metadata.get("executables_unchanged") is not True:
        raise ValueError("invocation failed or binary changed")
    positive(metadata["seconds"], "invocation seconds")
    for field in ("executable", "binary"):
        if metadata.get(field) != binary["path"] or metadata.get(field + "_sha256") != binary["sha256"]:
            raise ValueError("executable identity differs from frozen binary")
    environment = read_json(output / "environment.json")
    if (environment.get("schema") != 1 or environment.get("binary") != binary["path"] or
            environment.get("binary_sha256") != binary["sha256"] or
            not environment.get("session_started_utc") or not environment.get("uname")):
        raise ValueError("campaign environment does not identify frozen binary")
    if len(environment["gpus"]) != 1:
        raise ValueError("screen requires exactly one recorded GPU")
    if metadata.get("gpus_before") != environment["gpus"] or metadata.get("gpus_after") != environment["gpus"]:
        raise ValueError("GPU/driver identity changed")
    if not metadata.get("telemetry_before") or not metadata.get("telemetry_after"):
        raise ValueError("missing before/after telemetry")
    rows = parse_rows(stem.with_suffix(".csv"), job, manifest)
    gpu = environment["gpus"][0]
    if rows[0]["gpu"] != gpu["name"] or not gpu.get("uuid") or rows[0]["gpu_uuid"].lower() != gpu["uuid"].lower():
        raise ValueError("CSV GPU differs from recorded environment")
    return {"rows": rows, "receipt": file_record(receipt_path)}


def summarize_job(job: dict, rows: list[dict]) -> dict:
    clusters = []
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        slots = {slot: shared_audit.geometric_mean([row["event_us"] for row in group if row["slot"] == slot])
                 for slot in ("A", "B")}
        clusters.append({"quartet": group[0]["quartet"], "pattern": group[0]["pattern"],
                         "event_b_over_a": slots["B"] / slots["A"], "quartet_host_us": group[0]["quartet_host_us"]})
    return {**{field: job[field] for field in (*GROUP_FIELDS, "index", "stem", "repetition", "data_seed", "order_seed")},
            "positions": len(rows), "clusters": clusters,
            "environment": {field: rows[0][field] for field in shared_audit.ENV_FIELDS},
            "addresses": {field: rows[0][field] for field in shared_audit.ADDRESS_FIELDS},
            "event_aa_ratios": shared_audit.describe([group["event_b_over_a"] for group in clusters]),
            "quartet_host_us": shared_audit.describe([group["quartet_host_us"] for group in clusters]),
            "position_durations": {field: shared_audit.describe([row[field] for row in rows]) for field in
                                   (("event_us", "submit_us", "total_host_us") if job["sync_mode"] == "position" else ("event_us", "submit_us"))}}


def analyze(output: Path) -> dict:
    manifest = read_json(output / "manifest.json")
    wanted = build_manifest(Path(manifest["binary"]["path"]), Path(manifest["source_provenance"]["path"]))
    if manifest != wanted:
        raise ValueError("manifest, source provenance, executable, or protocol differs")
    expected_paths = {output / "measurements" / (job["stem"] + suffix) for job in manifest["jobs"]
                      for suffix in (".csv", ".log", ".command.json", ".receipt.json")}
    actual_paths = {path for path in (output / "measurements").rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("measurement artifacts differ from complete frozen schedule")
    processes, receipts = [], []
    for job in manifest["jobs"]:
        checked = validate_job(output, manifest, job)
        processes.append(summarize_job(job, checked["rows"]))
        receipts.append(checked["receipt"])
    if any(process["environment"] != processes[0]["environment"] for process in processes):
        raise ValueError("GPU/API identity changed across processes")
    grouped = defaultdict(list)
    for process in processes:
        grouped[tuple(process[field] for field in GROUP_FIELDS)].append(process)
    summaries = []
    for key, selected in sorted(grouped.items()):
        if len(selected) != len(REPETITIONS) or {p["repetition"] for p in selected} != set(REPETITIONS):
            raise ValueError("wrong number of process repetitions in stratum")
        summaries.append({**dict(zip(GROUP_FIELDS, key)), "processes": len(selected),
                          "process_event_aa_ratios": shared_audit.describe([p["event_aa_ratios"]["geometric_mean"] for p in selected]),
                          "process_event_us": shared_audit.describe([p["position_durations"]["event_us"]["geometric_mean"] for p in selected]),
                          "process_quartet_host_us": shared_audit.describe([p["quartet_host_us"]["geometric_mean"] for p in selected]),
                          "process_max_abs_quartet_log_ratio": [max(abs(math.log(c["event_b_over_a"])) for c in p["clusters"]) for p in selected]})
    return {"schema": 1, "kind": "audited_matched_synchronization_aa_screen", "complete": True,
            "manifest": file_record(output / "manifest.json"), "binary": manifest["binary"],
            "invocations": len(processes), "raw_positions": sum(p["positions"] for p in processes),
            "quartet_clusters": sum(len(p["clusters"]) for p in processes),
            "receipts": receipts, "processes": processes, "summaries": summaries,
            "metric_scopes": manifest["metric_scopes"], "limitations": manifest["limitations"],
            "production_default_promotion": False}


def publish(output: Path) -> dict:
    report = analyze(output)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Matched synchronization AA screen", "",
             f"Audited {report['invocations']} processes, {report['quartet_clusters']} quartets, and {report['raw_positions']} raw positions.", "",
             "Each row is a separate case/comparison/batch/mode stratum with two process repetitions. AA ratios are slot B / slot A of the same implementation. Values below are geometric means of process summaries; ranges are the two process values. These are descriptive screens, not confidence intervals.", "",
             "| Case | AA binding | Batch | Sync mode | AA event ratio; process range | Event us/op; process range | Host us/quartet; process range |",
             "|---|---|---:|---|---:|---:|---:|"]
    for row in report["summaries"]:
        stats = [row[field] for field in ("process_event_aa_ratios", "process_event_us", "process_quartet_host_us")]
        values = [f"{s['geometric_mean']:.6f}; {s['minimum']:.6f}–{s['maximum']:.6f}" for s in stats]
        lines.append("| " + " | ".join([row["case"], row["comparison"], str(row["batch"]), row["sync_mode"], *values]) + " |")
    lines += ["", "## Metric definitions", ""] + [f"- `{field}`: {scope}." for field, scope in report["metric_scopes"].items()]
    lines += ["", "## Limits", ""] + ["- " + item for item in report["limitations"]]
    lines += ["", f"Executable SHA-256: `{report['binary']['sha256']}`.", ""]
    (output / "analysis.md").write_text("\n".join(lines))
    print(f"Audited {report['invocations']} processes; {output / 'analysis.md'}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--provenance", type=Path, help="default: EXE_PARENT/sources/provenance.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    options = parser.parse_args(argv)
    exe, output = options.exe.resolve(), options.output_root.resolve()
    provenance = (options.provenance or exe.parent / "sources/provenance.json").resolve()
    if options.dry_run:
        jobs = schedule()
        for job in jobs:
            job["command"] = command_for(str(exe), job)
        print(json.dumps({"gpu_access": False, "invocations": len(jobs),
                          "raw_positions": len(jobs) * 4 * PROTOCOL["quartets"], "jobs": jobs}, indent=2))
        return 0
    if options.audit:
        publish(output)
        return 0
    manifest_path = output / "manifest.json"
    if not manifest_path.exists() and any((output / "measurements").glob("*")):
        raise ValueError("refusing to adopt existing measurements without frozen manifest")
    manifest = build_manifest(exe, provenance)
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("frozen manifest differs; use a new output directory")
    else:
        write_new(manifest_path, manifest)
    recorder = legacy.import_file("sync_preservation_recorder", legacy.RECORDER_PATH)
    recorder.EXE = exe
    recorder.ensure_environment(output)
    incomplete = False
    for job in manifest["jobs"]:
        stem = output / "measurements" / job["stem"]
        receipt = stem.with_suffix(".receipt.json")
        artifacts = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
        if receipt.exists():
            if incomplete:
                raise ValueError("completed jobs are not a prefix of frozen schedule")
            validate_job(output, manifest, job)
            continue
        incomplete = True
        if any(path.exists() for path in artifacts):
            raise ValueError("partial invocation retained; investigate and use a new output directory: " + str(stem))
        if build_manifest(exe, provenance) != manifest:
            raise ValueError("source or binary changed during campaign")
        print(f"[{job['index'] + 1}/{len(manifest['jobs'])}] {job['stem']}", flush=True)
        recorder.run(stem, job["command"])
        write_new(receipt, {"schema": 1, "index": job["index"], "manifest": file_record(manifest_path),
                            "environment": file_record(output / "environment.json"),
                            "artifacts": [file_record(path) for path in artifacts]})
        validate_job(output, manifest, job)
    publish(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
