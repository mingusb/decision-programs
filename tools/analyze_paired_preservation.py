#!/usr/bin/env python3
"""Strict CPU-only audit of frozen same-process histogram comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_runner_for_audit", ROOT / "tools/run_paired_preservation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FIELDS = ["case", "comparison", "slot_a", "slot_b", "data_seed", "order_seed", "quartet", "pattern",
          "position", "slot", "backend", "n", "bins", "tuning", "blocks", "launch", "clear", "batch",
          "warmup_ms", "event_us", "submit_us", "total_host_us", "gpu", "gpu_uuid", "driver", "runtime",
          "old_launch_address", "new_launch_address", "context_address", "stream_address"]
INT_FIELDS = ("data_seed", "order_seed", "quartet", "position", "n", "bins", "tuning", "blocks",
              "batch", "warmup_ms", "driver", "runtime")
METRICS = ("event_us", "submit_us", "total_host_us")
ENV_FIELDS = ("gpu", "gpu_uuid", "driver", "runtime")
ADDRESS_FIELDS = ("old_launch_address", "new_launch_address", "context_address", "stream_address")


def fail(message: str) -> None:
    raise ValueError(message)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def describe(values: list[float]) -> dict:
    if not values:
        fail("cannot summarize an empty sample")
    return {"count": len(values), "geometric_mean": geometric_mean(values),
            "median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def validate_file(record: dict, expected: Path | None = None) -> None:
    if set(record) != {"path", "sha256"}:
        fail("invalid file record")
    path = Path(record["path"])
    if not path.is_absolute() or (expected is not None and path != expected.resolve()):
        fail("unexpected artifact path: " + str(path))
    if not path.is_file() or runner.sha256(path) != record["sha256"]:
        fail("missing or changed artifact: " + str(path))


def parse_rows(path: Path, job: dict, manifest: dict) -> list[dict]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            fail(f"{path}: CSV header differs from the declared schema")
        rows = list(reader)
    expected_count = manifest["protocol"]["quartets"] * 4
    if len(rows) != expected_count:
        fail(f"{path}: expected {expected_count} positions, found {len(rows)}")
    slot_a, slot_b = job["comparison"].split("-")
    patterns = runner.pattern_order(job["order_seed"], manifest["protocol"]["quartets"])
    if job["patterns"] != patterns:
        fail("frozen pattern sequence is not the independently reconstructed shuffle")
    fixed = {"case": job["case"], "comparison": job["comparison"], "slot_a": slot_a, "slot_b": slot_b,
             "data_seed": job["data_seed"], "order_seed": job["order_seed"],
             **manifest["cases"][job["case"]],
             **{field: manifest["protocol"][field] for field in ("batch", "warmup_ms")}}
    environment, addresses = None, None
    for index, row in enumerate(rows):
        if set(row) != set(FIELDS) or any(value is None or value == "" for value in row.values()):
            fail(f"{path}: incomplete or extra CSV values")
        for field in INT_FIELDS:
            row[field] = int(row[field])
        for field in METRICS:
            row[field] = float(row[field])
            if not math.isfinite(row[field]) or row[field] <= 0:
                fail(f"{path}: timing must be positive and finite: {field}")
        if row["total_host_us"] + 1e-9 < row["submit_us"]:
            fail(f"{path}: host total is shorter than its submission interval")
        if any(row[field] != value for field, value in fixed.items()):
            fail(f"{path}: row {index} workload/protocol differs from its frozen job")
        quartet, position = divmod(index, 4)
        pattern = patterns[quartet]
        slot = pattern[position]
        backend = slot_a if slot == "A" else slot_b
        if (row["quartet"], row["position"], row["pattern"], row["slot"], row["backend"]) != (
                quartet, position, pattern, slot, backend):
            fail(f"{path}: row {index} differs from the declared balanced execution order")
        current_addresses = {}
        for field in ADDRESS_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value.startswith("0x") or int(value, 16) <= 0:
                fail(f"{path}: missing nonzero hexadecimal address: {field}")
            current_addresses[field] = int(value, 16)
        if current_addresses["old_launch_address"] == current_addresses["new_launch_address"]:
            fail(f"{path}: old and new global backend functions have the same address")
        if addresses is not None and addresses != current_addresses:
            fail(f"{path}: backend/context/stream address changed within a process")
        addresses = current_addresses
        bindings = {"A": addresses[slot_a + "_launch_address"], "B": addresses[slot_b + "_launch_address"]}
        if (bindings["A"] == bindings["B"]) != (slot_a == slot_b):
            fail(f"{path}: backend slot bindings disagree with comparison kind")
        if not row["gpu_uuid"].startswith("GPU-") or row["driver"] <= 0 or row["runtime"] <= 0:
            fail(f"{path}: missing GPU/API identity")
        current_environment = {field: row[field] for field in ENV_FIELDS}
        if environment is not None and environment != current_environment:
            fail(f"{path}: GPU/API identity changed within a process")
        environment = current_environment
    if Counter(row["pattern"] for row in rows) != {"ABBA": expected_count // 2, "BAAB": expected_count // 2}:
        fail(f"{path}: quartet order is not balanced")
    return rows


def validate_job(output: Path, manifest: dict, job: dict) -> dict:
    stem = output / "measurements" / job["stem"]
    receipt_path = stem.with_suffix(".receipt.json")
    receipt = runner.read_json(receipt_path)
    if set(receipt) != {"schema", "index", "manifest", "environment", "artifacts"} or receipt["schema"] != 1 or receipt["index"] != job["index"]:
        fail(f"{receipt_path}: malformed invocation receipt")
    validate_file(receipt["manifest"], output / "manifest.json")
    validate_file(receipt["environment"], output / "environment.json")
    paths = [stem.with_suffix(suffix) for suffix in (".csv", ".log", ".command.json")]
    if len(receipt["artifacts"]) != len(paths):
        fail(f"{receipt_path}: missing artifact hashes")
    for record, path in zip(receipt["artifacts"], paths, strict=True):
        validate_file(record, path)
    metadata = runner.read_json(stem.with_suffix(".command.json"))
    binary = manifest["binary"]
    expected_command = runner.command_for(binary["path"], job)
    if job["command"] != expected_command or metadata.get("command") != expected_command:
        fail(f"{stem}: exact command differs from manifest")
    if metadata.get("exit_code") != 0 or metadata.get("executables_unchanged") is not True:
        fail(f"{stem}: invocation failed or binary changed")
    if not math.isfinite(metadata["seconds"]) or metadata["seconds"] <= 0:
        fail(f"{stem}: invalid invocation duration")
    for field in ("executable", "binary"):
        if metadata.get(field) != binary["path"] or metadata.get(field + "_sha256") != binary["sha256"]:
            fail(f"{stem}: executable identity differs from the frozen binary")
    environment = runner.read_json(output / "environment.json")
    if (environment.get("schema") != 1 or environment.get("binary") != binary["path"]
            or environment.get("binary_sha256") != binary["sha256"]
            or not environment.get("session_started_utc") or not environment.get("uname")):
        fail("campaign environment does not identify the frozen executable")
    if len(environment["gpus"]) != 1:
        fail("this bounded campaign requires exactly one recorded GPU")
    if metadata.get("gpus_before") != environment["gpus"] or metadata.get("gpus_after") != environment["gpus"]:
        fail(f"{stem}: GPU/driver identity changed")
    if not metadata.get("telemetry_before") or not metadata.get("telemetry_after"):
        fail(f"{stem}: missing before/after telemetry")
    rows = parse_rows(stem.with_suffix(".csv"), job, manifest)
    gpu = environment["gpus"][0]
    if (rows[0]["gpu"] != gpu["name"] or not gpu.get("uuid")
            or rows[0]["gpu_uuid"].lower() != gpu["uuid"].lower()):
        fail(f"{stem}: CSV GPU differs from recorded process environment")
    return {"rows": rows, "receipt": runner.file_record(receipt_path)}


def summarize_job(job: dict, rows: list[dict]) -> dict:
    real = job["comparison"] in ("old-new", "new-old")
    clusters = []
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        cluster = {"quartet": group[0]["quartet"], "pattern": group[0]["pattern"], "metrics": {}}
        for metric in METRICS:
            slots = {slot: geometric_mean([row[metric] for row in group if row["slot"] == slot]) for slot in ("A", "B")}
            slot_ratio = slots["B"] / slots["A"]
            normalized = 1 / slot_ratio if job["comparison"] == "new-old" else slot_ratio
            cluster["metrics"][metric] = {"slot_a_us": slots["A"], "slot_b_us": slots["B"],
                                            "slot_b_over_a": slot_ratio, "reported_ratio": normalized}
        clusters.append(cluster)
    return {**{field: job[field] for field in ("index", "stem", "case", "comparison", "data_seed", "order_seed")},
            "ratio_kind": "new/old" if real else "slot B/slot A",
            "environment": {field: rows[0][field] for field in ENV_FIELDS},
            "addresses": {field: rows[0][field] for field in ADDRESS_FIELDS},
            "positions": len(rows), "clusters": clusters,
            "metrics": {metric: {
                "quartet_ratios": describe([cluster["metrics"][metric]["reported_ratio"] for cluster in clusters]),
                "by_pattern": {pattern: describe([cluster["metrics"][metric]["reported_ratio"] for cluster in clusters
                                                   if cluster["pattern"] == pattern]) for pattern in ("ABBA", "BAAB")},
                "slot_a_us": describe([row[metric] for row in rows if row["slot"] == "A"]),
                "slot_b_us": describe([row[metric] for row in rows if row["slot"] == "B"]),
            } for metric in METRICS}}


def analyze(output: Path) -> dict:
    output = output.resolve()
    manifest = runner.read_json(output / "manifest.json")
    expected = runner.build_manifest(Path(manifest["binary"]["path"]), Path(manifest["source_provenance"]["path"]))
    if manifest != expected:
        fail("manifest, source provenance, or executable differs from the declared frozen campaign")
    expected_paths = {output / "measurements" / (job["stem"] + suffix) for job in manifest["jobs"]
                      for suffix in (".csv", ".log", ".command.json", ".receipt.json")}
    actual_paths = {path for path in (output / "measurements").rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        fail("measurement artifact set differs from the complete schedule; missing=" +
             str(sorted(map(str, expected_paths - actual_paths))) + "; undeclared=" +
             str(sorted(map(str, actual_paths - expected_paths))))
    processes, receipts = [], []
    environment = None
    for job in manifest["jobs"]:
        audited = validate_job(output, manifest, job)
        process = summarize_job(job, audited["rows"])
        if environment is not None and process["environment"] != environment:
            fail("CSV GPU/runtime/driver identity changed across processes")
        environment = process["environment"]
        processes.append(process)
        receipts.append(audited["receipt"])
    summaries = []
    for case in runner.CASES:
        for comparison in ("real", "old-old", "new-new"):
            selected = [process for process in processes if process["case"] == case
                        and (process["comparison"] in ("old-new", "new-old") if comparison == "real"
                             else process["comparison"] == comparison)]
            summaries.append({"case": case, "comparison": comparison,
                              "ratio_kind": "new/old" if comparison == "real" else "slot B/slot A",
                              "processes": len(selected),
                              "metrics": {metric: {
                                  "process_geometric_mean_ratios": describe([process["metrics"][metric]["quartet_ratios"]["geometric_mean"] for process in selected]),
                                  "process_median_ratios": describe([process["metrics"][metric]["quartet_ratios"]["median"] for process in selected]),
                                  "by_order_seed": {str(seed): describe([process["metrics"][metric]["quartet_ratios"]["geometric_mean"] for process in selected if process["order_seed"] == seed]) for seed in runner.ORDER_SEEDS},
                              } for metric in METRICS}})
    return {"schema": 1, "kind": "audited_same_process_paired_preservation", "complete": True,
            "manifest": runner.file_record(output / "manifest.json"), "binary": manifest["binary"],
            "environment": environment, "receipts": receipts, "invocations": len(processes),
            "raw_positions": sum(process["positions"] for process in processes),
            "quartet_clusters": sum(len(process["clusters"]) for process in processes),
            "production_default_promotion": False, "summaries": summaries, "processes": processes,
            "interpretation": manifest["interpretation"], "limitations": manifest["limitations"] + [
                "The single-valued case generates identical data for both data seeds; those runs are process replicates, not independent input distributions.",
                "Position timings within a quartet are dependent. Quartet ratios, process summaries, and AA controls are retained separately; positions are not treated as independent trials.",
                "AA control ranges provide descriptive noise context. Overlap or lack of overlap alone does not establish equivalence, significance, or a zero-loss guarantee.",
            ]}


def markdown(report: dict) -> str:
    lines = ["# Same-process preservation comparison", "",
             f"Audited {report['invocations']} invocations, {report['quartet_clusters']} quartet clusters, and {report['raw_positions']} raw timing positions.", "",
             "Real-pair ratios are **new / old**; above 1 means the new backend took longer. AA controls compare slot B / slot A of the same backend. Each table entry summarizes four separate processes. These are descriptive measurements, not equivalence or significance tests.", "",
             "| Case | Comparison | Event ratio: geometric mean; process range | Submission ratio: geometric mean; process range | Total host ratio: geometric mean; process range |",
             "|---|---|---:|---:|---:|"]
    for summary in report["summaries"]:
        values = []
        for metric in METRICS:
            stats = summary["metrics"][metric]["process_geometric_mean_ratios"]
            values.append(f"{stats['geometric_mean']:.6f}; {stats['minimum']:.6f}–{stats['maximum']:.6f}")
        lines.append("| " + " | ".join([summary["case"], summary["comparison"], *values]) + " |")
    lines += ["", "## Every process", "",
              "Ratios below summarize the 32 quartet clusters in each invocation. Reversed real-pair slot mappings are normalized to new / old. The JSON report retains every quartet ratio, ABBA/BAAB summaries, and slot latency distributions.", "",
              "| Case | Comparison | Data seed | Order seed | Event geometric mean | Event median | Submission geometric mean | Total host geometric mean |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for process in sorted(report["processes"], key=lambda item: (item["case"], item["comparison"], item["data_seed"], item["order_seed"])):
        values = [process["metrics"][metric]["quartet_ratios"]["geometric_mean"] for metric in METRICS]
        event_median = process["metrics"]["event_us"]["quartet_ratios"]["median"]
        lines.append(f"| {process['case']} | {process['comparison']} | {process['data_seed']} | {process['order_seed']} | {values[0]:.6f} | {event_median:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines += ["", "## Limits", ""]
    lines += ["- " + item for item in report["limitations"]]
    lines += ["", f"Executable SHA-256: `{report['binary']['sha256']}`.", "",
              "The frozen manifest, per-invocation receipts, original CSV files, command metadata, telemetry, and source provenance remain beside this report. Automatic defaults were not changed.", ""]
    return "\n".join(lines)


def publish(output: Path) -> dict:
    report = analyze(output)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "analysis.md").write_text(markdown(report))
    print(f"Audited {report['invocations']} invocations, {report['quartet_clusters']} quartets, {report['raw_positions']} raw positions.")
    print(output / "analysis.md")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=runner.DEFAULT_OUTPUT)
    options = parser.parse_args(argv)
    publish(options.output_root.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
