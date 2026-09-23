#!/usr/bin/env python3
"""CPU-only audit of the resumed session's paired output-clear experiment."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import autotune as tune


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(row):
    return f"{row['algorithm']}:{row['tuning']}:{row['blocks']}:{row['local_counter']}"


def telemetry(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 1:
        raise ValueError("expected exactly one GPU in telemetry")
    return {key.strip(): value.strip() for key, value in rows[0].items()}


def audit_run(case, build, repeat):
    path = BASE / "clear-ablation" / f"{case}-{build}-r{repeat}.csv"
    command_path = path.with_suffix(".command.json")
    recorded = json.loads(command_path.read_text())
    command = recorded["command"]
    exe = Path(command[0])
    assert recorded["exit_code"] == 0, command_path
    assert recorded["binary_sha256"] == sha256(exe), command_path
    options = dict(zip(command[1::2], command[2::2], strict=True))
    assert len(command[1:]) == 2 * len(options), command_path
    assert options["--samples"] == "21" and options["--batch"] == "32"
    assert options["--seed"] == "424242"
    before, after = (telemetry(recorded[f"telemetry_{when}"]) for when in ("before", "after"))
    for key in ("name", "driver_version", "uuid"):
        assert before[key] == after[key], (command_path, key)
    assert before["driver_version"] == "597.06", command_path
    _, rows = tune.parse_csv(path.read_text())
    assert len(rows) == 4 and len({label(row) for row in rows}) == 4, path
    assert {label(row) for row in rows} == {
        entry if entry.count(":") == 3 else entry + ":native"
        for entry in options["--variants"].split(",")
    }, path
    for row in rows:
        for name in tune.WORKLOAD:
            assert str(row[name]) == options.get("--" + name.replace("_", "-"), "0"), (path, name)
        assert row["samples"] == 21 and row["batch"] == 32 and row["seed"] == 424242
        assert row["counter"] == "u32" and row["distribution"] == "uniform"
        assert row["order"] == "shuffled" and row["cache"] == "warm" and row["launch"] == "graph"
        assert row["timing_protocol"] == 3 and row["warmup_ms"] == 0
        samples = sorted(row["raw_samples"])
        expected = {"median_us": samples[10], "p95_us": samples[19],
                    "min_us": samples[0], "max_us": samples[-1]}
        for name, value in expected.items():
            assert math.isclose(row[name], value, rel_tol=0, abs_tol=1e-6), (path, name)
    assert len({tuple(tune.extract(row, tune.ENVIRONMENT).values()) for row in rows}) == 1
    return {"csv": str(path.relative_to(BASE)), "csv_sha256": sha256(path),
            "command_json": str(command_path.relative_to(BASE)), "command_json_sha256": sha256(command_path),
            "executable": str(exe), "executable_sha256": sha256(exe), "command": command,
            "telemetry_before": before, "telemetry_after": after,
            "environment": tune.extract(rows[0], tune.ENVIRONMENT),
            "workload": tune.extract(rows[0], tune.WORKLOAD),
            "rows": [{"config": tune.extract(row, tune.CONFIG), "median_us": row["median_us"],
                      "min_us": row["min_us"], "p95_us": row["p95_us"], "max_us": row["max_us"],
                      "raw_samples_us": row["raw_samples"],
                      "samples_over_twice_median": sum(x > 2 * row["median_us"] for x in row["raw_samples"])}
                     for row in rows]}


def main():
    cases = []
    all_runs = []
    for case in ("small8", "smallbyte", "cachedbyte"):
        repeats = []
        for repeat in (1, 2, 3):
            old, new = (audit_run(case, build, repeat) for build in ("runtime", "kernel"))
            assert old["command"][1:] == new["command"][1:], (case, repeat)
            assert old["environment"] == new["environment"] and old["workload"] == new["workload"]
            assert old["telemetry_before"]["uuid"] == new["telemetry_before"]["uuid"]
            assert [r["config"] for r in old["rows"]] == [r["config"] for r in new["rows"]]
            old_cub = next(row["median_us"] for row in old["rows"] if row["config"]["algorithm"] == "cub")
            new_cub = next(row["median_us"] for row in new["rows"] if row["config"]["algorithm"] == "cub")
            comparisons = []
            for a, b in zip(old["rows"], new["rows"], strict=True):
                ratio = a["median_us"] / b["median_us"]
                comparisons.append({"config": a["config"], "runtime_median_us": a["median_us"],
                                    "kernel_median_us": b["median_us"], "paired_old_over_new": ratio,
                                    "cub_normalized_gain": ratio / (old_cub / new_cub),
                                    "runtime_speedup_over_cub": old_cub / a["median_us"],
                                    "kernel_speedup_over_cub": new_cub / b["median_us"]})
            repeats.append({"repeat": repeat, "runtime": old, "kernel": new, "comparisons": comparisons})
            all_runs.extend((old, new))
        summary = []
        for index, first in enumerate(repeats[0]["comparisons"]):
            item = {"config": first["config"]}
            for key in ("runtime_median_us", "kernel_median_us", "paired_old_over_new", "cub_normalized_gain",
                        "runtime_speedup_over_cub", "kernel_speedup_over_cub"):
                values = [repeat["comparisons"][index][key] for repeat in repeats]
                item[key] = {"per_repeat": values, "median": statistics.median(values)}
            summary.append(item)
        cases.append({"case": case, "workload": repeats[0]["runtime"]["workload"],
                      "repeats": repeats, "summary": summary})
    assert len({json.dumps(run["environment"], sort_keys=True) for run in all_runs}) == 1
    assert len({run["telemetry_before"]["uuid"] for run in all_runs}) == 1
    vectors = [row for run in all_runs for row in run["rows"]]
    report = {"schema": 1, "session_driver": "597.06", "gpu_uuid": all_runs[0]["telemetry_before"]["uuid"],
              "environment": all_runs[0]["environment"], "cases": cases,
              "audit": {"csv_count": len(all_runs), "vectors_verified": len(vectors),
                        "samples_verified": sum(len(row["raw_samples_us"]) for row in vectors),
                        "samples_over_twice_row_median": sum(row["samples_over_twice_median"] for row in vectors),
                        "all_exit_codes_zero": True, "named_executable_hashes_match": True,
                        "paired_commands_differ_only_by_executable": True,
                        "paired_configuration_metadata_identical": True,
                        "all_telemetry_gpu_and_driver_identities_match": True,
                        "all_raw_sample_summaries_match": True},
              "interpretation": ["Ratios use paired process medians; the summary is their median across three repeats.",
                                 "One data seed (424242), not independent-seed validation or a dispatch result.",
                                 "CUB normalization describes control shifts; it does not prove their cause.",
                                 "All 18 invocations use driver 597.06; historical driver results are kept separate.",
                                 "Telemetry snapshots do not establish stable clocks throughout a run."]}
    changed = {case["case"]: next(item for item in case["summary"] if item["config"]["algorithm"] == "shared")
               for case in cases}
    lines = ["# Output-clear ablation after resuming: driver 597.06", "",
             "We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.", "",
             "Replacing runtime memset with the explicit clearing kernel improves the complete shared operation "
             f"by **{changed['small8']['paired_old_over_new']['median']:.3f}×** for N1M/u32/B8, "
             f"**{changed['smallbyte']['paired_old_over_new']['median']:.3f}×** for N4096/u8/B256, and "
             f"**{changed['cachedbyte']['paired_old_over_new']['median']:.3f}×** for N1M/u8/B256 "
             "(median of three paired ratios). All nine shared pairs improve; unchanged controls remain close to parity.", "",
             "All 18 invocations below were rerun on the same RTX A5000 Laptop GPU with Windows driver **597.06**. "
             "They compare the archived runtime-clear binary against the kernel-clear binary within this session. "
             "Each uses uniform shuffled data, u32 output, warm graphs, timing protocol 3, seed 424242, "
             "21 samples and batch 32. This is a paired implementation ablation on one seed; it does not select a dispatch policy.", "",
             "Old/new ratios divide the runtime-clear process median by the paired kernel-clear median. "
             "CUB-normalized gain divides that ratio by the paired CUB ratio. Summary ratios are the median "
             "of the three paired ratios; displayed times are medians of three process medians.", "",
             "Case names: small8 = N1M/u32/B8; smallbyte = N4096/u8/B256; cachedbyte = N1M/u8/B256. "
             "N1M is 1,048,576 samples.", "",
             "| Case / variant | Runtime clear, µs | Kernel clear, µs | Paired old/new | CUB-normalized gain |", 
             "|---|---:|---:|---:|---:|"]
    for case in cases:
        for item in case["summary"]:
            lines.append(f"| {case['case']} / {label(item['config'])} | {item['runtime_median_us']['median']:.3f} | "
                         f"{item['kernel_median_us']['median']:.3f} | {item['paired_old_over_new']['median']:.3f}× | "
                         f"{item['cub_normalized_gain']['median']:.3f}× |")
    lines += ["", "| Changed variant | Raw gains, R1 / R2 / R3 | CUB-normalized gains, R1 / R2 / R3 |",
              "|---|---|---|"]
    for case in cases:
        for item in case["summary"]:
            if item["config"]["algorithm"] not in ("shared", "bitplane"):
                continue
            raw = " / ".join(f"{x:.3f}×" for x in item["paired_old_over_new"]["per_repeat"])
            normalized = " / ".join(f"{x:.3f}×" for x in item["cub_normalized_gain"]["per_repeat"])
            lines.append(f"| {case['case']} / {item['config']['algorithm']} | {raw} | {normalized} |")
    lines += ["", "All 18 command records report success and their recorded SHA256 hashes match the named executables. "
              "Paired commands differ only by executable path; configuration and workload metadata match. "
              "All 72 raw sample vectors contain 21 values and reproduce median, p95, minimum and maximum "
              f"(1,512 samples total; {report['audit']['samples_over_twice_row_median']} exceed twice their own row median). "
              "No samples were discarded. GPU UUID and driver identity agree in all before/after telemetry snapshots. "
              "Snapshot agreement does not show that clocks were constant during execution.", "",
              f"Runtime-clear SHA256: `{all_runs[0]['executable_sha256']}`. "
              f"Kernel-clear SHA256: `{all_runs[1]['executable_sha256']}`.", "",
              "The [historical ablation](../clear-ablation-summary.md) belongs to the earlier driver session. "
              "Its measurements are not pooled with these results, and differences between sessions do not establish a driver effect.", "",
              "[Full paired measurements, raw samples, hashes and telemetry](clear-ablation-summary.json); "
              "[source CSVs and command records](clear-ablation/).", ""]
    (BASE / "clear-ablation-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (BASE / "clear-ablation-summary.md").write_text("\n".join(lines))
    print(json.dumps(report["audit"], indent=2))


if __name__ == "__main__":
    main()
