#!/usr/bin/env python3
"""Audit the frozen 24-process stream ablation without loading mutable tuner code."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
CASES = ("small8", "smallbyte", "cachedbyte", "large4096")
METRICS = {"median_us", "min_us", "p95_us", "max_us", "input_gb_s", "sample_us"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(row):
    return ":".join(row[name] for name in ("algorithm", "tuning", "blocks", "local_counter"))


def main():
    environment = json.loads((BASE / "environment.json").read_text())
    archives = {"runtime": ROOT / "build/profiled-loads/histogram_bench",
                "kernel": ROOT / "build/profiled/histogram_bench"}
    hashes = {name: digest(path) for name, path in archives.items()}
    cases, row_count, sample_count, excursion_count = [], 0, 0, 0
    for case in CASES:
        repeats = []
        for repeat in (1, 2, 3):
            builds = {}
            for build in ("runtime", "kernel"):
                path = BASE / "stream-ablation" / f"{case}-{build}-r{repeat}.csv"
                command_path = path.with_suffix(".command.json")
                record = json.loads(command_path.read_text())
                assert record["exit_code"] == 0 and record["executables_unchanged"]
                assert record["binary_sha256"] == record["executable_sha256"] == hashes[build]
                assert record["gpus_before"] == record["gpus_after"] == environment["gpus"]
                command = record["command"]
                options = dict(zip(command[1::2], command[2::2], strict=True))
                assert len(command[1:]) == 2 * len(options)
                assert options["--launch"] == "stream" and options["--cache"] == "warm"
                assert options["--samples"] == "21" and options["--batch"] == "32"
                assert options["--seed"] == "424242"
                rows = list(csv.DictReader(path.open()))
                by_label = {label(row): row for row in rows}
                expected = {entry if entry.count(":") == 3 else entry + ":native"
                            for entry in options["--variants"].split(",")}
                assert len(by_label) == len(rows) and set(by_label) == expected
                for row in rows:
                    for field in ("n", "bins", "input", "counter", "distribution", "order",
                                  "cache", "launch", "samples", "batch", "seed"):
                        assert row[field] == options["--" + field]
                    assert row["timing_protocol"] == "3" and row["warmup_ms"] == "0"
                    samples = sorted(map(float, row["sample_us"].split(";")))
                    assert len(samples) == 21
                    for field, value in (("median_us", samples[10]), ("p95_us", samples[19]),
                                         ("min_us", samples[0]), ("max_us", samples[-1])):
                        assert math.isclose(float(row[field]), value, rel_tol=0, abs_tol=1e-6)
                    row_count += 1
                    sample_count += len(samples)
                    excursion_count += sum(x > 2 * float(row["median_us"]) for x in samples)
                builds[build] = {"command": command, "csv": str(path.relative_to(BASE)),
                                 "csv_sha256": digest(path), "command_sha256": digest(command_path),
                                 "binary_sha256": hashes[build], "rows": by_label,
                                 "telemetry_before": record["telemetry_before"],
                                 "telemetry_after": record["telemetry_after"]}
            old, new = builds["runtime"], builds["kernel"]
            assert old["command"][1:] == new["command"][1:]
            assert set(old["rows"]) == set(new["rows"])
            cub_label = next(key for key in old["rows"] if key.startswith("cub:"))
            cub_ratio = float(old["rows"][cub_label]["median_us"]) / float(new["rows"][cub_label]["median_us"])
            comparisons = []
            for key, a in old["rows"].items():
                b = new["rows"][key]
                assert {k: v for k, v in a.items() if k not in METRICS} == {
                    k: v for k, v in b.items() if k not in METRICS}
                ratio = float(a["median_us"]) / float(b["median_us"])
                comparisons.append({"variant": key, "runtime_us": float(a["median_us"]),
                                    "kernel_us": float(b["median_us"]), "runtime_over_kernel": ratio,
                                    "cub_normalized_ratio": ratio / cub_ratio})
            repeats.append({"repeat": repeat, "builds": builds, "comparisons": comparisons})
        summaries = []
        for index, first in enumerate(repeats[0]["comparisons"]):
            summary = {"variant": first["variant"]}
            for field in ("runtime_us", "kernel_us", "runtime_over_kernel", "cub_normalized_ratio"):
                values = [r["comparisons"][index][field] for r in repeats]
                summary[field] = {"values": values, "minimum": min(values), "maximum": max(values),
                                  "median": statistics.median(values)}
            summaries.append(summary)
        cases.append({"case": case, "repeats": repeats, "summary": summaries})
    report = {"schema": 1, "environment": environment,
              "archives": {name: {"path": str(path), "sha256": hashes[name]} for name, path in archives.items()},
              "audit": {"invocations": 24, "measurement_rows": row_count, "samples": sample_count,
                        "samples_over_twice_row_median": excursion_count,
                        "paired_commands_match_except_executable": True, "paired_metadata_matches": True,
                        "recorded_hashes_match_archives": True, "driver_gpu_identity_matches": True,
                        "all_raw_sample_summaries_match": True}, "cases": cases}
    lines = ["# Output-clear ablation with direct stream launches", "",
             "We compare against NVIDIA’s CUB library of optimized GPU routines; its histogram is our reference implementation.", "",
             "The unconditional clearing-kernel change regresses the small8 stream case. "
             "This prompted an explicit output-clearing policy, retaining runtime clearing as the public API default. "
             "The graph-mode gains must not be generalized to direct stream launches.", "",
             "All 24 processes use driver 597.06, seed 424242, warm stream mode, 21 samples, and batch 32. "
             "The archived binaries differ in output clearing; paired commands otherwise match. "
             "Rows below are medians of three process medians, while ratios are medians of paired ratios. "
             "Values above 1 favor kernel clearing. Normalization to the NVIDIA histogram reference describes concurrent control shifts; "
             "it does not identify their cause.", "",
             "| Case / variant | Runtime clear, µs | Kernel clear, µs | Runtime/kernel | Reference-normalized ratio | Normalized range |",
             "|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        for item in case["summary"]:
            ratio = item["cub_normalized_ratio"]
            lines.append(f"| {case['case']} / {item['variant']} | {item['runtime_us']['median']:.3f} | "
                         f"{item['kernel_us']['median']:.3f} | {item['runtime_over_kernel']['median']:.3f}× | "
                         f"{ratio['median']:.3f}× | {ratio['minimum']:.3f}–{ratio['maximum']:.3f}× |")
    lines += ["", "The small8 shared configuration has a normalized ratio below 1 in all three pairs. "
              "The byte cases show substantial process-to-process variation, including shifts in unchanged controls; "
              "the large4096 shared case is close to parity. Stream timings include host submission gaps, "
              "so these numbers characterize complete API operation behavior rather than isolated device kernels.", "",
              f"Verified 24 successful command records, {row_count} CSV rows, and {sample_count} raw samples; "
              f"{excursion_count} samples exceed twice their own row median. No samples are discarded. "
              "Recorded executable hashes match preserved archives, even if the working build path later changes. "
              "GPU/driver identities match the session environment; snapshots do not establish constant clocks.", "",
              "[Full audit](stream-ablation-summary.json); [raw CSVs and command records](stream-ablation/).", ""]
    (BASE / "stream-ablation-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (BASE / "stream-ablation-summary.md").write_text("\n".join(lines))
    print(f"Verified {row_count} rows and {sample_count} samples across 24 stream invocations.")


if __name__ == "__main__":
    main()
