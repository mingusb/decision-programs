#!/usr/bin/env python3
"""Compare matched timing ladders across benchmark protocols, using CPU files only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
FILENAME = re.compile(r"(?P<case>.+)-b(?P<batch>\d+)-s(?P<seed>\d+)\.csv")
CONFIG = ("algorithm", "tuning", "threads", "items", "replicas", "blocks", "local_counter")
CONTEXT = ("gpu", "sm", "driver_api", "runtime", "cub_version", "n", "bins", "input",
           "counter", "distribution", "order", "cache", "launch", "seed", "samples",
           "batch", "eviction_bytes")
INTEGERS = {"sm", "driver_api", "runtime", "cub_version", "n", "bins", "seed", "samples",
            "batch", "eviction_bytes", "tuning", "threads", "items", "replicas", "blocks"}


def load_ladder(directory: Path) -> dict:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"no ladder CSV files in {directory}")
    ladder = {}
    for path in files:
        match = FILENAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"unexpected ladder filename: {path.name}")
        with path.open(newline="", encoding="utf-8") as source:
            records = list(csv.DictReader(source))
        if not records:
            raise ValueError(f"empty CSV: {path}")
        entries = {}
        for raw in records:
            needed = CONFIG + CONTEXT + ("median_us", "sample_us")
            missing = [key for key in needed if raw.get(key) is None]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            values = {key: int(raw[key]) if key in INTEGERS else raw[key]
                      for key in CONFIG + CONTEXT}
            if (values["seed"] != int(match["seed"])
                    or values["batch"] != int(match["batch"])):
                raise ValueError(f"{path}: filename does not match row seed/batch")
            if not 0 <= values["tuning"] <= 5:
                raise ValueError(f"{path}: ladder must contain preserved tuning0..5 controls")
            samples = [float(value) for value in raw["sample_us"].split(";")]
            median = float(raw["median_us"])
            if (values["samples"] != len(samples) or not samples
                    or any(not math.isfinite(x) or x <= 0 for x in samples + [median])):
                raise ValueError(f"{path}: invalid or incomplete positive timing samples")
            ordered = sorted(samples)
            measured_median = ordered[math.ceil(0.5 * len(ordered)) - 1]
            if not math.isclose(median, measured_median, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{path}: median disagrees with raw samples")
            declared_protocol = raw.get("timing_protocol")
            key = tuple(values[name] for name in CONFIG)
            if key in entries:
                raise ValueError(f"{path}: duplicate configuration {key}")
            entries[key] = {
                "case": match["case"],
                "config": {name: values[name] for name in CONFIG},
                "context": {name: values[name] for name in CONTEXT},
                "declared_timing_protocol": int(declared_protocol) if declared_protocol else None,
                "warmup_ms": int(raw.get("warmup_ms", "0")),
                "load_policy": raw.get("load_policy"),
                "shared_limit": int(raw["shared_limit"]) if raw.get("shared_limit") else None,
                "median_us": median,
                "minimum_us": min(samples),
                "maximum_us": max(samples),
                "max_over_median": max(samples) / median,
                "samples_over_twice_median": sum(x > 2 * median for x in samples),
                "sample_count": len(samples),
            }
        ladder[path.name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": entries,
        }
    return ladder


def summarize(ladder: dict) -> dict:
    entries = [(filename, row) for filename, item in ladder.items() for row in item["rows"].values()]
    total_samples = sum(row["sample_count"] for _, row in entries)
    twice = sum(row["samples_over_twice_median"] for _, row in entries)
    worst_file, worst = max(entries, key=lambda pair: pair[1]["max_over_median"])
    return {
        "files": len(ladder), "rows": len(entries), "samples": total_samples,
        "samples_over_twice_median": twice,
        "fraction_over_twice_median": twice / total_samples,
        "declared_timing_protocols": sorted({row["declared_timing_protocol"] for _, row in entries},
                                             key=lambda value: -1 if value is None else value),
        "requested_warmup_ms": sorted({row["warmup_ms"] for _, row in entries}),
        "largest_max_over_median": {
            "file": worst_file, "config": worst["config"],
            "maximum_us": worst["maximum_us"], "median_us": worst["median_us"],
            "ratio": worst["max_over_median"],
        },
    }


def compare(before: dict, after: dict) -> dict:
    if set(before) != set(after):
        raise ValueError("ladder filenames differ; missing after="
                         f"{sorted(set(before) - set(after))}, extra after={sorted(set(after) - set(before))}")
    pairs = []
    grouped = defaultdict(list)
    for filename in sorted(before):
        left, right = before[filename], after[filename]
        if set(left["rows"]) != set(right["rows"]):
            raise ValueError(f"{filename}: candidate configurations differ across protocols")
        for key in sorted(left["rows"]):
            a, b = left["rows"][key], right["rows"][key]
            if a["context"] != b["context"]:
                differences = [name for name in CONTEXT if a["context"][name] != b["context"][name]]
                raise ValueError(f"{filename}: unmatched context fields {differences}")
            if b["declared_timing_protocol"] != 3:
                raise ValueError(f"{filename}: after data must declare timing_protocol=3")
            pair = {
                "file": filename, "case": a["case"], "config": a["config"],
                "context": a["context"], "before": a, "after": b,
                "before_over_after_timing_ratio": a["median_us"] / b["median_us"],
                "after_time_change_percent": (b["median_us"] / a["median_us"] - 1) * 100,
            }
            pairs.append(pair)
            grouped[(a["case"], key, a["context"]["batch"])].append(pair)
    groups = []
    for (case, _, batch), members in sorted(grouped.items()):
        seeds = sorted(pair["context"]["seed"] for pair in members)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{case}/batch{batch}: duplicate seed in aggregate")
        group = {"case": case, "config": members[0]["config"], "batch": batch, "seeds": seeds}
        for protocol in ("before", "after"):
            medians = {str(pair["context"]["seed"]): pair[protocol]["median_us"] for pair in members}
            times = list(medians.values())
            group[protocol] = {
                "per_seed_median_us": medians,
                "median_of_seed_medians_us": statistics.median(times),
                "across_seed_max_over_min_median": max(times) / min(times),
                "largest_within_row_max_over_median": max(pair[protocol]["max_over_median"] for pair in members),
                "samples_over_twice_median": sum(pair[protocol]["samples_over_twice_median"] for pair in members),
            }
        groups.append(group)
    return {
        "schema": 1,
        "interpretation": "Matched preserved tuning0..5 controls across benchmark timing protocols; "
                          "before/after ratios are timing-protocol comparisons, not kernel speedup claims.",
        "caveats": [
            "Before CSV files do not declare a timing_protocol field; null preserves that absence.",
            "Per-row medians use the benchmark's nearest-rank convention; across-seed summaries use statistics.median.",
            "Across-seed spread compares different data and candidate schedules, not independent repeats of one fixed seed.",
            "Each seed has one invocation per batch/protocol; same-seed repeatability within a protocol is not established.",
            "Max/median and >2x counts describe variability and do not discard or relabel samples.",
            "Warmup settings are recorded separately; unequal settings add a conditioning change to the protocol comparison.",
            "Clock, power, cache and scheduling causes cannot be inferred from these CSV files alone.",
        ],
        "summary": {"before": summarize(before), "after": summarize(after)},
        "files": {protocol: [{"name": name, "path": item["path"], "sha256": item["sha256"]}
                              for name, item in ladder.items()]
                  for protocol, ladder in (("before", before), ("after", after))},
        "groups": groups,
        "matched_rows": pairs,
    }


def candidate_label(config: dict) -> str:
    return f"{config['algorithm']}:{config['tuning']}:{config['blocks']}:{config['local_counter']}"


def markdown(report: dict) -> str:
    lines = [
        "Comparison of the preserved external-event graph timing protocol with protocol3's embedded graph events. "
        "These are matched tuning0–5 controls. Changes below describe measured timing and variability; "
        "they are not kernel-improvement claims.", "",
    ]
    cases = {}
    for pair in report["matched_rows"]:
        cases.setdefault(pair["case"], pair["context"])
    for case, context in sorted(cases.items()):
        lines.append(f"- {case}: N={context['n']}, B={context['bins']}, {context['input']} input, "
                     f"{context['counter']} output, {context['distribution']}/{context['order']}, "
                     f"{context['cache']} cache, {context['launch']} launches.")
    lines += ["", "| Data | Rows / samples | Samples >2× own row median | Largest within-row max/median | Warmup ms |",
              "|---|---:|---:|---:|---:|"]
    for protocol, summary in report["summary"].items():
        lines.append(f"| {protocol} | {summary['rows']} / {summary['samples']} | "
                     f"{summary['samples_over_twice_median']} ({100 * summary['fraction_over_twice_median']:.2f}%) | "
                     f"{summary['largest_max_over_median']['ratio']:.2f}× | "
                     f"{', '.join(map(str, summary['requested_warmup_ms']))} |")
    batches = sorted({group["batch"] for group in report["groups"]})
    lines += ["", "Per-operation microseconds, **before → after**. Each number is the median of the "
              "seed-specific row medians; the JSON preserves every seed and raw-sample variability statistic.", "",
              "| Case / candidate | " + " | ".join(f"Batch {batch}" for batch in batches) + " |",
              "|---|" + "---:|" * len(batches)]
    table = defaultdict(dict)
    for group in report["groups"]:
        table[(group["case"], candidate_label(group["config"]))][group["batch"]] = group
    for (case, label), by_batch in sorted(table.items()):
        cells = []
        for batch in batches:
            group = by_batch.get(batch)
            cells.append("—" if group is None else
                         f"{group['before']['median_of_seed_medians_us']:.3f} → "
                         f"{group['after']['median_of_seed_medians_us']:.3f}")
        lines.append(f"| {case} / {label} | " + " | ".join(cells) + " |")
    lines += ["", "Largest across-seed spread of row medians (max/min):", "",
              "| Protocol | Case / candidate / batch | Seed medians (us) | Spread |",
              "|---|---|---|---:|"]
    for protocol in ("before", "after"):
        group = max(report["groups"], key=lambda item: item[protocol]["across_seed_max_over_min_median"])
        details = group[protocol]
        seeds = ", ".join(f"{seed}: {value:.3f}" for seed, value in sorted(details["per_seed_median_us"].items()))
        lines.append(f"| {protocol} | {group['case']} / {candidate_label(group['config'])} / {group['batch']} | "
                     f"{seeds} | {details['across_seed_max_over_min_median']:.2f}× |")
    lines += ["", "Each seed has one invocation per batch and protocol, so this spread is not same-seed "
              "repeatability. Data and candidate order both vary with seed. Timing-protocol changes and any "
              "warmup change prevent attributing before/after ratios to kernels. No samples are removed. "
              "Clock, power, cache and scheduling causes require separate evidence.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=BASE / "timing-before")
    parser.add_argument("--after", type=Path, default=BASE / "timing-after")
    parser.add_argument("--output-prefix", type=Path, default=BASE / "timing-comparison")
    args = parser.parse_args()
    try:
        report = compare(load_ladder(args.before), load_ladder(args.after))
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        rendered = markdown(report)
        prefix = args.output_prefix.resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_name(prefix.name + ".json")
        markdown_path = prefix.with_name(prefix.name + ".md")
        json_path.write_text(encoded, encoding="utf-8")
        markdown_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {markdown_path} and {json_path}; {len(report['matched_rows'])} matched rows.")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
