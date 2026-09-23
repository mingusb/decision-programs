#!/usr/bin/env python3
"""Audit measured two-node histogram graphs in exported Nsight Systems SQLite.

CPU-only; SQLite inputs are opened read-only. Selection intentionally matches the
protocol-3 benchmark: each candidate NVTX range encloses warmup then measurement.
This is a trace-specific audit, not a general Nsight export reader.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


KERNEL_TAGS = (
    "DeviceHistogramInitKernel",
    "DeviceHistogramSweepKernel",
    "clear_histogram_output",
    "shared_histogram_loaded",
    "reduce_partials",
)


def summary(values):
    return {
        "count": len(values),
        "min_us": min(values),
        "median_us": statistics.median(values),
        "max_us": max(values),
    }


def analyze(path, batch):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        launches = list(db.execute("""
            SELECT r.*, s.value AS api
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id = r.nameId
            WHERE s.value LIKE 'cudaGraphLaunch%'
        """))
        activities = collections.defaultdict(list)
        for row in db.execute("""
            SELECT k.*, s.value AS kernel_name
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
        """):
            name = row["kernel_name"]
            name = next((tag for tag in KERNEL_TAGS if tag in name), name)
            activities[row["correlationId"]].append(
                (row["start"], row["end"], name))
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            activities[row["correlationId"]].append(
                (row["start"], row["end"], "memset" + str(row["bytes"])))

        graphs = []
        gaps = collections.defaultdict(list)
        durations = collections.defaultdict(list)
        selected = set()
        for nvtx in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
            calls = [call for call in launches
                     if call["globalTid"] == nvtx["globalTid"]
                     and nvtx["start"] <= call["start"]
                     and call["end"] <= nvtx["end"]]
            if not calls:
                continue
            if len(calls) != 2:
                raise ValueError(f"Expected warmup plus measured call: {dict(nvtx)}")
            measured = max(calls, key=lambda call: call["start"])
            correlation_id = measured["correlationId"]
            if correlation_id in selected:
                raise ValueError(f"Nested/duplicate measured range: {correlation_id}")
            selected.add(correlation_id)
            events = sorted(activities[correlation_id])
            if len(events) != 2 * batch:
                raise ValueError(f"Graph {correlation_id}: expected {2 * batch} "
                                 f"activities, found {len(events)}")
            family = nvtx["text"].split(":")[0]
            span_ns = events[-1][1] - events[0][0]
            active_ns = sum(end - start for start, end, _ in events)
            graph_gaps = []
            for before, after in zip(events, events[1:]):
                gap_ns = after[0] - before[1]
                if gap_ns < 0:
                    raise ValueError(f"Overlapping graph activities: {correlation_id}")
                graph_gaps.append(gap_ns)
                gaps[(family, before[2], after[2])].append(gap_ns / 1000)
            if sum(graph_gaps) != span_ns - active_ns:
                raise ValueError(f"Graph accounting mismatch: {correlation_id}")
            graphs.append({
                "family": family,
                "nvtx_range": nvtx["text"],
                "correlation_id": correlation_id,
                "activities": len(events),
                "span_us": span_ns / 1000,
                "active_us": active_ns / 1000,
                "gap_us": sum(graph_gaps) / 1000,
            })
            for start, end, name in events:
                durations[(family, name)].append((end - start) / 1000)
        if not graphs:
            raise ValueError("No measured graphs matched")
        graph_summaries = {}
        for family in sorted({row["family"] for row in graphs}):
            rows = [row for row in graphs if row["family"] == family]
            graph_summaries[family] = {
                "correlation_ids": [row["correlation_id"] for row in rows],
                **{field: summary([row[field] for row in rows])
                   for field in ("span_us", "active_us", "gap_us")},
            }
        event_counts = db.execute("""
            SELECT COUNT(*), SUM(CASE WHEN timestamp != 0 THEN 1 ELSE 0 END)
            FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT
        """).fetchone()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "batch": batch,
            "selection": "Later of two same-thread cudaGraphLaunch calls inside each NVTX range",
            "units": "microseconds, except count/correlation IDs",
            "graphs": graphs,
            "graph_summaries": graph_summaries,
            "transitions": [
                {"family": family, "from": before, "to": after, **summary(values)}
                for (family, before, after), values in sorted(gaps.items())
            ],
            "activities": [
                {"family": family, "name": name, **summary(values)}
                for (family, name), values in sorted(durations.items())
            ],
            "cuda_event_records": event_counts[0],
            "nonzero_device_event_timestamps": event_counts[1],
            "memset_table_columns": [row["name"] for row in db.execute(
                "PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMSET)")],
        }
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, nargs="+")
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    print(json.dumps({"schema": 1, "traces": [analyze(path, args.batch)
                                             for path in args.sqlite]}, indent=2))


if __name__ == "__main__":
    main()
