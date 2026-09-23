#!/usr/bin/env python3
"""CPU-only correlation audit of the frozen four-report stream diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import statistics


BASE = Path(__file__).resolve().parent
LABEL = "shared:t10:grid48:local=native:clear=runtime:warm:stream"
BATCH, SAMPLES = 32, 21
FLAGS = ["--n", "1048576", "--bins", "4096", "--input", "u32", "--counter", "u32",
         "--distribution", "uniform", "--order", "shuffled", "--cache", "warm",
         "--launch", "stream", "--warmup-ms", "200", "--seed", "2026092211",
         "--variants", "shared:10:48:native:runtime", "--samples", "21", "--batch", "32"]
PROFILE = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
           "--cuda-event-trace=false", "--force-overwrite=false", "--export=sqlite", "--output"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    values = list(values)
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values))


def clipped(start, end, left, right):
    return max(0, min(end, right) - max(start, left))


def load_csv(path):
    lines = path.read_text().splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("gpu,sm,")]
    require(len(headers) == 1, f"{path}: expected exactly one benchmark CSV header")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[headers[0]:headers[0] + 2]))))
    require(len(rows) == 1, f"{path}: expected one benchmark row")
    row = rows[0]
    expected = dict(n="1048576", bins="4096", input="u32", counter="u32", algorithm="shared",
                    tuning="10", threads="512", items="8", replicas="1", blocks="48",
                    scratch_bytes="0", local_counter="native", clear_policy="runtime",
                    load_policy="vector4", shared_limit="49152", samples="21", batch="32",
                    distribution="uniform", order="shuffled", cache="warm", launch="stream",
                    warmup_ms="200", timing_protocol="3", seed="2026092211")
    require(all(row.get(k) == v for k, v in expected.items()), f"{path}: unexpected CSV configuration")
    values = [float(v) for v in row["sample_us"].split(";")]
    require(len(values) == SAMPLES and statistics.median(values) == float(row["median_us"]),
            f"{path}: CSV median/raw samples disagree")
    return row, values


def analyze_profile(directory, position, role, manifest, environment):
    stem = directory / f"p{position}-{role}"
    paths = {suffix: stem.with_suffix(suffix) for suffix in
             (".sqlite", ".nsys-rep", ".command.json", ".stdout", ".log")}
    require(all(p.is_file() for p in paths.values()), f"{stem}: incomplete artifacts")
    command = read_json(paths[".command.json"])
    binary = manifest["binaries"][role]
    expected = PROFILE + [str(stem), binary["path"]] + FLAGS
    require(command["command"] == expected, f"{stem}: unexpected command")
    require(command["exit_code"] == 0 and command["executables_unchanged"] is True,
            f"{stem}: profiler command failed or changed")
    require(command["executable_sha256"] == command["binary_sha256"], f"{stem}: profiler hash mismatch")
    require(command["gpus_before"] == command["gpus_after"] == environment["gpus"],
            f"{stem}: GPU identity changed")
    require(record(Path(binary["path"]))["sha256"] == binary["sha256"], f"{stem}: target binary changed")
    csv_row, samples = load_csv(paths[".stdout"])
    connection = sqlite3.connect(f"file:{paths['.sqlite']}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    runtime = [dict(r) for r in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")]
    for row in runtime:
        row["name"] = names[row["nameId"]]
    correlations = {}
    for kind in ("KERNEL", "MEMSET"):
        for raw in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_" + kind):
            row = dict(raw, kind=kind)
            # A single launched target process is required; no ambiguity is silently accepted.
            require(row["correlationId"] not in correlations, f"{stem}: duplicate GPU correlation")
            correlations[row["correlationId"]] = row
    nvtx = [dict(r) for r in connection.execute("SELECT * FROM NVTX_EVENTS ORDER BY start")
            if (r["text"] or names.get(r["textId"])) == LABEL]
    require(len(nvtx) == SAMPLES, f"{stem}: expected 21 NVTX sample ranges")
    event_timestamps = [r[0] for r in connection.execute("SELECT timestamp FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT")]
    require(all(t == 0 for t in event_timestamps), "unexpected device event timestamps: revisit window reconstruction")
    diagnostics = [dict(r) for r in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT")]
    rows = []
    kernel_names = set()
    for index, region in enumerate(nvtx):
        api = [r for r in runtime if r["globalTid"] == region["globalTid"]
               and r["start"] >= region["start"] and r["end"] <= region["end"]]
        events = [r for r in api if r["name"].startswith("cudaEventRecord_")]
        require(len(events) == 2, f"{stem}: range {index} lacks its event API pair")
        all_gpu = [correlations[r["correlationId"]] for r in api if r["correlationId"] in correlations]
        require(sum(r["kind"] == "KERNEL" for r in all_gpu) == 33 and
                sum(r["kind"] == "MEMSET" for r in all_gpu) == 33,
                f"{stem}: range {index} lacks 1 warmup + 32 operations")
        timed = [r for r in api if r["start"] >= events[0]["end"] and r["end"] <= events[1]["start"]]
        require(all(r["returnValue"] == 0 for r in timed), f"{stem}: failed timed API")
        memset_api = [r for r in timed if r["name"].startswith("cudaMemsetAsync_")]
        launch_api = [r for r in timed if r["name"].startswith("cudaLaunchKernel_")]
        name_api = [r for r in timed if r["name"] == "cuKernelGetName"]
        require(len(timed) == BATCH * 3 and len(memset_api) == len(launch_api) == len(name_api) == BATCH,
                f"{stem}: range {index} has unexpected timed APIs")
        require(all(a["end"] <= b["start"] for a, b in zip(timed, timed[1:])),
                f"{stem}: nested timed APIs need union accounting")
        by_correlation = {r["correlationId"]: r for r in timed}
        gpu = sorted([correlations[r["correlationId"]] for r in timed
                      if r["correlationId"] in correlations], key=lambda r: r["start"])
        require(len(gpu) == BATCH * 2, f"{stem}: missing timed GPU activity")
        require(len({(r["globalPid"], r["deviceId"], r["contextId"], r["streamId"]) for r in gpu}) == 1,
                f"{stem}: mixed process/device/context/stream")
        require(all(r["globalPid"] == (region["globalTid"] & ~0xffffff) for r in gpu),
                f"{stem}: runtime/GPU process mismatch")
        require(all(a["end"] <= b["start"] for a, b in zip(gpu, gpu[1:])),
                f"{stem}: overlapping GPU activities need union accounting")
        for i in range(BATCH):
            memset, kernel = gpu[2*i:2*i+2]
            require(memset["kind"] == "MEMSET" and kernel["kind"] == "KERNEL",
                    f"{stem}: nonalternating GPU operations")
            require(memset["bytes"] == 16384 and memset["value"] == 0,
                    f"{stem}: incorrect output clear")
            require(kernel["gridX"] == 48 and kernel["blockX"] == 512 and
                    kernel["registersPerThread"] == 32 and kernel["dynamicSharedMemory"] == 16384,
                    f"{stem}: unexpected kernel resources")
            kernel_names.add(names[kernel["demangledName"]])
        kernel_ns = [r["end"] - r["start"] for r in gpu if r["kind"] == "KERNEL"]
        memset_ns = [r["end"] - r["start"] for r in gpu if r["kind"] == "MEMSET"]
        gaps = []
        for previous, following in zip(gpu, gpu[1:]):
            left, right = previous["end"], following["start"]
            a = by_correlation[following["correlationId"]]
            before = max(0, min(right, a["start"]) - left)
            during = clipped(left, right, a["start"], a["end"])
            after = max(0, right - max(left, a["end"]))
            require(before + during + after == right - left, "GPU gap partition failed")
            gaps.append(dict(ns=right-left, next_kind=following["kind"], next_api_not_entered_ns=before,
                             next_api_active_ns=during, next_api_returned_ns=after))
        span = gpu[-1]["end"] - gpu[0]["start"]
        require(span == sum(kernel_ns) + sum(memset_ns) + sum(g["ns"] for g in gaps),
                "GPU activity span does not reconcile")
        metrics = {
            "profiled_event_us": samples[index],
            "kernel_us": sum(kernel_ns) / (BATCH * 1000),
            "memset_us": sum(memset_ns) / (BATCH * 1000),
            "gpu_activity_span_us": span / (BATCH * 1000),
            "gpu_gap_us": sum(g["ns"] for g in gaps) / (BATCH * 1000),
            "cuda_launch_api_us": sum(r["end"]-r["start"] for r in launch_api) / (BATCH * 1000),
            "cuda_memset_api_us": sum(r["end"]-r["start"] for r in memset_api) / (BATCH * 1000),
            "kernel_name_api_us": sum(r["end"]-r["start"] for r in name_api) / (BATCH * 1000),
            "all_observed_api_us": sum(r["end"]-r["start"] for r in timed) / (BATCH * 1000),
            "host_between_observed_apis_us": sum(b["start"]-a["end"] for a,b in zip(timed,timed[1:])) / (BATCH * 1000),
            "host_between_operations_us": sum(memset_api[i+1]["start"]-launch_api[i]["end"] for i in range(BATCH-1)) / (BATCH * 1000),
            "start_event_api_us_per_batch": (events[0]["end"]-events[0]["start"]) / 1000,
            "end_event_api_us_per_batch": (events[1]["end"]-events[1]["start"]) / 1000,
        }
        for part in ("not_entered", "active", "returned"):
            metrics[f"gpu_gap_next_api_{part}_us"] = sum(g[f"next_api_{part}_ns"] for g in gaps) / (BATCH * 1000)
        rows.append(dict(sample_index=index, nvtx_start_ns=region["start"], nvtx_end_ns=region["end"],
                         metrics=metrics, kernel_durations_ns=kernel_ns, memset_durations_ns=memset_ns,
                         gpu_gaps=gaps, runtime_api_correlations=[r["correlationId"] for r in timed]))
    require(len(kernel_names) == 1, f"{stem}: multiple counting kernels")
    connection.close()
    return dict(profile=stem.name, position=position, role=role, diagnostic_only=True,
                target_binary=binary, artifacts={s: record(p) for s,p in paths.items()},
                command_metadata=command, csv_metadata=csv_row, kernel=next(iter(kernel_names)),
                profiler_diagnostics=diagnostics,
                sample_ranges=rows, summary={k: describe(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=BASE / "preservation-timeline")
    directory = parser.parse_args().output_root.resolve()
    manifest, environment = read_json(directory / "manifest.json"), read_json(directory / "environment.json")
    require(manifest["order"] == ["old", "current", "current", "old"], "expected ABBA diagnostic")
    require(environment["binary_sha256"] == manifest["binaries"]["current"]["sha256"], "environment target mismatch")
    profiles = [analyze_profile(directory, i, role, manifest, environment)
                for i,role in enumerate(manifest["order"],1)]
    require(len({p["kernel"] for p in profiles}) == 1, "counting kernel identity differs")
    require(len({p["command_metadata"]["executable_sha256"] for p in profiles}) == 1, "profiler identity differs")
    limits = [
        "Diagnostic instrumentation changes runtime behavior; these traces do not replace unprofiled preservation evidence.",
        "Each NVTX sample contains one warmup plus 32 timed operations. Timed activities are selected by runtime correlation IDs between the two CPU event-record APIs; the warmup pair is excluded.",
        "Device event timestamps are zero. GPU activity span covers first timed memset start through last timed kernel end, not the exact CUDA event interval.",
        "GPU gaps mean absence of this target stream's recorded memset/kernel activity. They do not establish whole-device idleness or identify external GPU work.",
        "Gap partitions describe when the next operation's API is not entered, active, or returned. They do not assign causation to application instructions, operating-system scheduling, driver queues, or profiler overhead.",
        "Observed API durations include instrumentation and possible waiting; cuKernelGetName is reported separately. CPU context switches and instruction sampling were disabled.",
        "CPU API durations overlap GPU execution; they must not be added to GPU durations as one elapsed-time decomposition.",
        "Nsight warns that scheduling information is absent and not all NVTX events might have been collected. This audit nevertheless finds all 21 expected benchmark ranges and every expected operation within them; raw profiler diagnostics are retained.",
        "Per-profile table entries are medians of per-range values divided by 32; medians of component columns need not sum exactly.",
        "Telemetry surrounding the profiler includes startup and export time and cannot establish clocks during the target's measured ranges.",
    ]
    result = dict(schema=1, diagnostic_only=True, manifest=record(directory/"manifest.json"),
                  environment=record(directory/"environment.json"), analyzer=record(Path(__file__)),
                  profiles=profiles, counts=dict(profiles=4,nvtx_ranges=84,timed_kernels=2688,timed_memsets=2688), limitations=limits)
    (directory/"timeline-analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines = ["# Matched stream timeline diagnostic", "", "Diagnostic only; unprofiled preservation remains unresolved.", "",
             "All four reports passed command, identity, correlation, operation-count, kernel-resource, and gap-reconciliation checks. Each profile has 21 ranges of 32 timed operations; warmup operations are excluded.", "",
             "Values below are microseconds per operation, except profile names. Each cell is the median across 21 sample ranges.", "",
             "| Profile | Traced event | Kernel | Memset | GPU gaps | Launch API | Memset API | Between observed APIs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    columns = ("profiled_event_us","kernel_us","memset_us","gpu_gap_us","cuda_launch_api_us","cuda_memset_api_us","host_between_observed_apis_us")
    for profile in profiles:
        lines.append("| "+profile["profile"]+" | "+" | ".join(f"{profile['summary'][k]['median']:.3f}" for k in columns)+" |")
    lines += ["", "| Profile | GPU gap before next API enters | During next API | After next API returns |", "|---|---:|---:|---:|"]
    for p in profiles:
        lines.append("| "+p["profile"]+" | "+" | ".join(f"{p['summary'][f'gpu_gap_next_api_{part}_us']['median']:.3f}" for part in ("not_entered","active","returned"))+" |")
    lines += ["", "The first old profile is the slowest and the second old profile is the fastest. Counting-kernel and device-memset durations are close across binaries; the large traced differences occur in submission/API time and gaps between GPU activities. Most gap time occurs after the next operation's API has already returned, so it cannot all be labeled late host submission. This localizes the traced variability but does not establish the cause of the unprofiled regression.", "", "## Limits", ""]
    lines += ["- "+v for v in limits]
    (directory/"timeline-analysis.md").write_text("\n".join(lines)+"\n")
    with (directory/"timeline-ranges.csv").open("w",newline="") as output:
        writer = csv.DictWriter(output,fieldnames=["profile","role","sample_index"]+list(profiles[0]["sample_ranges"][0]["metrics"]))
        writer.writeheader()
        for p in profiles:
            for r in p["sample_ranges"]:
                writer.writerow(dict(profile=p["profile"],role=p["role"],sample_index=r["sample_index"],**r["metrics"]))
    print(f"Audited four profiles / 84 ranges; wrote {directory}/timeline-analysis.json/.md and timeline-ranges.csv")


if __name__ == "__main__":
    main()
