#!/usr/bin/env python3
"""CPU-only, read-only Nsight SQLite/NVTX audit. Never launches a CUDA tool.

Example: python3 audit_traces.py --output trace-audit.json --markdown trace-audit.md
Outputs must be new. Counts are attributed by runtime enqueue call containment
on the NVTX range's host thread, not by GPU completion-time overlap.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

DEFAULT_CAPTURES = ["nsys-base", "nsys-both"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {value}")))


def api_name(value):
    return re.sub(r"_v[0-9]+$", "", value)


def runtime_summary(calls):
    counts = Counter(call["name"] for call in calls)
    transfers = [copy for call in calls for copy in call["copies"]]
    d2h = [copy for copy in transfers if copy["direction"] == "Device-to-Host"]
    return {
        "runtime_calls": len(calls), "api_counts": dict(sorted(counts.items())),
        "stream_synchronize_calls": counts["cudaStreamSynchronize"],
        "device_synchronize_calls": counts["cudaDeviceSynchronize"],
        "event_synchronize_calls": counts["cudaEventSynchronize"],
        "d2h_runtime_calls": sum(any(copy["direction"] == "Device-to-Host" for copy in call["copies"]) for call in calls),
        "d2h_activity_records": len(d2h), "d2h_copy_operations": sum(copy["operations"] for copy in d2h),
        "d2h_bytes": sum(copy["bytes"] for copy in d2h),
    }


class RuntimeIndex:
    def __init__(self, calls):
        self.threads = defaultdict(list)
        for call in calls:
            self.threads[call["tid"]].append(call)
        self.starts = {}
        for tid, values in self.threads.items():
            values.sort(key=lambda call: call["start"])
            self.starts[tid] = [call["start"] for call in values]
            # CUPTI runtime calls on this capture's thread are sequential.
            # Reject nested API intervals rather than silently missing a boundary.
            require(all(a["end"] <= b["start"] for a, b in zip(values, values[1:])), "overlapping runtime API intervals on one thread")

    def contained(self, span):
        values = self.threads.get(span["tid"], [])
        starts = self.starts.get(span["tid"], [])
        first = bisect_left(starts, span["start"])
        last = bisect_right(starts, span["end"])
        inside = [call for call in values[first:last] if call["end"] <= span["end"]]
        boundary = [call for call in values[max(0, first - 1):last]
                    if call["start"] < span["end"] and call["end"] > span["start"] and call not in inside]
        return inside, boundary


def expected_exports(result):
    outputs = result["outputs"]
    tile = outputs if result["objective"] == "multiclass" else min(outputs, result["output_tile_size"])
    width = max(1, result.get("tree_export_batch_effective", 0))
    expected = []
    for round_id in range(result["rounds"]):
        for begin in range(0, outputs, tile):
            end = min(begin + tile, outputs)
            for first in range(begin, end, width):
                expected.append((round_id, first, min(width, end - first)))
    return expected


def audit_capture(root, name):
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "capture name must be a simple basename")
    paths = {"sqlite": root / f"{name}.sqlite", "result": root / f"{name}-benchmark/result.json",
             "command": root / f"{name}-command.json", "stdout": root / f"{name}.stdout",
             "report": root / f"{name}.nsys-rep"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    result, command = read_json(paths["result"]), read_json(paths["command"])
    require(command.get("returncode") == 0 and command.get("trainer_executable_unchanged") is True,
            f"{name}: successful unchanged executable receipt required")
    require(re.fullmatch(r"[0-9a-f]{64}", command.get("trainer_executable_sha256", "")), f"{name}: binary hash missing")
    emitted = []
    for line in paths["stdout"].read_text().splitlines():
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("kind") == "ghb.training":
                emitted.append(value)
    require(emitted == [result], f"{name}: emitted result does not match saved result JSON")
    samples = result["samples"]
    require(isinstance(samples, list), "samples must be a list")
    for sample in samples:
        require(type(sample["id"]) is int and sample["id"] >= 0, "invalid sample id")
        require(type(sample["host_start_ns"]) is int and type(sample["host_end_ns"]) is int and
                0 <= sample["host_start_ns"] <= sample["host_end_ns"], "invalid sample host interval")
        require(sample["gpu_ms"] is None or (math.isfinite(sample["gpu_ms"]) and sample["gpu_ms"] >= 0), "invalid sample GPU duration")
    require(len({sample["id"] for sample in samples}) == len(samples), f"{name}: duplicate sample ids")
    with sqlite3.connect(paths["sqlite"].resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        integrity = [row[0] for row in db.execute("PRAGMA integrity_check")]
        require(integrity == ["ok"], f"{name}: SQLite integrity check failed: {integrity}")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_MEMCPY", "ENUM_CUDA_MEMCPY_OPER"}
        require(required <= tables, f"{name}: required trace tables missing")
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        kinds = {row["id"]: row["label"] for row in db.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")}
        calls = []
        for row in db.execute("SELECT rowid,* FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"):
            require(row["end"] >= row["start"], "reversed runtime API interval")
            calls.append(dict(id=row["rowid"], start=row["start"], end=row["end"], tid=row["globalTid"],
                              correlation=row["correlationId"], name=api_name(strings[row["nameId"]]),
                              return_value=row["returnValue"], copies=[]))
        by_correlation = {call["correlation"]: call for call in calls}
        require(len(by_correlation) == len(calls), "runtime correlations are ambiguous across captured processes")
        copies = []
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            require(row["end"] >= row["start"] and row["bytes"] >= 0, "invalid GPU memcpy activity")
            copy = dict(start=row["start"], end=row["end"], bytes=row["bytes"], direction=kinds[row["copyKind"]],
                        operations=row["copyCount"] or 1, graph_node=row["graphNodeId"], correlation=row["correlationId"])
            require(copy["correlation"] in by_correlation, "GPU memcpy cannot be attributed to a runtime enqueue call")
            by_correlation[copy["correlation"]]["copies"].append(copy)
            copies.append(copy)
        index = RuntimeIndex(calls)
        nvtx = []
        if "NVTX_EVENTS" in tables:
            domains = {row["domainId"] for row in db.execute("SELECT * FROM NVTX_EVENTS")
                       if (row["text"] or strings.get(row["textId"])) == "ghb" and row["uint64Value"] is None}
            for row in db.execute("SELECT * FROM NVTX_EVENTS"):
                if row["domainId"] in domains and row["uint64Value"] is not None:
                    require(row["end"] is not None and row["end"] >= row["start"], "unclosed/reversed ghb NVTX range")
                    require(row["endGlobalTid"] in (None, row["globalTid"]), "cross-thread ghb range unsupported")
                    nvtx.append(dict(id=row["uint64Value"], stage=row["text"] or strings.get(row["textId"]),
                                     start=row["start"], end=row["end"], tid=row["globalTid"], range_id=row["rangeId"]))
        require(len({span["id"] for span in nvtx}) == len(nvtx), "duplicate NVTX sample payload ids")
        if samples:
            require(result["instrumentation"] == "nvtx", "sample matching requires NVTX instrumentation")
            require({sample["id"] for sample in samples} == {span["id"] for span in nvtx}, "emitted sample ids and NVTX payload ids differ")
        else:
            require(not nvtx, "ghb NVTX ranges exist without emitted samples")
        sample_map = {sample["id"]: sample for sample in samples}
        sample_matches, origins_lower, origins_upper, spans = [], [], [], []
        for span in sorted(nvtx, key=lambda value: value["id"]):
            sample = sample_map[span["id"]]
            require(span["stage"] == sample["stage"], "NVTX range name disagrees with emitted stage")
            inside, boundary = index.contained(span)
            require(not boundary, "runtime API crosses an NVTX boundary; exact scoped counts are ambiguous")
            span.update(context=sample["context"], timing=sample["timing"], calls=inside)
            spans.append(span)
            origins_lower.append(span["end"] - sample["host_end_ns"])
            origins_upper.append(span["start"] - sample["host_start_ns"])
            sample_matches.append(dict(id=span["id"], stage=span["stage"], timing=sample["timing"], context=sample["context"],
                                       nvtx_range_id=span["range_id"], nvtx_start_ns=span["start"], nvtx_end_ns=span["end"],
                                       runtime=runtime_summary(inside), boundary_runtime_ids=[call["id"] for call in boundary]))
        origin = None
        if spans:
            origin = [max(origins_lower), min(origins_upper)]
            require(origin[0] <= origin[1], "sample and NVTX intervals do not share a consistent recorder-clock origin")
        trees = [span for span in spans if span["stage"] == "tree_build"]
        exports = [span for span in spans if span["stage"] == "download" and span["context"]["round"] >= 0 and span["context"]["output"] >= 0]
        exported_operations = sum(span["context"]["operations"] for span in exports)
        tree_calls = {call["id"]: call for span in trees for call in span["calls"]}
        export_calls = {call["id"]: call for span in exports for call in span["calls"]}
        tree_summary, export_summary = runtime_summary(tree_calls.values()), runtime_summary(export_calls.values())
        if spans:
            expected_trees = {(round_id, output) for round_id in range(result["rounds"]) for output in range(result["outputs"])}
            actual_trees = [(span["context"]["round"], span["context"]["output"]) for span in trees]
            require(len(trees) == len(expected_trees) and set(actual_trees) == expected_trees, "tree-build stage coverage mismatch")
            require(all(span["context"]["depth"] == -1 for span in trees + exports), "tree or export scope has a per-level context")
            require(tree_summary["stream_synchronize_calls"] == 0 and tree_summary["device_synchronize_calls"] == 0 and
                    tree_summary["event_synchronize_calls"] == 0 and tree_summary["d2h_runtime_calls"] == 0,
                    "tree-build submission contains a host wait or correlated D2H operation")
            actual_exports = [(span["context"]["round"], span["context"]["output"], span["context"]["operations"]) for span in exports]
            require(actual_exports == expected_exports(result), "export batches do not match rounds/output tiles/effective width")
            require(exported_operations == len(trees), "export operations do not cover every output tree")
            expected_waits = 1 if result.get("tree_export_batch_effective", 0) else 2
            require(all(runtime_summary(span["calls"])["stream_synchronize_calls"] == expected_waits for span in exports),
                    "completed-tree export wait count disagrees with compact/batched contract")
        stage_counts = Counter(span["stage"] for span in spans)
        all_stage_calls = {call["id"]: call for span in spans for call in span["calls"]}
        outside_stage = [call for call in calls if call["id"] not in all_stage_calls]
        kernels = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            kernels = [dict(row) for row in db.execute("SELECT start,end,shortName,graphNodeId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        graph_kernels = [row for row in kernels if row["graphNodeId"] not in (None, 0)]
        tree_correlations = {call["correlation"] for call in tree_calls.values()}
        attributed_graph_kernels = [row for row in graph_kernels if row["correlationId"] in tree_correlations]
        graph_trace_count = db.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0] if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else 0
        kernel_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        for row in kernels:
            group = kernel_groups[strings[row["shortName"]]]
            group["count"] += 1; group["total_ns"] += row["end"] - row["start"]
        transfer_calls = [dict(runtime_id=call["id"], name=call["name"], start_ns=call["start"], host_api_ns=call["end"] - call["start"],
                               transfers=call["copies"]) for call in calls if call["copies"]]
        report = dict(name=name, status="verified_scopes" if spans else "verified_artifacts_without_training_scopes",
                      sqlite_integrity="ok", executable_sha256=command["trainer_executable_sha256"],
                      configuration={key: result.get(key) for key in ["objective", "rows", "test_rows", "features", "outputs", "seed", "test_seed", "rounds", "max_depth", "max_bins", "output_tile_size", "tree_execution", "tree_export_batch_requested", "tree_export_batch_effective", "instrumentation"]},
                      sample_count=len(samples), matched_nvtx_count=len(nvtx), sample_stage_counts=dict(stage_counts),
                      consistent_recorder_origin_ns=origin, runtime_boundary_crossings=0, sample_matches=sample_matches,
                      whole_capture_runtime=runtime_summary(calls), matched_stage_union_runtime=runtime_summary(all_stage_calls.values()),
                      outside_matched_stages_runtime=runtime_summary(outside_stage),
                      tree_build=dict(ranges=len(trees), runtime=tree_summary, expected_trees=result["rounds"] * result["outputs"],
                                      observed_graph_node_kernels_attributed_to_tree_launches=len(attributed_graph_kernels)),
                      tree_export=dict(ranges=len(exports), tree_operations=exported_operations, runtime=export_summary,
                                       stream_waits_per_range=[runtime_summary(span["calls"])["stream_synchronize_calls"] for span in exports]),
                      graph_trace=dict(node_granularity_requested="--cuda-graph-trace=node" in command["command"],
                                       graph_execution_records=graph_trace_count, kernel_records=len(kernels), graph_node_kernel_records=len(graph_kernels),
                                       graph_node_d2h_records=sum(copy["graph_node"] not in (None, 0) and copy["direction"] == "Device-to-Host" for copy in copies)),
                      whole_capture_kernel_groups=dict(kernel_groups), whole_capture_transfer_calls=transfer_calls)
    for key, path in paths.items():
        require(sha256(path) == hashes[key], f"{name}: {key} changed during audit")
    report["artifacts"] = {key: dict(path=str(path.resolve()), sha256=hashes[key]) for key, path in paths.items()}
    report["limits"] = [
        "NVTX intervals bound host submission. GPU work can finish later; D2H attribution follows CUDA correlation ids, not timestamp overlap.",
        "Counts outside matched stages include setup, prediction, validation, and cleanup; they are not whole-training counts.",
        "Host export scopes can overlap subsequent tree submission scopes; their durations and nested runtime totals are not additive.",
        "Graph-only captures omit individual graph-node kernel/copy activity. Node and graph-only kernel totals are not comparable.",
        "Profiling observations explain this capture and are not uninstrumented performance rankings.",
    ]
    if not spans:
        report["limits"].append("Instrumentation was off: no emitted stage samples or NVTX training ranges exist. Only whole-capture API/kernel/transfer counts are reported; training phase attribution is unavailable.")
    return report


def markdown(report):
    rows = ["# Nsight trace audit", "", "CPU-only audit of immutable capture artifacts, SQLite integrity, and emitted stage/NVTX correspondence.", "",
            "| Capture | Matched stages | Tree scopes | Export scopes | Export stream waits | Tree-scope waits / D2H calls |", "|---|---:|---:|---:|---:|---:|"]
    for capture in report["captures"]:
        tree, export = capture["tree_build"], capture["tree_export"]
        scoped = capture["matched_nvtx_count"] > 0
        rows.append(f"| {capture['name']} | {capture['matched_nvtx_count']} | {tree['ranges'] if scoped else 'unavailable'} | {export['ranges'] if scoped else 'unavailable'} | {export['runtime']['stream_synchronize_calls'] if scoped else 'unavailable'} | {str(tree['runtime']['stream_synchronize_calls']) + ' / ' + str(tree['runtime']['d2h_runtime_calls']) if scoped else 'unavailable'} |")
    if report.get("export_comparison"):
        comparison = report["export_comparison"]
        rows.extend(["", f"For the same {comparison['trees']} output trees, completed-tree export stream waits fell from **{comparison['before_waits']} to {comparison['after_waits']}**, across {comparison['before_ranges']} compact exports versus {comparison['after_ranges']} batches. Every old export contains two waits; every batch contains one. These are counts in matched export ranges, not whole-process totals."])
    rows.extend(["", "Every emitted sample ID and stage name matches exactly one ghb NVTX range. Sample host intervals and NVTX intervals admit one consistent recorder-clock origin per instrumented capture. Each instrumented tree scope contains zero runtime stream/device/event waits and zero runtime calls correlated with device-to-host copies.", "",
                 "Both captures in this root/split experiment record graph-node activity. NVTX tree ranges bound host submission, so transfer classification follows CUDA correlation IDs, including GPU work completing after a host range ends.", "",
                 "Captures without stage instrumentation retain whole-capture API, kernel, and transfer records, but no training-stage counts are inferred. Counts outside matched ranges in instrumented captures can include setup, inference, validation, and cleanup.", "",
                 "The JSON includes every matched sample, batch count/context, raw transfer attribution, input SHA256, and scope limitations. Nested/export scope durations are not additive. Profiler observations are diagnostic, not a performance ranking.", ""])
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--capture", action="append", help="Repeat to select captures; defaults to first and optimized five captures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for path in [args.output, args.markdown]:
        if path is not None:
            require(not path.exists(), f"refusing to overwrite {path}")
    captures = [audit_capture(args.root.resolve(), name) for name in (args.capture or DEFAULT_CAPTURES)]
    report = dict(schema_version=1, audit="ghb.cpu_only_nsight_trace", auditor_sha256=sha256(Path(__file__)), captures=captures)
    names = {capture["name"]: capture for capture in captures}
    if "nsys-wide" in names and "nsys-batched" in names:
        before, after = names["nsys-wide"], names["nsys-batched"]
        for key in ["objective", "rows", "test_rows", "features", "outputs", "seed", "test_seed", "rounds", "max_depth", "max_bins", "output_tile_size", "tree_execution"]:
            require(before["configuration"][key] == after["configuration"][key], f"export comparison workload mismatch: {key}")
        report["export_comparison"] = dict(trees=before["tree_build"]["ranges"], before_ranges=before["tree_export"]["ranges"],
                                           after_ranges=after["tree_export"]["ranges"], before_waits=before["tree_export"]["runtime"]["stream_synchronize_calls"],
                                           after_waits=after["tree_export"]["runtime"]["stream_synchronize_calls"],
                                           graph_kernel_coverage_comparable=False)
    with args.output.open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False); target.write("\n")
    if args.markdown:
        with args.markdown.open("x") as target:
            target.write(markdown(report))
    print(json.dumps(dict(captures=len(captures), matched_samples=sum(c["matched_nvtx_count"] for c in captures),
                          export_comparison=report.get("export_comparison"), output=str(args.output)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
        print(f"trace audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
