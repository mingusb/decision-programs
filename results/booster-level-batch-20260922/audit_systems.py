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

DEFAULT_CAPTURES = ["nsys-per-output-graph", "nsys-output-batch-graph", "nsys-output-batch-stream"]


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
    calls = list(calls)
    counts = Counter(call["name"] for call in calls)
    transfers = [copy for call in calls for copy in call["copies"]]
    d2h = [copy for copy in transfers if copy["direction"] == "Device-to-Host"]
    return {
        "runtime_calls": len(calls), "api_counts": dict(sorted(counts.items())),
        "stream_synchronize_calls": counts["cudaStreamSynchronize"],
        "device_synchronize_calls": counts["cudaDeviceSynchronize"],
        "event_synchronize_calls": counts["cudaEventSynchronize"],
        "host_synchronize_calls": sum(counts[name] for name in ["cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaThreadSynchronize"]),
        "device_allocation_calls": sum(count for name, count in counts.items() if device_allocation(name)),
        "device_free_calls": sum(count for name, count in counts.items() if name in {"cudaFree", "cudaFreeAsync", "cudaFreeArray", "cudaFreeMipmappedArray"}),
        "host_allocation_calls": sum(count for name, count in counts.items() if name in {"cudaMallocHost", "cudaHostAlloc"}),
        "blocking_copy_calls": sum(count for name, count in counts.items() if name.startswith("cudaMemcpy") and "Async" not in name),
        "kernel_launch_calls": sum(count for name, count in counts.items() if name.startswith("cudaLaunchKernel") or name in {"cudaLaunchCooperativeKernel", "cudaLaunchCooperativeKernelMultiDevice"}),
        "graph_launch_calls": counts["cudaGraphLaunch"],
        "d2h_runtime_calls": sum(any(copy["direction"] == "Device-to-Host" for copy in call["copies"]) for call in calls),
        "d2h_activity_records": len(d2h), "d2h_copy_operations": sum(copy["operations"] for copy in d2h),
        "d2h_bytes": sum(copy["bytes"] for copy in d2h),
    }


def device_allocation(name):
    return (name.startswith("cudaMalloc") and name != "cudaMallocHost") or name == "cudaGraphAddMemAllocNode"


def batch_builder(result):
    return result.get("tree_build") == "output-batch"


def expected_trees(result):
    width = result["tree_batch_size"] if batch_builder(result) else 1
    return [(round_id, output, min(width, result["outputs"] - output))
            for round_id in range(result["rounds"]) for output in range(0, result["outputs"], width)]


def kernel_family(name):
    # Function names identify stages inside graph replay; they are not NVTX
    # per-node scope measurements. Whole-capture groups retain full names.
    if any(token in name for token in ["warp_candidates", "warp_winners", "split_candidates", "split_winners"]):
        return "split_search"
    if any(token in name for token in ["global_accumulate", "shared_accumulate", "global_histogram", "shared_histogram", "clear_active_histograms", "::clear<"]):
        return "deeper_histogram"
    if any(token in name for token in ["::accumulate<", "seed_batch", "seed_counts"]):
        return "root_histogram"
    if any(token in name for token in ["count_global", "count_shared"]):
        return "immutable_root_counts"
    if any(token in name for token in ["materialize", "scan_splits", "scan_batch", "prefix_blocks", "prefix_batch", "route_rows", "route_batch", "::advance(", "advance_batch"]):
        return "frontier_materialize_route_advance"
    if any(token in name for token in ["initialize_batch", "::initialize("]):
        return "tree_initialize"
    if any(token in name for token in ["predict_batch", "predict_tree"]):
        return "tree_prediction"
    if "gradients" in name:
        return "derivatives"
    if "loss" in name:
        return "objective"
    return "other"


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
    if batch_builder(result):
        return expected_trees(result)
    outputs = result["outputs"]
    tile = result.get("root_histogram_batch_size", 0) or (outputs if result["objective"] == "multiclass" else min(outputs, result["output_tile_size"]))
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
    require(command.get("returncode") == 0 and command.get("command_executables_unchanged") is True,
            f"{name}: successful unchanged executable receipt required")
    trainer_entries = [(path, digest) for path, digest in command.get("command_executables_sha256", {}).items() if Path(path).name == "ghb_bench"]
    require(len(trainer_entries) == 1 and re.fullmatch(r"[0-9a-f]{64}", trainer_entries[0][1]), f"{name}: unique binary hash missing")
    trainer_path, trainer_hash = trainer_entries[0]
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
    require(samples and result["instrumentation"] == "nvtx", "this experiment requires emitted NVTX stage samples")
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
        required = {"StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "ENUM_CUDA_MEMCPY_OPER", "NVTX_EVENTS"}
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
            planned_trees = expected_trees(result)
            actual_trees = [(span["context"]["round"], span["context"]["output"], span["context"]["operations"]) for span in trees]
            require(actual_trees == planned_trees, "tree-build stage coverage/operation width mismatch")
            require(all(span["context"]["depth"] == -1 for span in trees + exports), "tree or export scope has a per-level context")
            require(tree_summary["host_synchronize_calls"] == 0 and tree_summary["d2h_runtime_calls"] == 0 and
                    tree_summary["device_allocation_calls"] == 0 and tree_summary["device_free_calls"] == 0 and tree_summary["blocking_copy_calls"] == 0,
                    "tree-build submission contains a host wait, correlated D2H operation, device allocation or device free")
            actual_exports = [(span["context"]["round"], span["context"]["output"], span["context"]["operations"]) for span in exports]
            require(actual_exports == expected_exports(result), "export batches do not match rounds/output tiles/effective width")
            require(exported_operations == result["rounds"] * result["outputs"], "export operations do not cover every output tree")
            export_width = result.get("tree_export_batch_effective", 0)
            def expected_waits(span):
                return ((span["context"]["operations"] + export_width - 1) // export_width if batch_builder(result) else 1) if export_width else 2
            require(all(runtime_summary(span["calls"])["stream_synchronize_calls"] == expected_waits(span) for span in exports),
                    "completed-tree export wait count disagrees with compact/batched contract")
        stage_counts = Counter(span["stage"] for span in spans)
        all_stage_calls = {call["id"]: call for span in spans for call in span["calls"]}
        outside_stage = [call for call in calls if call["id"] not in all_stage_calls]
        kernels = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            kernels = [dict(row) for row in db.execute("SELECT rowid,start,end,shortName,demangledName,graphNodeId,correlationId,gridX,gridY,gridZ,blockX,blockY,blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        graph_kernels = [row for row in kernels if row["graphNodeId"] not in (None, 0)]
        tree_correlations = {call["correlation"] for call in tree_calls.values()}
        boosting_calls = {call["id"]: call for span in spans
                          if span["context"]["round"] >= 0 or span["stage"] == "histogram"
                          for call in span["calls"]}
        boosting_correlations = {call["correlation"] for call in boosting_calls.values()}
        attributed_graph_kernels = [row for row in graph_kernels if row["correlationId"] in tree_correlations]
        graph_trace_count = db.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0] if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else 0
        kernel_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        tree_kernel_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        family_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        scoped_kernel_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        boosting_kernel_groups = defaultdict(lambda: {"count": 0, "total_ns": 0})
        # Shortest containing NVTX range is the most specific attribution;
        # outer tree_build scopes deliberately contain stream stage scopes.
        innermost = {}
        for span in sorted(spans, key=lambda span: span["end"] - span["start"], reverse=True):
            for call in span["calls"]:
                innermost[call["correlation"]] = span["stage"]
        uncorrelated_kernels = []
        for row in kernels:
            kernel_name = strings[row["demangledName"]]
            group = kernel_groups[kernel_name]
            group["count"] += 1; group["total_ns"] += row["end"] - row["start"]
            require(row["end"] >= row["start"], "reversed kernel activity interval")
            if row["correlationId"] not in by_correlation:
                uncorrelated_kernels.append(row["rowid"])
            if row["correlationId"] in tree_correlations:
                for key, groups in [(kernel_name, tree_kernel_groups), (kernel_family(kernel_name), family_groups)]:
                    groups[key]["count"] += 1; groups[key]["total_ns"] += row["end"] - row["start"]
            if row["correlationId"] in boosting_correlations:
                group = boosting_kernel_groups[kernel_name]
                group["count"] += 1; group["total_ns"] += row["end"] - row["start"]
            stage = innermost.get(row["correlationId"], "outside_matched_scopes")
            scoped_kernel_groups[stage]["count"] += 1; scoped_kernel_groups[stage]["total_ns"] += row["end"] - row["start"]
        require(not uncorrelated_kernels, "kernel activities lack runtime enqueue correlation")
        for span in trees:
            counts = runtime_summary(span["calls"])
            require(not any(counts[key] for key in ["host_synchronize_calls", "device_allocation_calls", "device_free_calls", "d2h_runtime_calls", "blocking_copy_calls"]),
                    "individual tree scope violates GPU-resident submission contract")
            require(counts["graph_launch_calls"] == (1 if result["tree_execution"] == "graph" else 0), "unexpected graph launches per tree scope")
        require("--cuda-graph-trace=node" in command["command"], "node-granularity CUDA graph tracing required")
        require(result["tree_execution"] != "graph" or attributed_graph_kernels, "graph tree scopes have no correlated graph-node kernel records")
        first_tree = min((span["start"] for span in trees), default=None)
        before_first_tree = [call for call in calls if first_tree is not None and call["end"] <= first_tree]
        setup_spans = [span for span in spans if span["stage"] == "initialize" and span["context"]["round"] >= 0 and span["context"]["output"] >= 0]
        setup_calls = {call["id"]: call for span in setup_spans for call in span["calls"] if call["id"] not in tree_calls}
        transfer_calls = [dict(runtime_id=call["id"], name=call["name"], start_ns=call["start"], host_api_ns=call["end"] - call["start"],
                               transfers=call["copies"]) for call in calls if call["copies"]]
        report = dict(name=name, status="verified_scopes" if spans else "verified_artifacts_without_training_scopes",
                      sqlite_integrity="ok", executable_sha256=trainer_hash, executable_path=trainer_path,
                      configuration={key: result.get(key) for key in ["objective", "rows", "test_rows", "features", "outputs", "seed", "test_seed", "rounds", "max_depth", "max_bins", "output_tile_size", "tree_execution", "tree_build", "tree_batch_size", "histogram", "root_histogram", "root_counts", "split_batch", "split_policy", "tree_export_batch_requested", "tree_export_batch_effective", "instrumentation"]},
                      sample_count=len(samples), matched_nvtx_count=len(nvtx), sample_stage_counts=dict(stage_counts),
                      consistent_recorder_origin_ns=origin, runtime_boundary_crossings=0, sample_matches=sample_matches,
                      whole_capture_runtime=runtime_summary(calls), matched_stage_union_runtime=runtime_summary(all_stage_calls.values()),
                      outside_matched_stages_runtime=runtime_summary(outside_stage),
                      tree_build=dict(ranges=len(trees), runtime=tree_summary, expected_output_trees=result["rounds"] * result["outputs"],
                                      tree_operations=sum(span["context"]["operations"] for span in trees), expected_scopes=len(expected_trees(result)),
                                      kernel_records=sum(group["count"] for group in tree_kernel_groups.values()),
                                      observed_graph_node_kernels_attributed_to_tree_launches=len(attributed_graph_kernels),
                                      kernel_groups=dict(tree_kernel_groups), function_family_groups=dict(family_groups)),
                      tree_export=dict(ranges=len(exports), tree_operations=exported_operations, runtime=export_summary,
                                       stream_waits_per_range=[runtime_summary(span["calls"])["stream_synchronize_calls"] for span in exports]),
                      boosting_including_root_count_setup=dict(runtime=runtime_summary(boosting_calls.values()),
                          kernel_records=sum(group["count"] for group in boosting_kernel_groups.values()), kernel_groups=dict(boosting_kernel_groups)),
                      graph_trace=dict(node_granularity_requested="--cuda-graph-trace=node" in command["command"],
                                       graph_execution_records=graph_trace_count, kernel_records=len(kernels), graph_node_kernel_records=len(graph_kernels),
                                       graph_node_d2h_records=sum(copy["graph_node"] not in (None, 0) and copy["direction"] == "Device-to-Host" for copy in copies)),
                      before_first_tree_runtime=runtime_summary(before_first_tree),
                      graph_setup_outside_tree_runtime=runtime_summary(setup_calls.values()),
                      graph_setup_outside_tree_ranges=[dict(id=span["id"], context=span["context"], runtime=runtime_summary(span["calls"])) for span in setup_spans if span["timing"] == "host"],
                      kernel_records_by_innermost_nvtx_stage=dict(scoped_kernel_groups),
                      whole_capture_kernel_groups=dict(kernel_groups), whole_capture_transfer_calls=transfer_calls)
    for key, path in paths.items():
        require(sha256(path) == hashes[key], f"{name}: {key} changed during audit")
    report["artifacts"] = {key: dict(path=str(path.resolve()), sha256=hashes[key]) for key, path in paths.items()}
    report["limits"] = [
        "NVTX intervals bound host submission. GPU work can finish later; D2H attribution follows CUDA correlation ids, not timestamp overlap.",
        "Counts outside matched stages include setup, prediction, validation, and cleanup; they are not whole-training counts.",
        "Host export scopes can overlap subsequent tree submission scopes; their durations and nested runtime totals are not additive.",
        "Graph-only captures omit individual graph-node kernel/copy activity. Node and graph-only kernel totals are not comparable.",
        "Function-family breakdowns classify demangled kernel names, including graph nodes. Only the separate innermost-NVTX table is measured host-stage attribution.",
        "The trace records CUDA runtime APIs. It does not certify absence of arbitrary CPU computation or untraced direct driver calls.",
        "Profiling observations explain this capture and are not uninstrumented performance rankings.",
    ]
    if not spans:
        report["limits"].append("Instrumentation was off: no emitted stage samples or NVTX training ranges exist. Only whole-capture API/kernel/transfer counts are reported; training phase attribution is unavailable.")
    return report


def markdown(report):
    rows = ["# Nsight Systems audit of output-tile construction", "",
            "CPU-only audit of immutable captures, SQLite integrity, runtime enqueue correlations and exact emitted-sample/NVTX correspondence. Profiler durations explain these captures; they do not rank uninstrumented implementations.", "",
            "| Capture | Tree scopes | Output trees | Boosting kernels including count setup | Export scopes | Export stream waits | Tree waits / D2H / device allocations |", "|---|---:|---:|---:|---:|---:|---:|"]
    for capture in report["captures"]:
        tree, export = capture["tree_build"], capture["tree_export"]
        runtime = tree["runtime"]
        rows.append(f"| {capture['name']} | {tree['ranges']} | {tree['tree_operations']} | {capture['boosting_including_root_count_setup']['kernel_records']} | {export['ranges']} | {export['runtime']['stream_synchronize_calls']} | {runtime['host_synchronize_calls']} / {runtime['d2h_runtime_calls']} / {runtime['device_allocation_calls']} |")
    for comparison in report["comparisons"]:
        rows.extend(["", f"`{comparison['before']}` → `{comparison['after']}`: matched tree scopes {comparison['before_tree_scopes']} → {comparison['after_tree_scopes']}; complete-boosting kernel activity records including immutable count setup {comparison['before_boosting_kernels']} → {comparison['after_boosting_kernels']}; tree graph enqueue calls {comparison['before_graph_launches']} → {comparison['after_graph_launches']}. Both represent {comparison['output_trees']} independent output trees. These are observed counts, not timing speedups."])
    rows.extend(["", "Every emitted stage ID/name matches a single ghb NVTX range. Tree scopes cover the exact round/output schedule and retain correct operations counts for full and short tiles. Each tree scope has zero explicit runtime host synchronization, synchronous copies, correlated device-to-host transfer, device allocation and device free calls.", "",
                 "NVTX intervals bound host submission. GPU kernels and copies can finish after a host range ends; all attribution follows CUDA enqueue correlation IDs. Nested scopes are not summed twice in runtime unions. The scope contract checks runtime API records, not arbitrary CPU work or untraced direct driver calls.", "",
                 "## Setup outside tree scopes", "",
                 "The following calls are retained as setup observations. Their presence in a whole-process trace does not imply a wait/allocation inside a tree batch. Counts before the first tree also include preparation, initial objective evaluation and, on the comparison path, initial root work.", "",
                 "| Capture | Before-first-tree waits | Before-first-tree device allocations | Graph-setup waits | Graph-setup device allocations |", "|---|---:|---:|---:|---:|"])
    for capture in report["captures"]:
        before, setup = capture["before_first_tree_runtime"], capture["graph_setup_outside_tree_runtime"]
        rows.append(f"| {capture['name']} | {before['host_synchronize_calls']} | {before['device_allocation_calls']} | {setup['host_synchronize_calls']} | {setup['device_allocation_calls']} |")
    rows.extend(["", "## Kernel work inside tree scopes", "",
                 "Families classify actual demangled kernel names, including graph-node records. Graph captures have one outer tree scope; the JSON separately retains direct innermost-NVTX stage attribution for stream captures. Durations are sums of activity-record intervals and are diagnostic only.", "",
                 "| Capture | Kernel family | Records | Summed kernel ms | Share of tree kernel time |", "|---|---|---:|---:|---:|"])
    for capture in report["captures"]:
        families = capture["tree_build"]["function_family_groups"]
        total = sum(group["total_ns"] for group in families.values())
        for family, group in sorted(families.items(), key=lambda item: item[1]["total_ns"], reverse=True):
            share = 100 * group["total_ns"] / total if total else 0
            rows.append(f"| {capture['name']} | {family} | {group['count']} | {group['total_ns'] / 1e6:.6f} | {share:.2f}% |")
    rows.extend(["", "The JSON preserves full kernel names, raw transfer attribution, API counts, every sample/context match, setup ranges and SHA256 for all inputs. Root work outside legacy per-tree scopes remains outside that table; whole-capture groups are also retained. Kernel counts from graph-node and graph-only traces must not be compared; these captures explicitly request graph-node detail.", ""])
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--capture", action="append", help="Repeat to select captures; defaults to legacy graph, batch graph, batch stream")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    for path in [args.output, args.markdown]:
        if path is not None:
            require(not path.exists(), f"refusing to overwrite {path}")
    captures = [audit_capture(args.root.resolve(), name) for name in (args.capture or DEFAULT_CAPTURES)]
    report = dict(schema_version=2, audit="ghb.cpu_only_output_batch_nsight_trace", auditor_sha256=sha256(Path(__file__)), captures=captures, comparisons=[])
    names = {capture["name"]: capture for capture in captures}
    for before_name, after_name in [(DEFAULT_CAPTURES[0], DEFAULT_CAPTURES[1]), (DEFAULT_CAPTURES[1], DEFAULT_CAPTURES[2])]:
        if before_name not in names or after_name not in names:
            continue
        before, after = names[before_name], names[after_name]
        require(before["executable_sha256"] == after["executable_sha256"], "comparison binary differs")
        for key in ["objective", "rows", "test_rows", "features", "outputs", "seed", "test_seed", "rounds", "max_depth", "max_bins", "output_tile_size", "histogram", "root_histogram", "root_counts", "split_batch", "split_policy"]:
            require(before["configuration"][key] == after["configuration"][key], f"comparison workload mismatch: {key}")
        report["comparisons"].append(dict(before=before_name, after=after_name,
            output_trees=before["tree_build"]["tree_operations"],
            before_tree_scopes=before["tree_build"]["ranges"], after_tree_scopes=after["tree_build"]["ranges"],
            before_tree_kernels=before["tree_build"]["kernel_records"], after_tree_kernels=after["tree_build"]["kernel_records"],
            before_boosting_kernels=before["boosting_including_root_count_setup"]["kernel_records"], after_boosting_kernels=after["boosting_including_root_count_setup"]["kernel_records"],
            before_graph_launches=before["tree_build"]["runtime"]["graph_launch_calls"], after_graph_launches=after["tree_build"]["runtime"]["graph_launch_calls"],
            before_export_waits=before["tree_export"]["runtime"]["stream_synchronize_calls"], after_export_waits=after["tree_export"]["runtime"]["stream_synchronize_calls"],
            graph_kernel_coverage_comparable=True))
    with args.output.open("x") as target:
        json.dump(report, target, indent=2, allow_nan=False); target.write("\n")
    if args.markdown:
        with args.markdown.open("x") as target:
            target.write(markdown(report))
    print(json.dumps(dict(captures=len(captures), matched_samples=sum(c["matched_nvtx_count"] for c in captures),
                          comparisons=report["comparisons"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
        print(f"trace audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
