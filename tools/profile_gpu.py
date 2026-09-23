#!/usr/bin/env python3
"""Run one GPU diagnostic tool, retaining independent, non-ranking evidence.

Contract: the target must be an absolute executable path. The target runs in the
new output directory, so use absolute paths for its data/file arguments. No shell
or external timeout wrapper is used. Inherited injection is rejected; one tool
owns the target process. Exit zero requires both tool success and actual evidence.
Profiler/instrumentation timings must not rank uninstrumented implementations.
"""
import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid


TOOLS = ("nsys", "ncu", "memcheck", "racecheck", "synccheck", "initcheck",
         "nvbit-count", "nvbit-graph", "nvbit-memory", "cupti-trace",
         "cupti-range", "cupti-pc")
INJECTIONS = ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH",
              "NVTX_INJECTION64_PATH")
SANITIZERS = ("memcheck", "racecheck", "synccheck", "initcheck")
CTA_METRIC = "sm__ctas_launched.sum"
GRAPH_METRIC = "launch__graph_exec_cuda_id"
GRAPH_IDENTITIES = (GRAPH_METRIC, "launch__graph_src_cuda_id")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=TOOLS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--timeout", type=float, default=120, help="target timeout in seconds")
    parser.add_argument("--kernel", help="NCU kernel-name regular expression")
    parser.add_argument("--section", action="append", default=[], help="NCU section; repeatable")
    parser.add_argument("--metrics", help="NCU comma-separated metrics")
    parser.add_argument("--replay-mode", choices=("kernel", "application", "range", "app-range"))
    parser.add_argument("--graph-profiling", choices=("node", "graph"), help="NCU workload granularity")
    parser.add_argument("--cache-control", choices=("all", "none"), help="NCU cache flushing")
    parser.add_argument("--launch-count", type=positive, default=2, help="NCU/NVBit count launch bound")
    parser.add_argument("--launch-skip", type=nonnegative, default=0, help="NCU/NVBit count launch offset")
    parser.add_argument("--capture", action="store_true", help="GH_PROFILE_CAPTURE=1; Nsight CUDA profiler API range")
    parser.add_argument("--nsys-resolve-symbols", action="store_true", help="resolve Nsight CPU sample/backtrace symbols (off by default)")
    parser.add_argument("--nsys-graph-trace", choices=("node", "graph"), help="Systems CUDA graph trace granularity")
    parser.add_argument("--padding", type=nonnegative, help="memcheck allocation padding bytes")
    parser.add_argument("--leak-check", choices=("full", "no"), help="memcheck allocation leak checking")
    parser.add_argument("--initcheck-address-space", choices=("global", "shared", "all"))
    parser.add_argument("--track-unused-memory", action="store_true", help="initcheck unused allocations")
    parser.add_argument("--unused-memory-threshold", type=nonnegative, help="initcheck unused allocation reporting percentage")
    parser.add_argument("--instruction-end", type=positive, help="NVBit exclusive static instruction index")
    parser.add_argument("--function-count", type=positive, help="NVBit graph: unique functions, NOT graph launches")
    parser.add_argument("--function-skip", type=nonnegative, default=0, help="NVBit graph: first-seen function offset")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- /absolute/binary [arguments]")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command or not Path(args.command[0]).is_absolute():
        parser.error("use -- /absolute/binary [arguments]")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if any((args.kernel, args.section, args.metrics, args.replay_mode, args.graph_profiling, args.cache_control)) and args.tool != "ncu":
        parser.error("NCU metric/replay/graph/cache options apply only to ncu")
    if (args.nsys_resolve_symbols or args.nsys_graph_trace) and args.tool != "nsys":
        parser.error("Nsight Systems options apply only to nsys")
    if (args.padding is not None or args.leak_check is not None) and args.tool != "memcheck":
        parser.error("--padding/--leak-check apply only to memcheck")
    if (args.initcheck_address_space or args.track_unused_memory or args.unused_memory_threshold is not None) and args.tool != "initcheck":
        parser.error("address-space/unused-memory options apply only to initcheck")
    if args.unused_memory_threshold is not None and (args.unused_memory_threshold > 100 or not args.track_unused_memory):
        parser.error("--unused-memory-threshold must be 0..100 and requires --track-unused-memory")
    if args.tool == "ncu":
        replay, granularity = args.replay_mode or "kernel", args.graph_profiling or "node"
        aggregate = replay in ("range", "app-range") or granularity == "graph"
        if granularity == "graph" and replay != "kernel":
            parser.error("--graph-profiling graph requires kernel replay; app-range already profiles entire ranges containing graphs")
        if replay in ("range", "app-range") and not args.capture:
            parser.error("range replay modes require --capture for the CUDA profiler API range")
        if aggregate and args.kernel:
            parser.error("--kernel filters individual kernels; aggregate graph/range profiling has no kernel-name filter")
        if aggregate and any(name in section for section in args.section for name in ("SourceCounters", "InstructionStats")):
            parser.error("source sections contain metrics unavailable for aggregate graph/range workloads; use node kernel profiling or compatible explicit metrics")
        metrics = args.metrics or ""
        if aggregate and ("__sass_" in metrics or granularity == "graph" and "sass__" in metrics):
            parser.error("requested SASS source metrics are unsupported for this aggregate workload")
    if args.instruction_end is not None and not args.tool.startswith("nvbit-"):
        parser.error("--instruction-end applies only to NVBit")
    if (args.function_count is not None or args.function_skip) and args.tool != "nvbit-graph":
        parser.error("--function-count/--function-skip apply only to nvbit-graph")
    if args.function_skip >= 100 or (args.function_count is not None and
                                      args.function_skip + args.function_count > 100):
        parser.error("official NVBit graph counter has 100 unique-function slots")
    return args


def digest(path):
    path = Path(path).resolve(strict=True)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": checksum.hexdigest()}


def executable(name, environ):
    found = shutil.which(name, path=environ.get("PATH", os.defpath))
    if not found:
        raise ValueError(f"required executable is not on PATH: {name}")
    return str(Path(found).resolve(strict=True))


def sdk_root(kind, environ):
    variable = kind.upper() + "_ROOT"
    if environ.get(variable):
        root = Path(environ[variable]).expanduser().resolve(strict=True)
    else:
        base = Path.home() / ".local/opt/gpu-profiling"
        pattern = "nvbit-*/nvbit_release_x86_64" if kind == "nvbit" else "cupti-*/cuda_cupti-*-archive"
        roots = list(base.glob(pattern))
        if not roots:
            raise ValueError(f"set {variable} to the installed SDK directory")
        # Version components, rather than lexical order (13.10 must beat 13.9).
        root = max(roots, key=lambda p: tuple(map(int, re.findall(r"\d+", p.parent.name))))
    if not root.is_dir():
        raise ValueError(f"{variable} is not a directory: {root}")
    return root


def option_values(command, option):
    values = []
    for index, word in enumerate(command[1:], 1):
        if word == option and index + 1 < len(command):
            values.append(command[index + 1])
        elif word.startswith(option + "="):
            values.append(word.split("=", 1)[1])
    return values


def require_stream(tool, command):
    """The ordinary NVBit counter syncs during capture; range injection lacks graph callbacks."""
    tree = option_values(command, "--tree-execution")
    launch = option_values(command, "--launch")
    if "graph" in tree + launch:
        raise ValueError(f"{tool} does not support this graph target; choose explicit stream execution")
    name = Path(command[0]).name
    if (name in ("ghb_bench", "ghb_real_bench") or "booster" in name.lower()) and tree != ["stream"]:
        raise ValueError(f"{tool} requires --tree-execution stream for booster benchmarks")
    if name in ("ghb_instrumentation_bench", "histogram_bench") and launch != ["stream"]:
        raise ValueError(f"{tool} requires --launch stream for histogram benchmarks")


def build_plan(args, environ):
    inherited = [name for name in INJECTIONS if environ.get(name)]
    if inherited:
        raise ValueError("refusing inherited injection: " + ", ".join(inherited))
    target = Path(args.command[0]).resolve(strict=True)
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ValueError(f"target is not executable: {target}")
    output = args.output.resolve()
    command = [str(target), *args.command[1:]]
    overrides = {"GH_PROFILE_CAPTURE": "1" if args.capture else "0"}
    artifacts = {"target": digest(target), "runner": digest(__file__)}
    notes = ["diagnostic only: instrumentation timings do not rank production performance",
             "target cwd is the evidence directory; file arguments should be absolute"]
    expected_reports = []
    decoder = None
    nsys_session = None
    workload = "kernel"
    settings = {}
    if args.tool == "nsys":
        tool = executable("nsys", environ)
        nsys_session = "gh-profile-" + uuid.uuid4().hex
        granularity = args.nsys_graph_trace or "tool-default"
        workload = "graph" if granularity == "graph" else "kernel"
        settings = {"graph_trace": granularity}
        prefix = [tool, "profile", "--trace=cuda,nvtx,osrt",
                  "--sample=none", "--cpuctxsw=none", "--session-new=" + nsys_session,
                  "--resolve-symbols=" + str(args.nsys_resolve_symbols).lower(),
                  "--output=" + str(output / "profile")]
        if args.nsys_graph_trace:
            prefix += ["--cuda-graph-trace=" + granularity]
        else:
            notes.append("Systems graph tracing retains its tool default; legacy proof checks ordinary kernel records only; select explicit graph or node for that workload's proof")
        if args.capture:
            prefix += ["--capture-range=cudaProfilerApi", "--capture-range-end=repeat"]
        command = prefix + command
        expected_reports = [output / ("profile*.nsys-rep" if args.capture else "profile.nsys-rep")]
        artifacts["profiler"] = digest(tool)
    elif args.tool == "ncu":
        tool = executable("ncu", environ)
        replay, granularity, cache = args.replay_mode or "kernel", args.graph_profiling or "node", args.cache_control or "all"
        workload = "range" if replay in ("range", "app-range") else "graph" if granularity == "graph" else "kernel"
        settings = {"replay_mode": replay, "graph_profiling": granularity, "cache_control": cache}
        if replay == "range":
            require_stream("NCU range replay", args.command)
        if replay in ("application", "app-range"):
            notes.append("application is relaunched; execution and external file effects must permit deterministic replay")
        if replay == "app-range":
            notes.append("app-range profiles aggregate ranges; instruction-level SASS excludes JIT-compiled kernels")
        if workload != "kernel":
            notes.append("aggregate workload: launch count/skip bound workloads or ranges; no per-kernel attribution; unit-level source metrics unavailable")
        prefix = [tool, "--target-processes", "all", "--clock-control", "none", "--kernel-name-base", "demangled",
                  "--replay-mode", replay, "--graph-profiling", granularity, "--cache-control", cache,
                  "--launch-count", str(args.launch_count),
                  "--launch-skip", str(args.launch_skip), "--export", str(output / "profile")]
        if not args.section and not args.metrics:
            prefix += ["--set", "basic"]
        for section in args.section:
            prefix += ["--section", section]
        metrics = args.metrics
        if workload != "kernel":
            required = [CTA_METRIC, *([GRAPH_METRIC] if workload == "graph" else [])]
            metrics = ",".join(dict.fromkeys([*(metrics.split(",") if metrics else []), *required]))
        if metrics:
            prefix += ["--metrics", metrics]
        if replay == "application":
            prefix += ["--app-replay-match", "all", "--app-replay-mode", "strict"]
            settings.update(app_replay_match="all", app_replay_mode="strict")
        if args.kernel:
            prefix += ["--kernel-name", "regex:" + args.kernel]
        if args.capture and replay not in ("range", "app-range"):
            prefix += ["--profile-from-start", "off"]
        command = prefix + command
        expected_reports = [output / "profile.ncu-rep*"]
        artifacts["profiler"] = digest(tool)
    elif args.tool in SANITIZERS:
        tool = executable("compute-sanitizer", environ)
        prefix = [tool, "--tool", args.tool, "--error-exitcode", "99"]
        for option, value in (("padding", args.padding), ("leak-check", args.leak_check),
                              ("initcheck-address-space", args.initcheck_address_space),
                              ("unused-memory-threshold", args.unused_memory_threshold)):
            if value is not None:
                prefix += ["--" + option, str(value)]
                settings[option] = value
        if args.track_unused_memory:
            prefix += ["--track-unused-memory"]
            settings["track-unused-memory"] = True
        command = [*prefix, *command]
        artifacts["profiler"] = digest(tool)
    elif args.tool.startswith("nvbit-"):
        root = sdk_root("nvbit", environ)
        name = {"nvbit-count": "instr_count", "nvbit-graph": "instr_count_cuda_graph",
                "nvbit-memory": "mem_trace"}[args.tool]
        library = root / "tools" / name / (name + ".so")
        overrides.update(LD_PRELOAD=str(library), INSTR_BEGIN="0", INSTR_END=str(args.instruction_end or 4294967295),
                         TOOL_VERBOSE="0", MANGLED_NAMES="0", ACTIVE_FROM_START="1",
                         COUNT_WARP_LEVEL="1", EXCLUDE_PRED_OFF="0")
        artifacts["injector"] = digest(library)
        artifacts["disassembler"] = digest(executable("nvdisasm", environ))
        if args.tool in ("nvbit-count", "nvbit-memory"):
            require_stream(args.tool, args.command)
        if args.tool == "nvbit-count":
            overrides.update(START_GRID_NUM=str(args.launch_skip), END_GRID_NUM=str(args.launch_skip + args.launch_count))
            notes.append("NVBit count range is dynamic launch IDs; excluded launches still incur tool callbacks")
        elif args.tool == "nvbit-graph":
            overrides.update(START_GRID_NUM=str(args.function_skip),
                             END_GRID_NUM=str(args.function_skip + args.function_count if args.function_count is not None else 100))
            notes.append("NVBit graph bounds select first-seen unique functions, not launches; counters aggregate by function")
        if args.instruction_end is not None:
            notes.append("NVBit instruction range is partial; reported totals are not whole-program instruction counts")
        if args.capture:
            notes.append("GH_PROFILE_CAPTURE marks the target, but NVBit collection follows its own explicit instruction/function/launch bounds")
    else:
        root = sdk_root("cupti", environ)
        relative = {"cupti-trace": "cupti_trace_injection/libcupti_trace_injection.so",
                    "cupti-range": "profiling_injection/libinjection.so",
                    "cupti-pc": "pc_sampling_continuous/libpc_sampling_continuous.so"}[args.tool]
        library = root / "samples" / relative
        overrides["CUDA_INJECTION64_PATH"] = str(library)
        overrides["LD_LIBRARY_PATH"] = str(root / "lib") + (":" + environ["LD_LIBRARY_PATH"] if environ.get("LD_LIBRARY_PATH") else "")
        artifacts["injector"] = digest(library)
        for name in ("libcupti.so", "libpcsamplingutil.so", "libnvperf_host.so"):
            if (root / "lib" / name).exists():
                artifacts[name] = digest(root / "lib" / name)
        if args.tool == "cupti-trace":
            overrides["NVTX_INJECTION64_PATH"] = str(library)
        elif args.tool == "cupti-range":
            require_stream(args.tool, args.command)
            overrides["INJECTION_METRICS"] = "sm__ctas_launched.sum"
            notes.append("official range injector handles cuLaunchKernel only; no graph coverage or configurable kernel-count bound")
        else:
            overrides["INJECTION_PARAM"] = "--collection-mode 1 --sampling-period 12 --file-name pcsampling.dat --verbose"
            decoder = root / "samples/pc_sampling_utility/pc_sampling_utility"
            artifacts["pc_decoder"] = digest(decoder)
            notes.append("continuous PC sampling is statistical; a short run may legitimately produce zero samples and fail the evidence gate")
    environment = dict(environ)
    for name in INJECTIONS:
        environment.pop(name, None)
    environment.update(overrides)
    return {"argv": command, "cwd": str(output), "env_overrides": overrides,
            "env_removed": list(INJECTIONS), "environment": environment, "artifacts": artifacts,
            "expected_reports": [str(p) for p in expected_reports],
            "decoder": str(decoder) if decoder else None, "nsys_session": nsys_session,
            "workload_kind": workload, "collection_settings": settings, "notes": notes}


def process_identity(pid):
    """Linux PID plus start ticks identifies a process across reparenting/setsid."""
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                "session": int(fields[3]), "start_ticks": int(fields[19]),
                "state": fields[0], "uid": directory.stat().st_uid}
    except (OSError, ValueError, IndexError):
        return None


def same_process(expected, current):
    return current is not None and (expected["pid"], expected["start_ticks"]) == (current["pid"], current["start_ticks"])


def has_session_argument(pid, session):
    if not session:
        return False
    try:
        args = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return False
    # Match complete option values, never a shared substring or the default
    # 'profile<pid>' name. A UUID belongs only to this invocation.
    names = ("--session-name", "--session", "--session-new")
    return any(arg == name + "=" + session or
               (arg == name and index + 1 < len(args) and args[index + 1] == session)
               for index, arg in enumerate(args) for name in names)


def owned_processes(known, session=None):
    table = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity:
                table[identity["pid"]] = identity
    owned = {pid: value for pid, value in known.items() if same_process(value, table.get(pid))}
    for pid, value in table.items():
        if value["uid"] == os.getuid() and has_session_argument(pid, session):
            owned[pid] = value
    # Descendants may have their own process groups or sessions. Keep each
    # identity before killing its parent so reparenting cannot erase ownership.
    while True:
        additions = {pid: value for pid, value in table.items() if pid not in owned and value["ppid"] in owned}
        if not additions:
            return owned
        owned.update(additions)


def signal_owned(identity, sig):
    if not same_process(identity, process_identity(identity["pid"])):
        return False
    fd = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            fd = os.pidfd_open(identity["pid"])
            if not same_process(identity, process_identity(identity["pid"])):
                return False
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(identity["pid"], sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def stop_group(process, session=None, observed=None):
    """Stop owned processes across groups; PID/start-time checks avoid PID reuse."""
    identity = process_identity(process.pid)
    known = dict(observed or {})
    if identity:
        known[process.pid] = identity
    known.update(owned_processes(known, session))
    observations = {pid: value for pid, value in known.items()}
    errors = []
    sent = {signal.SIGTERM: set(), signal.SIGKILL: set()}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + 1
        while True:
            known.update(owned_processes(known, session))
            observations.update(known)
            active = [value for value in known.values()
                      if same_process(value, current := process_identity(value["pid"])) and current["state"] != "Z"]
            for value in sorted(active, key=lambda p: p["pid"] == process.pid):
                key = (value["pid"], value["start_ticks"])
                if key in sent[sig]:
                    continue
                try:
                    if signal_owned(value, sig):
                        sent[sig].add(key)
                except OSError as error:
                    errors.append({"pid": value["pid"], "signal": int(sig), "error": str(error)})
            if not active or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        process.poll()  # Reap the direct child while waiting for detached ones.
    survivors = [value for value in observations.values()
                 if same_process(value, current := process_identity(value["pid"])) and current["state"] != "Z"]
    return {"owned_processes": list(observations.values()), "survivors": survivors,
            "errors": errors, "nsys_session": session,
            "identity_check": "pid and /proc start_ticks; pidfd signals when available"}


def execute(argv, environment, cwd, timeout, stdout, stderr, owned_session=None):
    start = time.monotonic()
    with Path(stdout).open("wb", buffering=0) as out, Path(stderr).open("wb", buffering=0) as err:
        process = subprocess.Popen(argv, env=environment, cwd=cwd, stdout=out, stderr=err,
                                   start_new_session=True)
        timed_out = False
        cleanup = None
        identity = process_identity(process.pid)
        observed = {process.pid: identity} if identity else {}
        try:
            while True:
                observed.update(owned_processes(observed, owned_session))
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    timed_out = True
                    cleanup = stop_group(process, owned_session, observed)
                    break
                try:
                    process.wait(timeout=min(remaining, 0.25))
                    break
                except subprocess.TimeoutExpired:
                    pass
            if not timed_out:
                # A profiler can exit while its detached daemon or target still
                # runs. Discover exact-session processes even after reparenting.
                observed.update(owned_processes(observed, owned_session))
                if any(same_process(value, current := process_identity(value["pid"])) and current["state"] != "Z"
                       for value in observed.values()):
                    cleanup = stop_group(process, owned_session, observed)
        except BaseException:
            stop_group(process, owned_session, observed)
            raise
    return {"exit_code": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - start, "cleanup": cleanup,
            "observed_processes": list(observed.values())}


def log_lines(output, extra=()):
    for name in ("stdout.log", "stderr.log", *extra):
        path = Path(output) / name
        if path.exists():
            with path.open(errors="replace") as stream:
                yield from stream


def evidence(tool, output, expected_reports=(), decoder_logs=()):
    """Require measured records, never just an injection banner or process exit zero."""
    if expected_reports:
        reports = []
        found_all = True
        for pattern in expected_reports:
            path = Path(pattern)
            found = [p for p in path.parent.glob(path.name) if p.is_file() and p.stat().st_size
                     and p.name.endswith((".nsys-rep", ".ncu-rep", ".ncu-repz"))]
            found_all &= bool(found)
            reports.extend(digest(p) for p in sorted(found))
        return {"passed": found_all, "reports": reports,
                "reason": "nonempty profiler reports required"}
    values = []
    kernel_names = set()
    pc_records = 0
    dropped_samples = 0
    failures = []
    for line in log_lines(output, decoder_logs):
        if tool == "nvbit-graph" and "We ran out of kernel_counters" in line:
            failures.append(line.strip())
        if tool in ("nvbit-count", "nvbit-graph"):
            match = re.search(r"\bkernel instructions\s+(\d+)", line)
            if match:
                values.append(int(match[1]))
                name = re.search(r"kernel \d+ - (.*?) - #thread-blocks", line)
                if name and int(match[1]) > 0:
                    kernel_names.add(name[1])
        elif tool == "nvbit-memory":
            if re.search(r"MEMTRACE:.*\bgrid_launch_id\s+\d+.*\bwarp\s+\d+.* - 0x[0-9a-fA-F]+", line):
                values.append(1)
        elif tool in SANITIZERS:
            if re.search(r"ERROR SUMMARY: 0 errors", line) or re.search(r"RACECHECK SUMMARY: 0 hazards", line):
                values.append(1)
        elif tool == "cupti-trace":
            match = re.search(r"\b(?:CONCURRENT_KERNEL|KERNEL):\s*(\d+)\s+records", line)
            if match:
                values.append(int(match[1]))
            if re.search(r"\b(?:CONCURRENT_KERNEL|KERNEL)\b", line):
                name = re.search(r'"([^"]+)"', line)
                if name:
                    kernel_names.add(name[1])
        elif tool == "cupti-range":
            match = re.search(r"sm__ctas_launched\.sum\s+([+\d.eE-]+)\s*$", line)
            if match:
                value = float(match[1])
                if math.isfinite(value):
                    values.append(value)
        elif tool == "cupti-pc":
            match = re.search(r"Count of PC records:\s*\d+, Total Samples:\s*(\d+)", line)
            if match:
                values.append(int(match[1]))
            match = re.search(r"Count of PC records:\s*(\d+)", line)
            if match:
                pc_records += int(match[1])
            match = re.search(r"Total Dropped Samples:\s*(\d+)", line)
            if match:
                dropped_samples += int(match[1])
            match = re.search(r"functionName:\s*(.*?),\s*functionIndex:", line)
            if match:
                kernel_names.add(match[1])
    proof = {"passed": any(value > 0 for value in values) and not failures, "records": len(values),
             "positive_records": sum(value > 0 for value in values),
             "kernel_names": sorted(kernel_names),
             "failures": failures,
             "reason": "actual nonzero diagnostic records required"}
    if tool == "cupti-pc":
        files = [digest(path) for path in Path(output).glob("*_pcsampling.dat") if path.stat().st_size]
        proof["sample_files"] = files
        proof.update(pc_records=pc_records, dropped_samples=dropped_samples)
        proof["passed"] = bool(files) and proof["passed"] and pc_records > 0 and bool(kernel_names)
    return proof


def exported_kernels(tool, paths, kernel=None, workload="kernel"):
    """Read NCU raw or Systems kernel-summary CSV; headers alone are not proof."""
    names = set()
    records = 0
    ctas = 0.0
    workload_ids = set()
    graph_ids = set()
    ignored_standalone_records = 0
    matcher = re.compile(kernel) if kernel else None
    for path in paths:
        header = None
        aggregates = {}
        with Path(path).open(errors="replace", newline="") as stream:
            for row in csv.reader(stream):
                if tool == "ncu" and workload != "kernel":
                    if "ID" in row and (CTA_METRIC in row or "Metric Name" in row):
                        header = row
                        continue
                    if header is None or len(row) != len(header):
                        continue
                    values = dict(zip(header, row))
                    if not values["ID"].isdigit():
                        continue
                    identity = (values.get("Process ID", ""), values.get("Context", ""), values["ID"])
                    aggregate = aggregates.setdefault(identity, {"ctas": 0.0, "names": set(), "graphs": set()})
                    raw = values.get(CTA_METRIC)
                    if raw is None and values.get("Metric Name") == CTA_METRIC:
                        raw = values.get("Metric Value")
                    try:
                        count = float((raw or "").replace(",", ""))
                    except ValueError:
                        count = 0.0
                    if math.isfinite(count) and count > 0:
                        aggregate["ctas"] = count
                    for metric in GRAPH_IDENTITIES:
                        raw = values.get(metric)
                        if raw is None and values.get("Metric Name") == metric:
                            raw = values.get("Metric Value")
                        match = re.fullmatch(r"\+?(\d+)(?:\.0+)?", (raw or "").replace(",", ""))
                        if match and int(match[1]) > 0:
                            aggregate["graphs"].add(str(int(match[1])))
                    name = next((values[key] for key in ("Kernel Name", "Range Name", "Workload Name", "Name") if values.get(key)), "")
                    if name:
                        aggregate["names"].add(name)
                    continue
                key = "Kernel Name" if tool == "ncu" else "Name"
                if key in row and ("ID" in row if tool == "ncu" else "Instances" in row):
                    header = row
                    continue
                if header is None or len(row) != len(header):
                    continue
                values = dict(zip(header, row))
                name = values[key]
                if tool == "ncu":
                    valid = values["ID"].isdigit() and bool(name)
                    count = 1
                else:
                    number = values["Instances"].replace(",", "")
                    valid = number.isdigit() and int(number) > 0 and bool(name)
                    count = int(number) if valid else 0
                if valid and (matcher is None or matcher.search(name)):
                    names.add(name)
                    records += count
        for identity, aggregate in aggregates.items():
            if aggregate["ctas"] <= 0:
                continue
            if workload == "graph" and not aggregate["graphs"]:
                ignored_standalone_records += 1
                continue
            records += 1
            ctas += aggregate["ctas"]
            workload_ids.add(identity[-1])
            names.update(aggregate["names"])
            graph_ids.update(aggregate["graphs"])
    proof = {"passed": records > 0, "kernel_records": records if workload == "kernel" else 0,
             "kernel_names": sorted(names) if workload == "kernel" else [], "workload_kind": workload,
             "workload_records": records, "exports": [digest(path) for path in paths], "requested_kernel_regex": kernel}
    if workload != "kernel":
        proof.update(workload_names=sorted(names), workload_ids=sorted(workload_ids), ctas_launched=ctas,
                     proof_metric=CTA_METRIC, per_kernel_attribution=False)
    if workload == "graph":
        proof.update(graph_ids=sorted(graph_ids), graph_identity_metrics=list(GRAPH_IDENTITIES),
                     ignored_standalone_records=ignored_standalone_records,
                     reason="positive CTA count and positive CUDA graph identity on the same workload required")
    return proof


def exported_graphs(paths):
    """A CPU graph launch API is not proof that a GPU graph executed."""
    records, duration = 0, 0
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            continue
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
            table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            if not database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            row = database.execute(f"SELECT count(*), coalesce(sum(end-start),0) FROM {table} WHERE end>start").fetchone()
            records += row[0]
            duration += row[1]
    return {"passed": records > 0, "workload_kind": "graph", "workload_records": records,
            "device_graph_duration_ns": duration, "per_kernel_attribution": False,
            "exports": [digest(path) for path in paths if path.is_file()]}


def export_reports(tool, plan, reports, output, timeout, kernel=None):
    commands, paths = [], []
    clean_env = dict(plan["environment"])
    for name in (*INJECTIONS, "INJECTION_PARAM", "INJECTION_METRICS"):
        clean_env.pop(name, None)
    for index, report in enumerate(reports):
        stdout = output / f"export-{index}.csv"
        stderr = output / f"export-{index}.stderr.log"
        if tool == "ncu":
            command = [plan["argv"][0], "--import", report["path"], "--page", "raw", "--csv"]
        elif plan.get("workload_kind") == "graph":
            database = output / f"export-{index}.sqlite"
            stdout = output / f"export-{index}.stdout.log"
            command = [plan["argv"][0], "export", "--type", "sqlite", "--output", str(database), report["path"]]
        else:
            command = [plan["argv"][0], "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--output", "-", report["path"]]
        result = execute(command, clean_env, output, min(timeout, 60), stdout, stderr)
        commands.append({"argv": command, **result})
        paths.append(database if tool == "nsys" and plan.get("workload_kind") == "graph" else stdout)
    proof = (exported_graphs(paths) if tool == "nsys" and plan.get("workload_kind") == "graph"
             else exported_kernels(tool, paths, kernel, plan.get("workload_kind", "kernel")))
    proof["passed"] &= bool(commands) and all(c["exit_code"] == 0 and not c["timed_out"] for c in commands)
    return commands, proof


def write_manifest(path, manifest):
    # Replace only our own manifest, never historical evidence from another run.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args, environ=None):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = output
    manifest_path = output / "manifest.json"
    manifest = {"schema": "gh.profile_gpu.v1", "tool": args.tool, "diagnostic_only": True,
                "performance_ranking_valid": False, "status": "preparing", "requested_command": args.command,
                "timeout_seconds": args.timeout, "capture_requested": args.capture,
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_manifest(manifest_path, manifest)
    try:
        plan = build_plan(args, dict(os.environ if environ is None else environ))
        manifest.update({k: v for k, v in plan.items() if k != "environment"})
        manifest["status"] = "running"
        write_manifest(manifest_path, manifest)
        result = execute(plan["argv"], plan["environment"], output, args.timeout,
                         output / "stdout.log", output / "stderr.log", plan["nsys_session"])
        manifest["process"] = result
        decoder_logs = []
        decoder_ok = True
        if plan["decoder"] and not result["timed_out"] and result["exit_code"] == 0:
            manifest["decoders"] = []
            clean_env = dict(plan["environment"])
            for key in INJECTIONS:
                clean_env.pop(key, None)
            clean_env.pop("INJECTION_PARAM", None)
            clean_env.pop("INJECTION_METRICS", None)
            for index, sample in enumerate(sorted(output.glob("*_pcsampling.dat"))):
                command = [plan["decoder"], "--file-name", str(sample), "--disable-source-correlation", "--verbose"]
                stdout, stderr = f"decoder-{index}.stdout.log", f"decoder-{index}.stderr.log"
                decoded = execute(command, clean_env, output, min(args.timeout, 30), output / stdout, output / stderr)
                manifest["decoders"].append({"argv": command, **decoded})
                decoder_logs += [stdout, stderr]
                decoder_ok &= decoded["exit_code"] == 0 and not decoded["timed_out"]
        manifest["evidence"] = evidence(args.tool, output, plan["expected_reports"], decoder_logs)
        if args.tool in ("nsys", "ncu") and not result["timed_out"] and result["exit_code"] == 0:
            exports, proof = export_reports(args.tool, plan, manifest["evidence"]["reports"], output, args.timeout, args.kernel)
            manifest["exports"] = exports
            manifest["evidence"]["kernel_data"] = proof
            manifest["evidence"]["passed"] &= proof["passed"]
        cleanup_ok = not result["cleanup"] or not result["cleanup"]["survivors"]
        good = result["exit_code"] == 0 and not result["timed_out"] and cleanup_ok and decoder_ok and manifest["evidence"]["passed"]
        manifest["status"] = "passed" if good else "timed_out" if result["timed_out"] else "failed"
        code = 0 if good else 124 if result["timed_out"] else 1
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["runner_exit_code"] = code
    manifest["logs"] = {p.name: digest(p) for p in output.glob("*.log") if p.is_file()}
    write_manifest(manifest_path, manifest)
    return code


def main(argv=None):
    args = arguments(argv)
    try:
        code = run(args)
    except FileExistsError:
        print(f"refusing to overwrite existing evidence directory: {args.output}", file=sys.stderr)
        return 2
    print(f"{args.tool}: {'passed' if code == 0 else 'failed'}; evidence: {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
