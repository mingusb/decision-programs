#!/usr/bin/env python3
"""Capture, audit and compare instrumentation evidence; profiler runs are diagnostic only."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid

SCHEMA = 1
STAGES = {"quantize", "upload", "initialize", "gradients", "histogram", "histogram_subtract",
          "split_search", "route", "prediction", "evaluate", "download", "checkpoint"}
TIMINGS = {"prepare_wall_ms", "upload_wall_ms", "capture_wall_ms", "warmup_wall_ms", "submit_wall_ms",
           "completion_wall_ms", "collection_wall_ms", "device_span_ms", "operation_ms", "operation_median_ms",
           "readback_submit_wall_ms", "validation_wall_ms", "end_to_end_wall_ms", "preflight_wall_ms"}
WORKLOAD = {"rows", "features", "bins", "seed", "input", "counter", "distribution", "cache", "launch",
            "repetitions", "batch", "warmup_ms", "window_bins"}
MEMORY = {"input_bytes", "output_bytes", "scratch_bytes", "validation_snapshot_bytes",
          "free_before_bytes", "free_after_alloc_bytes"}
CONTEXT = {"round", "depth", "output", "repetition", "active_nodes", "features", "bins", "stream_id",
           "rows", "logical_read_bytes", "logical_write_bytes", "scratch_bytes", "operations"}
ENVIRONMENT = {"gpu", "sm", "driver_api", "runtime", "total_memory_bytes"}
TOP = {"schema", "kind", "benchmark", "implementation", "instrumented", "nvtx", "build", "environment",
       "workload", "validation", "timing", "memory", "samples"}
PROFILER = {"nsys": "nsys", "ncu": "ncu", "memcheck": "compute-sanitizer",
            "racecheck": "compute-sanitizer", "synccheck": "compute-sanitizer"}
PERFORMANCE_ENV = ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING", "CUDA_CACHE_DISABLE",
                   "CUDA_DEVICE_MAX_CONNECTIONS")
FINAL_FILES = {"manifest.json", "manifest.sha256", ".incomplete"}
PROCESS_KEYS = {"command", "started_utc", "finished_utc", "wall_seconds", "exit_code", "timed_out", "error"}
TELEMETRY_COMMAND = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fail(message: str):
    raise ValueError(message)


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        fail(label + ": object fields differ from schema")


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum or value > (1 << 64) - 1:
        fail(label + ": invalid integer")
    return value


def finite(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        fail(label + ": duration must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        fail(label + ": expected nonempty string")


def hash_string(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail(label + ": invalid SHA-256")


def timestamp(value, label):
    nonempty(value, label)
    parsed = dt.datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        fail(label + ": timezone required")
    return parsed


def validate_process(value, label, output=False, require_success=False):
    keys(value, PROCESS_KEYS | ({"stdout", "stderr"} if output else set()), label)
    if not isinstance(value["command"], list) or not value["command"] or any(not isinstance(arg, str) for arg in value["command"]):
        fail(label + ": invalid command")
    start = timestamp(value["started_utc"], label + ".started_utc")
    if timestamp(value["finished_utc"], label + ".finished_utc") < start:
        fail(label + ": reversed timestamps")
    finite(value["wall_seconds"], label + ".wall_seconds")
    if value["exit_code"] is not None and type(value["exit_code"]) is not int:
        fail(label + ": invalid exit code")
    if type(value["timed_out"]) is not bool or (value["error"] is not None and not isinstance(value["error"], str)):
        fail(label + ": invalid process failure fields")
    if value["exit_code"] is None and not value["error"]:
        fail(label + ": missing exit status and error")
    if value["timed_out"] and (value["exit_code"] is not None or not value["error"]):
        fail(label + ": inconsistent timeout status")
    if output and any(not isinstance(value[field], str) for field in ("stdout", "stderr")):
        fail(label + ": invalid process output")
    if require_success and (value["exit_code"] != 0 or value["timed_out"] or value["error"] is not None):
        fail(label + ": recorded process did not succeed")


def validate_result(result: dict) -> dict:
    """Validate the executable's count-component schema, not a trainer-speed claim."""
    keys(result, TOP, "result")
    if type(result["schema"]) is not int or result["schema"] != SCHEMA or result["kind"] != "ghb.instrumentation":
        fail("unsupported result schema/kind")
    if result["benchmark"] != "count_histogram_component":
        fail("unsupported benchmark contract")
    nonempty(result["implementation"], "implementation")
    for flag in ("instrumented", "nvtx"):
        if type(result[flag]) is not bool:
            fail(flag + ": expected bool")
    if result["nvtx"] and not result["instrumented"]:
        fail("NVTX requires enabled instrumentation")
    implementation = result["implementation"].split(":")
    if (len(implementation) != 5 or implementation[0] not in ("global_window", "global") or
            implementation[3:] != ["u32", "kernel"] or any(not value.isascii() or not value.isdecimal() for value in implementation[1:3])):
        fail("unsupported implementation policy")
    tuning, blocks = map(int, implementation[1:3])
    if tuning > 5 or not 1 <= blocks <= (1 << 31) - 1 or implementation[1:3] != [str(tuning), str(blocks)]:
        fail("implementation tuning/block bounds or representation differ from producer contract")
    keys(result["build"], {"id"}, "build")
    hash_string(result["build"]["id"], "build.id")
    env = result["environment"]
    keys(env, ENVIRONMENT, "environment")
    nonempty(env["gpu"], "environment.gpu")
    for field in ENVIRONMENT - {"gpu"}:
        integer(env[field], "environment." + field, 1)
    work = result["workload"]
    keys(work, WORKLOAD, "workload")
    for field in ("rows", "features", "bins", "repetitions", "batch", "window_bins"):
        integer(work[field], "workload." + field, 1)
    integer(work["seed"], "workload.seed")
    integer(work["warmup_ms"], "workload.warmup_ms")
    if (work["rows"] > (1 << 32) - 1 or work["bins"] >= (1 << 31) - 1 or work["window_bins"] >= (1 << 31) - 1
            or work["repetitions"] > 4096 or work["batch"] > 4096 or work["warmup_ms"] > 60000
            or work["bins"] * 8 * work["repetitions"] > (512 << 20)):
        fail("workload exceeds producer measurement bounds")
    if work["features"] != 1 or work["input"] != "u32" or work["counter"] != "u64":
        fail("count component requires one feature and u32 input/u64 output")
    if work["cache"] != "warm" or work["distribution"] != "uniform" or work["launch"] not in ("stream", "graph"):
        fail("unsupported cache/launch protocol")
    valid = result["validation"]
    keys(valid, {"passed", "checked_bins", "checked_repetitions", "mismatched_bins"}, "validation")
    if valid["passed"] is not True:
        fail("validation did not pass")
    for field in ("checked_bins", "checked_repetitions", "mismatched_bins"):
        integer(valid[field], "validation." + field)
    if valid["checked_bins"] != work["bins"] or valid["checked_repetitions"] != work["repetitions"] or valid["mismatched_bins"] != 0:
        fail("validation coverage or mismatch counts disagree with workload")
    timing = result["timing"]
    keys(timing, TIMINGS, "timing")
    for field in set(timing) - {"operation_ms"}:
        finite(timing[field], "timing." + field, field in ("device_span_ms", "operation_median_ms"))
    operations = timing["operation_ms"]
    if not isinstance(operations, list) or len(operations) != work["repetitions"]:
        fail("operation timing count differs from repetitions")
    for value in operations:
        finite(value, "operation_ms", True)
    if not math.isclose(timing["operation_median_ms"], statistics.median(operations), rel_tol=1e-7, abs_tol=1e-10):
        fail("reported operation median differs from raw operations")
    memory = result["memory"]
    keys(memory, MEMORY, "memory")
    for field, value in memory.items():
        integer(value, "memory." + field)
    if (memory["input_bytes"] != work["rows"] * 4 or memory["output_bytes"] != work["bins"] * 8 or
            memory["validation_snapshot_bytes"] != work["bins"] * 8 * work["repetitions"]):
        fail("memory sizes disagree with input/output/snapshot contract")
    scratch_bins = min(work["bins"], work["window_bins"]) if implementation[0] == "global_window" else work["bins"]
    if memory["scratch_bytes"] != scratch_bins * 4:
        fail("scratch bytes disagree with implementation/workload")
    if memory["free_before_bytes"] > env["total_memory_bytes"] or memory["free_after_alloc_bytes"] > env["total_memory_bytes"]:
        fail("free device memory exceeds reported device capacity")
    samples = result["samples"]
    if not isinstance(samples, list) or (not result["instrumented"] and samples):
        fail("uninstrumented mode must have an empty sample list")
    ids, histogram_repetitions = set(), set()
    for sample in samples:
        keys(sample, {"id", "stage", "timing", "context", "host_start_ns", "host_end_ns", "gpu_ms"}, "sample")
        integer(sample["id"], "sample.id")
        if sample["id"] in ids:
            fail("duplicate sample ID")
        ids.add(sample["id"])
        if sample["stage"] not in STAGES or sample["timing"] not in ("host", "gpu"):
            fail("unknown sample stage/timing")
        for field in ("host_start_ns", "host_end_ns"):
            integer(sample[field], "sample." + field)
        if sample["host_end_ns"] < sample["host_start_ns"]:
            fail("sample host timestamps reversed")
        if sample["timing"] == "host":
            if sample["gpu_ms"] is not None:
                fail("host sample must not contain GPU duration")
        else:
            finite(sample["gpu_ms"], "sample.gpu_ms")
        context = sample["context"]
        keys(context, CONTEXT, "sample.context")
        for field, value in context.items():
            integer(value, "sample.context." + field, -1 if field in ("round", "depth", "output", "repetition") else (1 if field == "operations" else 0))
        if sample["stage"] == "histogram" and sample["timing"] == "gpu":
            rep = context["repetition"]
            if rep not in range(work["repetitions"]) or rep in histogram_repetitions:
                fail("duplicate or invalid histogram repetition")
            histogram_repetitions.add(rep)
            if (context["rows"] != work["rows"] or context["features"] != 1 or context["bins"] != work["bins"]
                    or context["operations"] != work["batch"] or context["scratch_bytes"] != memory["scratch_bytes"]):
                fail("histogram sample context disagrees with workload/memory")
            scans = (work["bins"] + work["window_bins"] - 1) // work["window_bins"] if implementation[0] == "global_window" else 1
            if (context["logical_read_bytes"] != memory["input_bytes"] * scans * work["batch"] or
                    context["logical_write_bytes"] != memory["output_bytes"] * work["batch"]):
                fail("histogram declared logical traffic disagrees with workload/batch")
    if result["instrumented"] and histogram_repetitions != set(range(work["repetitions"])):
        fail("instrumented samples do not cover every histogram repetition")
    # Inner recorder intervals and outer operation event intervals are distinct
    # observables. Do not assert equality or add their durations together.
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("duplicate JSON object key: " + key)
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: fail("nonfinite JSON constant: " + value))


def read_json(path: Path):
    return parse_json(Path(path).read_text())


def atomic_bytes(path: Path, content: bytes):
    """A complete new artifact or no target artifact; never overwrite evidence."""
    if path.exists():
        fail("refusing overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if the target already exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def text_value(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


def run_process(command: list[str], timeout: float, cwd: Path) -> dict:
    started, begin = utc(), time.monotonic()
    try:
        completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return {"command": command, "started_utc": started, "finished_utc": utc(),
                "wall_seconds": time.monotonic() - begin, "exit_code": completed.returncode,
                "timed_out": False, "error": None,
                "stdout": text_value(completed.stdout), "stderr": text_value(completed.stderr)}
    except (subprocess.TimeoutExpired, OSError) as error:
        return {"command": command, "started_utc": started, "finished_utc": utc(),
                "wall_seconds": time.monotonic() - begin, "exit_code": None,
                "timed_out": isinstance(error, subprocess.TimeoutExpired), "error": str(error),
                "stdout": text_value(getattr(error, "stdout", "")), "stderr": text_value(getattr(error, "stderr", ""))}


def telemetry(cwd: Path) -> dict:
    return run_process(list(TELEMETRY_COMMAND), 10.0, cwd)


def telemetry_identity(record):
    """Optional stable inventory identity; raw clocks remain uninterpreted."""
    if record["exit_code"] != 0 or record["error"] is not None or record["timed_out"]:
        return None
    try:
        rows = list(csv.reader(record["stdout"].splitlines(), strict=True))
    except csv.Error:
        return None
    if not rows or any(len(row) != 7 or any(not value.strip() for value in row[:3]) for row in rows):
        return None
    return sorted(tuple(value.strip() for value in row[:3]) for row in rows)


def snapshot_file(path: Path, output: Path, relative: str) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file():
        fail("provenance input is not a file: " + str(path))
    before = sha256(path)
    target = output / relative
    atomic_bytes(target, path.read_bytes())
    if sha256(target) != before:
        fail("provenance source changed while snapshotting: " + str(path))
    return {"path": str(path), "snapshot": relative, "sha256_before": before, "sha256_after": None}


def build_files(exe: Path) -> list[Path]:
    return [exe.parent / name for name in ("CMakeCache.txt", "build.ninja", "Makefile", "compile_commands.json")
            if (exe.parent / name).is_file()]


def artifact_records(output: Path) -> list[dict]:
    result = []
    for path in sorted(output.rglob("*")):
        if path.parent == output and path.name in FINAL_FILES:
            continue
        if path.is_symlink():
            fail("symlink in evidence directory: " + str(path))
        if path.is_file():
            result.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return result


def profile_command(tool: str, executable: str, binary: Path, arguments: list[str], output: Path) -> list[str]:
    if tool == "nsys":
        return [executable, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--output=" + str(output / "profile"), str(binary), *arguments]
    if tool == "ncu":
        # Bound diagnostic collection, including when the binary has a long
        # warmup. The first two launches may be setup/preflight kernels; this
        # trace is neither full stage coverage nor a steady-state ranking.
        return [executable, "--target-processes", "all", "--set", "basic", "--launch-count", "2",
                "--clock-control", "none", "--export", str(output / "profile"), str(binary), *arguments]
    return [executable, "--tool", tool, "--error-exitcode", "99", "--log-file", str(output / "profile.log"), str(binary), *arguments]


def capture(exe: Path, output: Path, arguments: list[str], timeout=3600.0, sources=(), tool: str | None = None) -> dict:
    exe = exe.resolve(strict=True)
    output = output.resolve()
    if not exe.is_file():
        fail("executable is not a file")
    finite(timeout, "timeout", True)
    paths = [Path(path).resolve(strict=True) for path in sources]
    if len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        fail("sources must be distinct regular files")
    cwd = Path.cwd().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    atomic_bytes(output / ".incomplete", b"Observation is not finalized.\n")
    manifest = {"schema": SCHEMA, "kind": "ghb.profile" if tool else "ghb.observation", "status": "failed",
                "created_utc": utc(), "finished_utc": None, "measurement_role": "diagnostic_only" if tool else "count_component_observation",
                "protocol": {"cwd": str(cwd), "output_directory": str(output), "timeout_seconds": timeout, "arguments": list(arguments)},
                "binary": None, "sources": [], "build_files": [], "runner": None, "tool": None,
                "process": None, "result": None, "error": None, "artifacts": []}
    snapshots = []
    try:
        manifest["binary"] = snapshot_file(exe, output, "provenance/executable")
        snapshots.append(manifest["binary"])
        manifest["runner"] = snapshot_file(Path(__file__), output, "provenance/observe.py")
        snapshots.append(manifest["runner"])
        for index, path in enumerate(paths):
            record = snapshot_file(path, output, f"provenance/sources/{index:03d}-{path.name}")
            manifest["sources"].append(record)
            snapshots.append(record)
        for index, path in enumerate(build_files(exe)):
            record = snapshot_file(path, output, f"provenance/build/{index:03d}-{path.name}")
            manifest["build_files"].append(record)
            snapshots.append(record)
        environment = {"schema": SCHEMA, "platform": dict(platform.uname()._asdict()),
                       "python": sys.version, "performance_environment": {name: os.environ.get(name) for name in PERFORMANCE_ENV},
                       "gpu_before": telemetry(cwd), "gpu_after": None}
        if tool:
            executable = shutil.which(PROFILER[tool]) or PROFILER[tool]
            version = run_process([executable, "--version"], 10.0, cwd)
            manifest["tool"] = {"name": tool, "executable": executable, "version_record": "tool-version.json"}
            write_json(output / "tool-version.json", version)
            command = profile_command(tool, executable, exe, arguments, output)
        else:
            command = [str(exe), *arguments]
        write_json(output / "command.json", command)
        process = run_process(command, timeout, cwd)
        environment["gpu_after"] = telemetry(cwd)
        write_json(output / "environment.json", environment)
        atomic_bytes(output / "stdout.txt", process.pop("stdout").encode())
        atomic_bytes(output / "stderr.txt", process.pop("stderr").encode())
        manifest["process"] = process
        if process["exit_code"] != 0 or process["timed_out"] or process["error"]:
            fail("observed process failed: " + str(process["exit_code"]) + (" (timeout)" if process["timed_out"] else ""))
        if tool:
            raw_name = "profile.nsys-rep" if tool == "nsys" else "profile.ncu-rep" if tool == "ncu" else "profile.log"
            if not (output / raw_name).is_file() or not (output / raw_name).stat().st_size:
                fail("diagnostic tool did not produce its nonempty raw artifact: " + raw_name)
        else:
            result = parse_json((output / "stdout.txt").read_text())
            write_json(output / "result.json", result)
            manifest["result"] = "result.json"
            validate_result(result)
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = type(error).__name__ + ": " + str(error)
    finally:
        for record in snapshots:
            path = Path(record["path"])
            record["sha256_after"] = sha256(path) if path.is_file() else None
            if record["sha256_after"] != record["sha256_before"]:
                manifest["status"] = "failed"
                manifest["error"] = "provenance input changed during observation: " + record["path"]
        manifest["finished_utc"] = utc()
        manifest["artifacts"] = artifact_records(output)
        write_json(output / "manifest.json", manifest)
        atomic_bytes(output / "manifest.sha256", (sha256(output / "manifest.json") + "\n").encode())
        (output / ".incomplete").unlink()
    return manifest


def safe_artifact(output: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative or relative == "."):
        fail("unsafe artifact path")
    path = output / relative
    if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
        fail("unsafe artifact symlink/path")
    return path


def audit(output: Path) -> dict:
    output = Path(output).resolve()
    if (output / ".incomplete").exists():
        fail("observation was not finalized")
    if (output / "manifest.json").is_symlink() or (output / "manifest.sha256").is_symlink():
        fail("manifest/checksum must not be symlinks")
    checksum = (output / "manifest.sha256").read_text()
    if checksum != sha256(output / "manifest.json") + "\n":
        fail("manifest hash mismatch")
    manifest = read_json(output / "manifest.json")
    expected_keys = {"schema", "kind", "status", "created_utc", "finished_utc", "measurement_role", "protocol", "binary",
                     "sources", "build_files", "runner", "tool", "process", "result", "error", "artifacts"}
    keys(manifest, expected_keys, "manifest")
    if type(manifest["schema"]) is not int or manifest["schema"] != SCHEMA or manifest["kind"] not in ("ghb.observation", "ghb.profile"):
        fail("unsupported observation manifest")
    if manifest["status"] != "passed" or manifest["error"] is not None:
        fail("observation failed: " + str(manifest["error"]))
    if timestamp(manifest["finished_utc"], "finished_utc") < timestamp(manifest["created_utc"], "created_utc"):
        fail("manifest timestamps reversed")
    seen = set()
    if not isinstance(manifest["artifacts"], list):
        fail("artifact records must be a list")
    for record in manifest["artifacts"]:
        keys(record, {"path", "sha256", "size_bytes"}, "artifact")
        path = safe_artifact(output, record["path"])
        if record["path"] in seen or record["path"] in FINAL_FILES:
            fail("duplicate or reserved artifact")
        seen.add(record["path"])
        hash_string(record["sha256"], "artifact hash")
        integer(record["size_bytes"], "artifact size")
        if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            fail("missing or changed artifact: " + record["path"])
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*")
              if path.is_file() and not (path.parent == output and path.name in FINAL_FILES)}
    if actual != seen or any(path.is_symlink() for path in output.rglob("*")):
        fail("artifact set differs from finalized manifest")
    if not {"command.json", "stdout.txt", "stderr.txt", "environment.json"}.issubset(seen):
        fail("missing required capture artifacts")
    if not isinstance(manifest["sources"], list) or not isinstance(manifest["build_files"], list):
        fail("malformed provenance lists")
    provenance_snapshots = set()
    for record in [manifest["binary"], manifest["runner"], *manifest["sources"], *manifest["build_files"]]:
        keys(record, {"path", "snapshot", "sha256_before", "sha256_after"}, "provenance")
        if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
            fail("provenance original path must be absolute")
        hash_string(record["sha256_before"], "provenance hash")
        if record["sha256_before"] != record["sha256_after"]:
            fail("provenance changed during observation")
        if record["snapshot"] in provenance_snapshots or record["snapshot"] not in seen or sha256(safe_artifact(output, record["snapshot"])) != record["sha256_before"]:
            fail("provenance snapshot hash mismatch")
        provenance_snapshots.add(record["snapshot"])
    protocol = manifest["protocol"]
    keys(protocol, {"cwd", "output_directory", "timeout_seconds", "arguments"}, "protocol")
    for field in ("cwd", "output_directory"):
        if not isinstance(protocol[field], str) or not Path(protocol[field]).is_absolute():
            fail("protocol paths must be absolute")
    finite(protocol["timeout_seconds"], "protocol timeout", True)
    if not isinstance(protocol["arguments"], list) or any(not isinstance(arg, str) for arg in protocol["arguments"]):
        fail("invalid exact argument list")
    process = manifest["process"]
    validate_process(process, "process", require_success=True)
    binary = Path(manifest["binary"]["path"])
    if manifest["kind"] == "ghb.profile":
        if manifest["measurement_role"] != "diagnostic_only" or manifest["result"] is not None:
            fail("profile must remain diagnostic-only")
        tool = manifest["tool"]
        keys(tool, {"name", "executable", "version_record"}, "tool")
        if tool["name"] not in PROFILER or tool["version_record"] not in seen:
            fail("missing profiler identity/version record")
        nonempty(tool["executable"], "tool.executable")
        version = read_json(safe_artifact(output, tool["version_record"]))
        validate_process(version, "tool version", output=True)
        if version["command"] != [tool["executable"], "--version"]:
            fail("incorrect tool version command")
        raw_name = "profile.nsys-rep" if tool["name"] == "nsys" else "profile.ncu-rep" if tool["name"] == "ncu" else "profile.log"
        if raw_name not in seen or not (output / raw_name).stat().st_size:
            fail("missing nonempty raw diagnostic artifact")
        expected_command = profile_command(tool["name"], tool["executable"], binary, protocol["arguments"], Path(protocol["output_directory"]))
        result = None
    else:
        if manifest["measurement_role"] != "count_component_observation" or manifest["tool"] is not None or manifest["result"] != "result.json" or "result.json" not in seen:
            fail("invalid count observation role/result")
        expected_command = [str(binary), *protocol["arguments"]]
        result = validate_result(read_json(output / "result.json"))
        if parse_json((output / "stdout.txt").read_text()) != result:
            fail("parsed result differs from raw executable stdout")
    if read_json(output / "command.json") != expected_command or process["command"] != expected_command:
        fail("exact command differs from protocol")
    environment = read_json(output / "environment.json")
    keys(environment, {"schema", "platform", "python", "performance_environment", "gpu_before", "gpu_after"}, "captured environment")
    if type(environment["schema"]) is not int or environment["schema"] != SCHEMA:
        fail("invalid captured environment")
    keys(environment["platform"], {"system", "node", "release", "version", "machine", "processor"}, "platform")
    if any(not isinstance(value, str) for value in environment["platform"].values()):
        fail("platform fields must be strings")
    nonempty(environment["python"], "Python version")
    keys(environment["performance_environment"], PERFORMANCE_ENV, "performance environment")
    if any(value is not None and not isinstance(value, str) for value in environment["performance_environment"].values()):
        fail("performance environment values must be strings or null")
    for field in ("gpu_before", "gpu_after"):
        validate_process(environment[field], field, output=True)
        if environment[field]["command"] != TELEMETRY_COMMAND:
            fail("unexpected GPU telemetry command")
    return {"schema": SCHEMA, "kind": manifest["kind"], "passed": True,
            "manifest": {"path": str(output / "manifest.json"), "sha256": sha256(output / "manifest.json")},
            "manifest_content": manifest, "environment": environment, "result": result}


def ratio(numerator, denominator):
    if denominator <= 0:
        return None
    value = numerator / denominator
    return finite(value, "comparison ratio")


def comparison_arguments(arguments):
    """Only implementation/instrumentation selector arguments may differ."""
    result, index = [], 0
    while index < len(arguments):
        if arguments[index] in ("--variant", "--tuning", "--blocks", "--instrumentation"):
            if index + 1 == len(arguments):
                fail("incomplete comparison selector argument")
            index += 2
        else:
            result.append(arguments[index])
            index += 1
    return result


def compare(left: Path, right: Path) -> dict:
    a, b = audit(left), audit(right)
    if a["kind"] != "ghb.observation" or b["kind"] != "ghb.observation":
        fail("profiler/sanitizer runs are diagnostic and cannot be used for ranking comparisons")
    x, y = a["result"], b["result"]
    for field in ("schema", "kind", "benchmark", "workload", "environment", "validation"):
        if x[field] != y[field]:
            fail("comparison protocol mismatch: " + field)
    for field in ("platform", "performance_environment"):
        if a["environment"][field] != b["environment"][field]:
            fail("comparison environment mismatch: " + field)
    for field in ("cwd", "timeout_seconds"):
        if a["manifest_content"]["protocol"][field] != b["manifest_content"]["protocol"][field]:
            fail("comparison recording protocol mismatch: " + field)
    if comparison_arguments(a["manifest_content"]["protocol"]["arguments"]) != comparison_arguments(b["manifest_content"]["protocol"]["arguments"]):
        fail("comparison binary argument protocol mismatch")
    identities = [[telemetry_identity(item["environment"][field]) for field in ("gpu_before", "gpu_after")] for item in (a, b)]
    available = [identity for pair in identities for identity in pair if identity is not None]
    if available and (len(available) != 4 or any(identity != available[0] for identity in available[1:])):
        fail("comparison optional GPU identity coverage or inventory mismatch")
    mode_changed = (x["instrumented"], x["nvtx"]) != (y["instrumented"], y["nvtx"])
    implementation_changed = x["implementation"] != y["implementation"]
    comparison_kind = ("mixed_instrumentation_and_implementation" if mode_changed and implementation_changed else
                       "instrumentation_overhead" if mode_changed else "count_component_comparison")
    return {"schema": SCHEMA, "kind": "ghb.observation_comparison", "comparison_kind": comparison_kind,
            "left": a["manifest"], "right": b["manifest"], "ratio_direction": "right / left; above one means longer duration",
            "workload": x["workload"], "environment": x["environment"],
            "left_mode": {field: x[field] for field in ("implementation", "instrumented", "nvtx", "build")},
            "right_mode": {field: y[field] for field in ("implementation", "instrumented", "nvtx", "build")},
            "raw_operation_ms": {"left": x["timing"]["operation_ms"], "right": y["timing"]["operation_ms"]},
            "right_over_left": {field: ratio(y["timing"][field], x["timing"][field]) for field in
                                ("operation_median_ms", "device_span_ms", "submit_wall_ms", "completion_wall_ms", "end_to_end_wall_ms")},
            "instrumented_over_uninstrumented_operation_ratio":
                (y["timing"]["operation_median_ms"] / x["timing"]["operation_median_ms"] if y["instrumented"] else
                 x["timing"]["operation_median_ms"] / y["timing"]["operation_median_ms"]) if x["instrumented"] != y["instrumented"] else None,
            "limitations": [
                "These are count-component observations, not weighted training histograms or end-to-end trainer performance.",
                "An instrumentation-mode difference measures observed overhead under the recorded protocol; near-one ratios do not prove universal zero cost.",
                "Different implementations or builds can confound attribution of an instrumentation difference.",
                "Comparisons require identical declared input workload/seed and validation coverage; the producer does not supply an input-content digest.",
                "GPU identity is checked through producer metadata and optional nvidia-smi inventory when available; clocks and profiler durations are not interpreted as performance claims.",
                "Raw repetition arrays are retained. No confidence interval, outlier deletion, fastest-ever claim, or zero-loss guarantee is produced.",
                "Recorder GPU stage intervals and outer operation timing have different boundaries; they are not added together.",
            ]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "profile"):
        command = sub.add_parser(action)
        command.add_argument("--exe", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--timeout", type=float, default=3600.0)
        command.add_argument("--source", type=Path, action="append", default=[])
        if action == "profile":
            command.add_argument("--tool", choices=PROFILER, required=True)
        command.add_argument("binary_arguments", nargs=argparse.REMAINDER)
    command = sub.add_parser("audit")
    command.add_argument("directory", type=Path)
    command = sub.add_parser("compare")
    command.add_argument("left", type=Path)
    command.add_argument("right", type=Path)
    options = parser.parse_args(argv)
    try:
        if options.action in ("capture", "profile"):
            arguments = options.binary_arguments
            if arguments and arguments[0] == "--":
                arguments = arguments[1:]
            report = capture(options.exe, options.output_dir, arguments, options.timeout, options.source, getattr(options, "tool", None))
            print(json.dumps({"status": report["status"], "manifest": str(options.output_dir.resolve() / "manifest.json"),
                              "error": report["error"]}, indent=2))
            return 0 if report["status"] == "passed" else 1
        report = audit(options.directory) if options.action == "audit" else compare(options.left, options.right)
        if options.action == "audit":
            report = {field: report[field] for field in ("schema", "kind", "passed", "manifest")}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
