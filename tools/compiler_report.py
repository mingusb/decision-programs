#!/usr/bin/env python3
"""Collect existing CUDA compiler evidence offline; never compile or launch a GPU target."""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time


def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def resources(text):
    """Parse cuobjdump resources without inventing missing zero-valued fields."""
    result, current = [], None
    for line in text.splitlines():
        found = re.match(r"\s*Function\s+(.+):\s*$", line)
        if found:
            current = {"symbol": found[1], "resources": {}}
            result.append(current)
        elif current:
            for key, value in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", line):
                current["resources"][key] = int(value)
    return result


def ptxas_records(text):
    records, current = [], None
    for line in text.splitlines():
        found = re.search(r"Function properties for (\S+)", line)
        if found:
            current = {"symbol": found[1], "stack_bytes": None,
                       "spill_store_bytes": None, "spill_load_bytes": None}
            records.append(current)
        if current:
            found = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if found:
                current.update(zip(("stack_bytes", "spill_store_bytes", "spill_load_bytes"), map(int, found.groups())))
    return records


def elf_sections(data):
    """Sizes and byte identities for executable CUDA text and constant sections."""
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected an ELF64 little-endian cubin")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    offset, entry_size, count, strings_index = header[6], header[11], header[12], header[13]
    if entry_size != 64 or not count or strings_index >= count or offset + count * entry_size > len(data):
        raise ValueError("invalid or unsupported ELF section table")
    entries = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size) for i in range(count)]
    strings = entries[strings_index]
    names = data[strings[4]:strings[4] + strings[5]]
    result = []
    for row in entries:
        if row[0] >= len(names):
            raise ValueError("invalid ELF section name offset")
        end = names.find(b"\0", row[0])
        if end < 0:
            raise ValueError("unterminated ELF section name")
        name = names[row[0]:end].decode()
        if not (name.startswith(".text.") or name.startswith(".nv.constant")):
            continue
        if row[4] + row[5] > len(data):
            raise ValueError("ELF section exceeds cubin extent")
        payload = data[row[4]:row[4] + row[5]]
        result.append({"name": name, "bytes": row[5], "sha256": hashlib.sha256(payload).hexdigest(),
                       "kind": "code" if name.startswith(".text.") else "constant"})
    return result


def trace_summary(document):
    """Chrome trace timestamps/durations are microseconds; inclusive totals can overlap."""
    events = document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("not a device compilation time trace")
    durations, starts, ends = defaultdict(float), [], []
    for event in events:
        if event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if (not isinstance(start, (int, float)) or not isinstance(duration, (int, float))
                or not math.isfinite(start) or not math.isfinite(duration) or duration < 0):
            raise ValueError("invalid trace event time")
        durations[str(event.get("name", "unnamed"))] += duration / 1000
        starts.append(start); ends.append(start + duration)
    return {"complete_event_count": len(starts),
            "complete_event_span_ms": (max(ends) - min(starts)) / 1000 if starts else None,
            "inclusive_duration_ms_by_name": dict(sorted(durations.items())),
            "duration_note": "Chrome trace microseconds converted to ms; nested/parallel totals overlap and are not build wall time"}


def ninja_costs(text):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"malformed Ninja log line {number}")
        start, end = int(fields[0]), int(fields[1])
        if end < start:
            raise ValueError("negative Ninja command duration")
        result.append({"output": fields[3], "duration_ms": end - start,
                       "command_hash": fields[4], "output_mtime": fields[2]})
    return result


def source_identity(root):
    result = {}
    excluded = {"build", "results", ".git", ".venv", "__pycache__", "node_modules"}
    suffixes = {".cu", ".cuh", ".cpp", ".hpp", ".h", ".cmake"}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix in suffixes or filename == "CMakeLists.txt":
                result[str(path.relative_to(root))] = digest(path)["sha256"]
    return result


def compare_reports(baseline, candidate):
    def sections(report):
        return {(b["name"], c["name"], s["name"]): s for b in report.get("binaries", [])
                for c in b["cubins"] for s in c["sections"]}
    old, new = sections(baseline), sections(candidate)
    keys = old.keys() & new.keys()
    changed = [{"identity": list(k), "old_bytes": old[k]["bytes"], "new_bytes": new[k]["bytes"],
                "old_sha256": old[k]["sha256"], "new_sha256": new[k]["sha256"]}
               for k in sorted(keys) if old[k] != new[k]]
    def res(report):
        return {(b["name"], c["name"], r["symbol"]): r["resources"] for b in report.get("binaries", [])
                for c in b["cubins"] for r in c["resources"]}
    a, b = res(baseline), res(candidate)
    resource_changes = [{"identity": list(k), "baseline": a[k], "candidate": b[k]}
                        for k in sorted(a.keys() & b.keys()) if a[k] != b[k]]
    return {"matched_sections": len(keys), "changed_sections": changed,
            "baseline_only": [list(k) for k in sorted(old.keys() - new.keys())],
            "candidate_only": [list(k) for k in sorted(new.keys() - old.keys())],
            "resource_changes": resource_changes,
            "exact_section_identity_match": bool(keys) and not changed and old.keys() == new.keys(),
            "scope": "Exact symbol names and code/constant section bytes only; unmatched symbols remain visible; no performance claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binary", type=Path, action="append", required=True)
    parser.add_argument("--build-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="previous report.json; exact section/resource comparison only")
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump")
    parser.add_argument("--ctadvisor", help="optional installed executable; never downloaded automatically")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    build, source, output = (p.expanduser().resolve() for p in (args.build_dir, args.source_root, args.output))
    if not build.is_dir() or not source.is_dir():
        parser.error("build/source directories must exist")
    binaries = [p.expanduser().resolve() for p in args.binary]
    if len({p.name for p in binaries}) != len(binaries) or any(not p.is_file() for p in binaries):
        parser.error("binaries must exist and have distinct basenames")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"refusing to overwrite evidence: {output}", file=sys.stderr); return 2
    report = {"schema": "gh.compiler-evidence.v1", "gpu_executed": False, "status": "collecting",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "commands": [], "binaries": [],
              "build_dir": str(build), "source_root": str(source), "runner": digest(__file__),
              "compiler_logs": [], "time_traces": [], "limitations": [
                  "Existing build evidence only; no build or GPU target is executed",
                  "Ninja command durations may include historical builds; they are not summed into build wall time",
                  "Compiler trace durations can overlap; absent build wall time is unavailable, not zero"]}

    def save():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")

    def run(name, command, cwd=None):
        out, err = output / (name + ".stdout.log"), output / (name + ".stderr.log")
        record = {"argv": list(map(str, command)), "cwd": str(cwd or output),
                  "stdout": out.name, "stderr": err.name, "timed_out": False}
        report["commands"].append(record); save()
        start = time.monotonic()
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=cwd or output, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                process.wait(timeout=args.timeout)
            except BaseException as error:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                record["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                if not record["timed_out"]: raise
        record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start)
        save()
        if process.returncode or record["timed_out"]:
            raise RuntimeError(f"offline command failed: {name}; raw logs retained")
        return out.read_text(errors="replace")

    save()
    try:
        report["source_sha256"] = source_identity(source)
        report["build_files"] = {}
        raw = output / "raw"; raw.mkdir()
        for name in ("CMakeCache.txt", "compile_commands.json", "build.ninja", ".ninja_log"):
            path = build / name
            if path.is_file():
                report["build_files"][name] = digest(path)
                shutil.copyfile(path, raw / name)
        if (build / "compile_commands.json").is_file():
            report["compile_commands"] = json.loads((build / "compile_commands.json").read_text())
        else:
            report["compile_commands"] = None
            report["limitations"].append("compile_commands.json missing; configure CMAKE_EXPORT_COMPILE_COMMANDS=ON")
        log = build / ".ninja_log"
        report["compile_cost"] = {"build_wall_ms": None, "ninja_commands": ninja_costs(log.read_text()) if log.exists() else None}
        for i, path in enumerate(args.build_log):
            path = path.expanduser().resolve()
            copied = raw / f"build-{i}.log"; shutil.copyfile(path, copied)
            report["compiler_logs"].append({**digest(path), "raw_copy": str(copied.relative_to(output)),
                                             "ptxas": ptxas_records(copied.read_text(errors="replace"))})
        trace_dir = output / "traces"; trace_dir.mkdir()
        for path in sorted(build.rglob("*.json")):
            if path == build / "compile_commands.json" or path.is_relative_to(output):
                continue
            try: document = json.loads(path.read_text())
            except (UnicodeError, json.JSONDecodeError): continue
            if not isinstance(document, dict) or "traceEvents" not in document:
                continue
            copied = trace_dir / f"{len(report['time_traces']):04d}.json"; shutil.copyfile(path, copied)
            report["time_traces"].append({**digest(path), "raw_copy": str(copied.relative_to(output)),
                                           **trace_summary(document)})
        cuobjdump = Path(args.cuobjdump).expanduser().resolve()
        report["cuobjdump"] = digest(cuobjdump)
        report["cuobjdump_version"] = run("cuobjdump-version", [str(cuobjdump), "--version"])
        cache = (build / "CMakeCache.txt").read_text() if (build / "CMakeCache.txt").exists() else ""
        report["configuration"] = {key: value for key, value in re.findall(r"^([^#/:\n][^:\n]*):[^=\n]+=(.*)$", cache, re.M)
                                   if key.startswith(("GH_", "GHB_", "CMAKE_CUDA_")) or key == "CMAKE_BUILD_TYPE"}
        compiler = re.search(r"^CMAKE_CUDA_COMPILER:[^=]+=(.+)$", cache, re.M)
        if compiler and Path(compiler[1]).is_file():
            report["compiler"] = digest(compiler[1])
            report["compiler_version"] = run("compiler-version", [compiler[1], "--version"])
            assembler = Path(compiler[1]).resolve().parent / "ptxas"
            if assembler.is_file():
                report["ptxas"] = digest(assembler)
                report["ptxas_version"] = run("ptxas-version", [str(assembler), "--version"])
        for i, binary in enumerate(binaries):
            record = {"name": binary.name, **digest(binary), "cubins": []}
            report["binaries"].append(record)
            directory = output / f"cubins-{i}"; directory.mkdir()
            run(f"binary-{i}-extract", [str(cuobjdump), "--extract-elf", "all", str(binary)], directory)
            for j, cubin in enumerate(sorted(directory.glob("*.cubin"))):
                usage = run(f"binary-{i}-cubin-{j}-resources", [str(cuobjdump), "--dump-resource-usage", str(cubin)])
                record["cubins"].append({"name": cubin.name, **digest(cubin), "sections": elf_sections(cubin.read_bytes()),
                                          "resources": resources(usage)})
            if not record["cubins"]:
                raise ValueError(f"no CUDA cubins extracted from {binary}")
            record["gpu_code_bytes"] = sum(s["bytes"] for c in record["cubins"] for s in c["sections"] if s["kind"] == "code")
            if not record["gpu_code_bytes"]:
                raise ValueError(f"no executable CUDA text sections found in {binary}")
            if digest(binary)["sha256"] != record["sha256"]:
                raise ValueError("binary changed during collection")
        advisor = args.ctadvisor or shutil.which("ctadvisor")
        report["ctadvisor"] = {"status": "unavailable" if not advisor else "no_traces"}
        if advisor:
            resolved = shutil.which(advisor) or advisor
            report["ctadvisor"]["binary"] = digest(resolved)
            if report["time_traces"]:
                run("ctadvisor", [resolved, "--trace-file-path", str(trace_dir)])
                report["ctadvisor"]["status"] = "passed"
        if args.baseline:
            report["baseline"] = digest(args.baseline)
            report["comparison"] = compare_reports(json.loads(args.baseline.read_text()), report)
        if source_identity(source) != report["source_sha256"]:
            raise ValueError("source tree changed during collection")
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["artifacts"] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                           if p.is_file() and p.name not in ("report.json", "report.tmp")}
    save()
    print(json.dumps({"status": report["status"], "output": str(output), "gpu_executed": False}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
