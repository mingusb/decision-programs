#!/usr/bin/env python3
"""Extract one exact CUDA cubin and retain bounded, offline disassembly evidence.

No GPU execution. Existing output directories are never overwritten. The exact
ELF name is required because cuobjdump itself accepts substring selectors.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elf", required=True, help="one exact name from cuobjdump --list-elf")
    parser.add_argument("--timeout", type=float, default=30, help="seconds per offline command")
    args = parser.parse_args(argv)
    if not args.binary.is_absolute() or not args.binary.is_file():
        parser.error("--binary must be an existing absolute file")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.cubin", args.elf) or args.elf.startswith("-"):
        parser.error("--elf must be an exact basename ending in .cubin")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"refusing to overwrite evidence: {output}", file=sys.stderr)
        return 2
    manifest = {"schema": "gh.offline-disassembly.v1", "gpu_executed": False,
                "status": "preparing", "elf": args.elf, "timeout_seconds_per_command": args.timeout,
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commands": []}

    def save():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "manifest.json")

    def run(name, command, stdout_name=None):
        stdout = output / (stdout_name or name + ".stdout.log")
        stderr = output / (name + ".stderr.log")
        record = {"name": name, "argv": list(map(str, command)), "cwd": str(output),
                  "stdout": stdout.name, "stderr": stderr.name, "status": "running"}
        manifest["commands"].append(record); save()
        started = time.monotonic()
        try:
            with stdout.open("xb") as out, stderr.open("xb") as err:
                process = subprocess.Popen(command, cwd=output, stdout=out, stderr=err, start_new_session=True)
                try:
                    process.wait(timeout=args.timeout)
                    record.update(exit_code=process.returncode, timed_out=False)
                except BaseException as error:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    record.update(exit_code=process.returncode, timed_out=isinstance(error, subprocess.TimeoutExpired))
                    if not isinstance(error, subprocess.TimeoutExpired):
                        raise
            record["status"] = "passed" if record["exit_code"] == 0 and not record["timed_out"] else "failed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            save()
        return record["status"] == "passed"

    save()
    try:
        binary = args.binary.resolve()
        manifest["binary"] = digest(binary)
        manifest["runner"] = digest(Path(__file__))
        tools = {}
        for name in ("cuobjdump", "nvdisasm", "dot"):
            found = shutil.which(name) or (str(Path("/usr/local/cuda/bin") / name) if name != "dot" else None)
            if not found or not Path(found).is_file():
                raise RuntimeError("required offline tool missing: " + name)
            tools[name] = str(Path(found).resolve())
        manifest["tools"] = {name: digest(Path(path)) for name, path in tools.items()}
        if not run("list-elf", [tools["cuobjdump"], "--list-elf", str(binary)]):
            raise RuntimeError("ELF listing failed; raw failure retained")
        names = re.findall(r"^ELF file\s+\d+:\s*(.+)$", (output / "list-elf.stdout.log").read_text(), re.M)
        if [name for name in names if args.elf in name] != [args.elf]:
            raise RuntimeError("ELF selector is absent or matches more than one listed cubin")
        if not run("extract-elf", [tools["cuobjdump"], "--extract-elf", args.elf, str(binary)]):
            raise RuntimeError("ELF extraction failed; raw failure retained")
        cubin = output / args.elf
        if not cubin.is_file() or cubin.stat().st_size == 0 or list(output.glob("*.cubin")) != [cubin]:
            raise RuntimeError("expected exactly the requested nonempty extracted cubin")
        manifest["cubin"] = digest(cubin)
        run("resources", [tools["cuobjdump"], "--dump-resource-usage", str(cubin)])
        run("sass", [tools["nvdisasm"], "-g", "-gi", str(cubin)], "sass.txt")
        run("register-liveness", [tools["nvdisasm"], "-plr", str(cubin)], "register-liveness.txt")
        cfg_ok = run("cfg", [tools["nvdisasm"], "-cfg", str(cubin)], "control-flow.dot")
        if cfg_ok:
            run("cfg-svg", [tools["dot"], "-Tsvg", str(output / "control-flow.dot")], "control-flow.svg")
        if digest(binary) != manifest["binary"]:
            raise RuntimeError("target binary changed during extraction")
        expected = ("resources.stdout.log", "sass.txt", "register-liveness.txt", "control-flow.dot", "control-flow.svg")
        good = all(r["status"] == "passed" for r in manifest["commands"]) and all((output / f).is_file() and (output / f).stat().st_size for f in expected)
        manifest["status"] = "passed" if good else "failed"
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["artifacts"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name not in ("manifest.json", "manifest.tmp")}
    save()
    print(json.dumps({"status": manifest["status"], "output": str(output), "gpu_executed": False}))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
