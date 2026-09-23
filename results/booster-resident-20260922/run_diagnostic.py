"""Capture an explicitly supplied diagnostic command without a shell."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def run(name, command):
    record_path = OUT / (name + "-command.json")
    stdout_path, stderr_path = OUT / (name + ".stdout"), OUT / (name + ".stderr")
    if any(path.exists() for path in (record_path, stdout_path, stderr_path)):
        raise RuntimeError("refusing to overwrite diagnostic " + name)
    executable = ROOT / "build/booster-resident/ghb_bench"
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    binaries = {}
    for argument in command:
        path = Path(argument)
        path = path if path.is_absolute() else ROOT / path
        if path.is_file() and os.access(path, os.X_OK):
            binaries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    record = dict(command=command, returncode=result.returncode, wall_seconds=time.monotonic() - started,
                  trainer_executable_sha256=digest,
                  command_executables_sha256=binaries,
                  command_executables_unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in binaries.items()),
                  trainer_executable_unchanged=digest == hashlib.sha256(executable.read_bytes()).hexdigest())
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_diagnostic.py NAME COMMAND [ARG ...]")
    raise SystemExit(run(sys.argv[1], sys.argv[2:]))
