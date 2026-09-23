#!/usr/bin/env python3
"""Separate device-wide sampled memory observation; timings are not rankings."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.command:
        p.error("a command is required after --")
    command = a.command[1:] if a.command[0] == "--" else a.command
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    samples = []
    query = ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    def sample():
        start = time.perf_counter()
        result = subprocess.run(query, text=True, capture_output=True, check=False)
        samples.append({"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "query_ms": 1000 * (time.perf_counter() - start),
                        "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    def worker():
        while not stop.wait(a.interval_ms / 1000):
            sample()
    sample()
    thread = threading.Thread(target=worker)
    thread.start()
    with (out / "stdout").open("w") as stdout, (out / "stderr").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    stop.set()
    thread.join()
    sample()
    used = []
    for record in samples:
        for line in record["stdout"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4 and fields[1] == "0":
                try:
                    used.append(int(fields[2]) * (1 << 20))
                except ValueError:
                    pass
    observation = {"command": command, "returncode": result.returncode, "requested_interval_ms": a.interval_ms,
                   "scope": "device-wide memory.used, includes other processes/driver/context allocations", "samples": samples,
                   "sampled_peak_bytes": max(used) if used else None, "sampled_before_bytes": used[0] if used else None,
                   "sampled_after_bytes": used[-1] if used else None,
                   "limitations": "Lower bound on instantaneous peak; sampling can miss short allocations. This instrumented run is excluded from time rankings."}
    (out / "memory.json").write_text(json.dumps(observation, indent=2) + "\n")
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()
