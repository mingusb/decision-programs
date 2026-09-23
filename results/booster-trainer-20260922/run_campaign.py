"""Reproduce the serial first-trainer campaign; every case directory must be new."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXE = ROOT / "build/booster/ghb_bench"


def telemetry():
    command = ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv"]
    result = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run(name, arguments):
    directory = OUT / name
    if directory.exists() or (OUT / (name + "-capture.json")).exists():
        raise RuntimeError("refusing to overwrite " + name)
    command = [str(EXE), *arguments, "--output-dir", str(directory)]
    before = hashlib.sha256(EXE.read_bytes()).hexdigest()
    record = {"command": command, "executable_sha256": before, "before": telemetry()}
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    record.update(returncode=result.returncode, process_wall_seconds=time.monotonic() - start,
                  executable_unchanged=before == hashlib.sha256(EXE.read_bytes()).hexdigest(), after=telemetry())
    (OUT / (name + "-stdout.json")).write_bytes(result.stdout)
    (OUT / (name + "-stderr.txt")).write_bytes(result.stderr)
    (OUT / (name + "-capture.json")).write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode or not record["executable_unchanged"]:
        raise RuntimeError(name + " failed: " + result.stderr.decode(errors="replace"))
    measured = json.loads(result.stdout)
    if measured != json.loads((directory / "result.json").read_text()) or (directory / ".incomplete").exists():
        raise RuntimeError("inconsistent or incomplete result: " + name)
    print(name, "train_ms", round(measured["timing"]["training_ms"], 3),
          "loss", measured["heldout"], "memory", measured["memory"], flush=True)


if __name__ == "__main__":
    base = ["--seed", "20260922601", "--rows", "65536", "--test-rows", "8192", "--features", "32", "--rounds", "10", "--depth", "5", "--bins", "64"]
    for name, policy in [("regression-global-a", "global"), ("regression-shared-a", "shared"),
                         ("regression-shared-b", "shared"), ("regression-global-b", "global")]:
        run(name, [*base, "--histogram", policy])
    run("regression-auto-stages", [*base, "--histogram", "auto", "--instrumentation", "nvtx"])
    classification = ["--rows", "32768", "--test-rows", "4096", "--features", "16", "--rounds", "10", "--depth", "4", "--bins", "64"]
    run("binary", [*classification, "--objective", "binary", "--instrumentation", "timing"])
    run("multiclass", [*classification, "--objective", "multiclass", "--classes", "5", "--instrumentation", "timing"])
    run("regression-129", ["--objective", "regression", "--outputs", "129", "--output-tile", "16", "--rows", "4096", "--test-rows", "256", "--features", "16", "--rounds", "3", "--depth", "2", "--bins", "32", "--histogram", "global"])
    wide = ["--objective", "binary", "--outputs", "1024", "--rows", "4096", "--test-rows", "256", "--features", "16", "--rounds", "2", "--depth", "2", "--bins", "16", "--histogram", "global"]
    run("multilabel-1024-tile16", [*wide, "--output-tile", "16"])
    run("multilabel-1024-tile1024", [*wide, "--output-tile", "1024"])
    run("multilabel-4096", ["--objective", "binary", "--outputs", "4096", "--output-tile", "16", "--rows", "1024", "--test-rows", "64", "--features", "16", "--rounds", "1", "--depth", "2", "--bins", "16", "--histogram", "global"])
