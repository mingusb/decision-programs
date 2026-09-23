"""Serial comparison against the immutable hybrid executable. No overwrites."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESIDENT = ROOT / "build/booster-resident/ghb_bench"
HYBRID = ROOT / "results/booster-trainer-20260922/baseline-provenance/build/booster/ghb_bench"


def telemetry():
    command = ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv"]
    result = subprocess.run(command, capture_output=True, text=True)
    return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def run(name, arguments, hybrid=False):
    executable = HYBRID if hybrid else RESIDENT
    directory = OUT / name
    capture = OUT / (name + "-capture.json")
    if directory.exists() or capture.exists():
        raise RuntimeError("refusing to overwrite " + name)
    command = [str(executable), *arguments, "--output-dir", str(directory)]
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    record = dict(command=command, executable_sha256=digest, before=telemetry())
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    record.update(returncode=result.returncode, process_wall_seconds=time.monotonic() - started,
                  executable_unchanged=digest == hashlib.sha256(executable.read_bytes()).hexdigest(), after=telemetry())
    (OUT / (name + "-stdout.json")).write_bytes(result.stdout)
    (OUT / (name + "-stderr.txt")).write_bytes(result.stderr)
    capture.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode or not record["executable_unchanged"]:
        raise RuntimeError(name + " failed: " + result.stderr.decode(errors="replace"))
    data = json.loads(result.stdout)
    if data != json.loads((directory / "result.json").read_text()) or (directory / ".incomplete").exists():
        raise RuntimeError("inconsistent/incomplete result " + name)
    print(name, "timing", data["timing"], "heldout", data["heldout"], flush=True)
    return data


if __name__ == "__main__":
    scalar = ["--seed", "20260922601", "--rows", "65536", "--test-rows", "8192", "--features", "32", "--rounds", "10", "--depth", "5", "--bins", "64", "--histogram", "shared"]
    # Reverse the second block to expose process-order and thermal sensitivity.
    for suffix, order in [("a", ("hybrid", "stream8", "graph8", "graph4")),
                          ("b", ("graph4", "graph8", "stream8", "hybrid"))]:
        for mode in order:
            extra = [] if mode == "hybrid" else ["--tree-execution", "stream" if mode.startswith("stream") else "graph", "--quantize-policy", "radix4" if mode.endswith("4") else "radix8"]
            run("scalar-" + mode + "-" + suffix, scalar + extra, mode == "hybrid")
    for objective in ("binary", "multiclass"):
        args = ["--objective", objective, "--rows", "32768", "--test-rows", "4096", "--features", "16", "--rounds", "10", "--depth", "4", "--bins", "64", "--classes", "5", "--histogram", "shared"]
        run(objective + "-hybrid", args, True)
        run(objective + "-resident", args + ["--tree-execution", "graph"])
    for outputs, rows, test_rows, rounds in [(129, 4096, 256, 3), (1024, 4096, 256, 2), (4096, 1024, 64, 1)]:
        args = ["--objective", "regression" if outputs == 129 else "binary", "--outputs", str(outputs), "--rows", str(rows), "--test-rows", str(test_rows), "--features", "16", "--rounds", str(rounds), "--depth", "2", "--bins", "32" if outputs == 129 else "16", "--histogram", "global", "--output-tile", "16"]
        run("outputs" + str(outputs) + "-hybrid", args, True)
        run("outputs" + str(outputs) + "-graph", args + ["--tree-execution", "graph"])
        if outputs == 1024:
            run("outputs1024-stream", args + ["--tree-execution", "stream"])
