"""Serial, same-binary, independent root/split experiments. Never overwrite."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXE = ROOT / "build/booster-root-split/ghb_bench"
MODES = {"base": ("per-tree", "block256"), "split": ("per-tree", "warp32"),
         "root": ("batched", "block256"), "both": ("batched", "warp32")}

def telemetry():
    command = ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv"]
    result = subprocess.run(command, capture_output=True, text=True)
    return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)

def run(name, arguments):
    directory, capture = OUT / name, OUT / (name + "-capture.json")
    if directory.exists() or capture.exists():
        raise RuntimeError("refusing to overwrite " + name)
    command = [str(EXE), *arguments, "--output-dir", str(directory)]
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    record = dict(command=command, executable_sha256=digest, before=telemetry())
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    record.update(returncode=result.returncode, process_wall_seconds=time.monotonic()-start,
                  executable_unchanged=digest == hashlib.sha256(EXE.read_bytes()).hexdigest(), after=telemetry())
    (OUT / (name + "-stdout.json")).write_bytes(result.stdout)
    (OUT / (name + "-stderr.txt")).write_bytes(result.stderr)
    capture.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode or not record["executable_unchanged"]:
        raise RuntimeError(name + " failed: " + result.stderr.decode(errors="replace"))
    data = json.loads(result.stdout)
    if data != json.loads((directory / "result.json").read_text()) or (directory / ".incomplete").exists():
        raise RuntimeError("inconsistent result " + name)
    print(name, data["timing"], flush=True)

def arguments(case):
    common = ["--tree-execution", "graph", "--tree-export-batch", "16", "--instrumentation", "off"]
    if case == "scalar":
        return common + ["--seed", "20260922601", "--rows", "65536", "--test-rows", "8192", "--features", "32", "--rounds", "10", "--depth", "5", "--bins", "64", "--histogram", "shared"]
    outputs = int(case)
    return common + ["--objective", "regression" if outputs == 129 else "binary",
            "--outputs", str(outputs), "--rows", "1024" if outputs == 4096 else "4096",
            "--test-rows", "64" if outputs == 4096 else "256", "--features", "16",
            "--rounds", "3" if outputs == 129 else "1" if outputs == 4096 else "2",
            "--depth", "2", "--bins", "32" if outputs == 129 else "16",
            "--histogram", "global", "--output-tile", "16"]

def mode_args(mode):
    root, split = MODES[mode]
    return ["--root-histogram", root, "--split-policy", split]

if __name__ == "__main__":
    for case in ("scalar", "129", "1024", "4096"):
        for suffix, modes in (("a", tuple(MODES)), ("b", tuple(reversed(MODES)))):
            for mode in modes:
                run(f"{case}-{mode}-{suffix}", arguments(case) + mode_args(mode))
    for mode in MODES:
        run(f"confirm-129-{mode}", arguments("129") + ["--seed", "20260922603"] + mode_args(mode))
    for mode in ("base", "both"):
        args = ["--objective", "multiclass", "--classes", "17", "--rows", "4096", "--test-rows", "512", "--features", "16", "--rounds", "3", "--depth", "2", "--bins", "32", "--histogram", "global", "--output-tile", "5", "--tree-export-batch", "5", "--tree-execution", "graph", "--instrumentation", "off"]
        run(f"multiclass17-{mode}", args + mode_args(mode))
