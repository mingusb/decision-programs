#!/usr/bin/env python3
"""Serial diagnostic profiles, separate from uninstrumented rankings."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent / "profiles"
EXE = ROOT / "build/window-experiment/histogram_bench"
spec = importlib.util.spec_from_file_location("recorder", ROOT / "results/a5000-profiled/run_round.py")
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)
rec.EXE = EXE


def main():
    rec.ensure_environment(BASE)
    binary_hash = rec.sha256(EXE)
    arguments = [str(EXE), "--n", "16777216", "--bins", "1048576", "--input", "u32",
                 "--counter", "u64", "--distribution", "uniform", "--order", "shuffled",
                 "--samples", "3", "--batch", "1", "--cache", "warm", "--seed", "2026092273"]
    jobs = []
    for label, algorithm, filter_name, skip, count in (
            ("narrow", "global", "global_narrow_histogram", 2, 1),
            ("window", "global_window", "global_window_histogram", 4, 2)):
        extra = ["--variants", algorithm + ":0:48:u32:kernel"]
        if label == "window": extra += ["--window-bins", "524288"]
        ncu = ["ncu", "--replay-mode", "kernel", "--cache-control", "all", "--clock-control", "none",
               "--kernel-name-base", "function", "--kernel-name", "regex:.*" + filter_name + ".*",
               "--launch-skip", str(skip), "--launch-count", str(count)]
        for section in ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
                        "SchedulerStats", "WarpStateStats", "SourceCounters"):
            ncu += ["--section", section]
        ncu += ["-o", str(BASE / label)] + arguments + ["--launch", "stream"] + extra
        jobs.append((label, ncu))
        nsys = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                "--cuda-graph-trace=node", "-o", str(BASE / (label + "-timeline"))]
        jobs.append((label + "-timeline", nsys + arguments + ["--launch", "graph"] + extra))
    manifest = dict(binary=str(EXE), binary_sha256=binary_hash,
                    script_sha256=rec.sha256(Path(__file__)), jobs=jobs,
                    limitation="Profiler replay/cache control changes execution. These are per-kernel diagnostics, not benchmark rankings. The window filter records both passes.")
    with (BASE / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2); stream.write("\n")
    for name, command in jobs:
        assert rec.sha256(EXE) == binary_hash
        rec.run(BASE / name, command, ".stdout")
        assert rec.sha256(EXE) == binary_hash
        if name.endswith("timeline"):
            rec.run(BASE / (name + "-stats"), ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
                    "--format", "csv", str(BASE / (name + ".nsys-rep"))])
        else:
            paths = list(BASE.glob(name + ".ncu-rep*"))
            assert len(paths) == 1
            rec.run(BASE / (name + "-export"), ["ncu", "--import", str(paths[0]), "--page", "details"], ".txt")
    print("All diagnostic profiles recorded.", flush=True)


if __name__ == "__main__":
    main()
