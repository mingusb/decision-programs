"""Serial diagnostic captures; timings are not performance ranking evidence."""
from run_diagnostic import run

OUT = "results/booster-root-split-20260922/"
EXE = "build/booster-root-split/ghb_bench"

def args(outputs, rounds, mode, instrumentation):
    return [EXE, "--rows", "4096", "--test-rows", "128", "--outputs", str(outputs),
            "--features", "16", "--rounds", str(rounds), "--depth", "2", "--bins", "32",
            "--instrumentation", instrumentation, "--histogram", "global", "--tree-execution", "graph",
            "--output-tile", "16", "--tree-export-batch", "16", "--root-histogram",
            "batched" if mode == "both" else "per-tree", "--split-policy", "warp32" if mode == "both" else "block256"]

if __name__ == "__main__":
    for mode in ("base", "both"):
        name = "nsys-" + mode
        command = ["/usr/local/bin/nsys", "profile", "--trace=cuda,nvtx,osrt", "--cuda-graph-trace=node",
                   "--sample=none", "--output", OUT + name, *args(129, 3, mode, "nvtx"), "--output-dir", OUT + name + "-benchmark"]
        if run(name, command):
            raise SystemExit(1)
        command = ["/usr/local/bin/nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum",
                   "--format", "csv", "--output", OUT + name + "-stats", OUT + name + ".nsys-rep"]
        if run(name + "-stats", command):
            raise SystemExit(1)
    for name, mode, kernel in (("ncu-global-root", "base", "global_histogram"),
                               ("ncu-batched-root", "both", "accumulate"),
                               ("ncu-block-split", "base", "split_candidates"),
                               ("ncu-warp-split", "both", "warp_candidates")):
        command = ["/opt/nvidia/nsight-compute/2026.3.0/ncu", "--set", "full", "--kernel-name", "regex:.*" + kernel + ".*",
                   "--launch-count", "1", "--clock-control", "none", "--export", OUT + name,
                   *args(16, 1, mode, "off"), "--output-dir", OUT + name + "-benchmark"]
        if run(name, command):
            raise SystemExit(1)
