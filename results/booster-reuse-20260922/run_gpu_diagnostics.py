"""Root-only serial GPU validation and profiles; never used for rankings."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from run_diagnostic import run

OUT = Path(__file__).resolve().parent
PREFIX = str(OUT.relative_to(OUT.parents[1]))
BUILD = "build/booster-reuse"

def checked(name, command):
    if run(name, command):
        raise SystemExit("diagnostic failed: " + name)

if __name__ == "__main__":
    sanitizer = "/usr/local/cuda/bin/compute-sanitizer"
    for suite in ("root_histogram", "split_search", "deeper_histogram"):
        for tool in ("memcheck", "racecheck", "synccheck"):
            checked(tool + "-" + suite, [sanitizer, "--tool", tool,
                "--error-exitcode", "99", f"{BUILD}/ghb_{suite}_tests"])
    checked("memcheck-booster", [sanitizer, "--tool", "memcheck",
            "--error-exitcode", "99", f"{BUILD}/ghb_booster_tests"])
    common = [f"{BUILD}/ghb_bench", "--rows", "4096", "--test-rows", "128",
        "--features", "16", "--depth", "2", "--bins", "32",
        "--tree-execution", "graph", "--output-tile", "16",
        "--tree-export-batch", "16", "--root-histogram", "batched",
        "--split-policy", "warp32"]
    for mode in ("base", "both", "deep-shared"):
        name = "nsys-" + mode
        args = common + ["--outputs", "129", "--rounds", "3", "--instrumentation", "nvtx",
            "--histogram", "shared" if mode == "deep-shared" else "global",
            "--root-counts", "per-output" if mode == "base" else "reuse-global",
            "--split-batch", "per-tree" if mode == "base" else "root",
            "--output-dir", f"{PREFIX}/{name}-benchmark"]
        checked(name, ["/usr/local/bin/nsys", "profile", "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node", "--sample=none", "--output", f"{PREFIX}/{name}"] + args)
        checked(name + "-stats", ["/usr/local/bin/nsys", "stats", "--report",
            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum", "--format", "csv",
            "--output", f"{PREFIX}/{name}-stats", f"{PREFIX}/{name}.nsys-rep"])
    ncu = "/opt/nvidia/nsight-compute/2026.3.0/ncu"
    for name, kernel in (("ncu-cached-root", "regex:.*accumulate.*"),
                         ("ncu-batched-split", "regex:.*warp_candidates.*")):
        args = common + ["--outputs", "16", "--rounds", "1", "--instrumentation", "off",
            "--histogram", "global", "--root-counts", "reuse-global", "--split-batch", "root",
            "--output-dir", f"{PREFIX}/{name}-benchmark"]
        checked(name, [ncu, "--set", "full", "--kernel-name", kernel, "--launch-count", "1",
            "--clock-control", "none", "--export", f"{PREFIX}/{name}"] + args)
        for page in ("details", "raw"):
            checked(name + "-" + page, [ncu, "--import", f"{PREFIX}/{name}.ncu-repz",
                "--page", page, "--csv"])
