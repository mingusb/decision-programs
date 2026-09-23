"""Diagnostic single-kernel captures, run serially after other GPU jobs."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from run_diagnostic import run

PREFIX = str(Path(__file__).resolve().parent.relative_to(Path(__file__).resolve().parents[2]))
NCU = "/opt/nvidia/nsight-compute/2026.3.0/ncu"
EXE = "build/booster-reuse/ghb_deeper_histogram_bench"
for case, variant, regex, skip in (
    (2, "global16", ".*global_accumulateILj16EE.*", 17),
    (2, "shared1-c4", ".*shared_accumulateILj1EE.*", 42),
    (6, "global16", ".*global_accumulateILj16EE.*", 17),
    (6, "shared4-c4", ".*shared_accumulateILj4EE.*", 42),
):
    name = f"ncu-deeper-{case}-{variant}"
    args = [NCU, "--set", "full", "--kernel-name-base", "mangled", "--kernel-name", "regex:" + regex,
            "--launch-skip", str(skip), "--launch-count", "1", "--clock-control", "none",
            "--cache-control", "none", "--export", f"{PREFIX}/{name}", EXE, "--case", str(case)]
    if run(name, args): raise SystemExit(name + " failed")
    for page in ("details", "raw"):
        if run(name + "-" + page, [NCU, "--import", f"{PREFIX}/{name}.ncu-repz", "--page", page, "--csv"]):
            raise SystemExit(name + " import failed")
