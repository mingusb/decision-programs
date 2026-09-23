"""Serialize all Compute Sanitizer workloads; retain each raw report."""
from run_diagnostic import run

if __name__ == "__main__":
    for test in ("root_histogram", "split_search"):
        for tool in ("memcheck", "racecheck", "synccheck"):
            code = run(f"{tool}-{test}", ["/usr/local/cuda/bin/compute-sanitizer", "--tool", tool,
                       "--error-exitcode", "99", f"build/booster-root-split/ghb_{test}_tests"])
            if code:
                raise SystemExit(code)
    raise SystemExit(run("memcheck-booster", ["/usr/local/cuda/bin/compute-sanitizer", "--tool", "memcheck",
                        "--error-exitcode", "99", "build/booster-root-split/ghb_booster_tests"]))
