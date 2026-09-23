#!/usr/bin/env python3
"""Bounded CPU-affinity diagnostic; execution uses the GPU serially, audit does not."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("preservation", BASE / "run_preservation.py")
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    options = parser.parse_args()
    output = BASE / "preservation-affinity"
    declaration = {
        "schema": 1,
        "purpose": "CPU-affinity intervention; retain original unpinned cohorts separately.",
        "case": "stream4096", "cpu": 6, "blocks": 5,
        "order_per_block": ["old", "current", "current", "old",
                            "current", "old", "old", "current"],
        "seeds": [2026092211, 2026092212],
        "samples": 21, "batch": 32, "warmup_ms": 200,
        "runner_sha256": preservation.recorder.sha256(Path(__file__)),
        "preservation_runner_sha256": preservation.recorder.sha256(BASE / "run_preservation.py"),
        "limitations": ["Same inputs repeated; no retuning or source changes.",
                        "Affinity is inherited by benchmark children, not applied system-wide.",
                        "GPU clocks remain unlocked; Windows scheduling remains uncontrolled.",
                        "This does not replace the original acceptance evidence or prove zero loss."],
    }
    manifest = output / "manifest.json"
    if options.audit:
        if json.loads(manifest.read_text()) != declaration:
            raise ValueError("affinity declaration changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        with manifest.open("x") as stream:
            json.dump(declaration, stream, indent=2)
            stream.write("\n")
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {declaration["cpu"]})
        effective = sorted(os.sched_getaffinity(0))
        if effective != [declaration["cpu"]]:
            raise RuntimeError("CPU affinity was not applied")
        (output / "affinity.json").write_text(json.dumps({
            "previous_allowed": allowed, "effective_parent_and_inherited_children": effective,
        }, indent=2) + "\n")
    for block in range(1, declaration["blocks"] + 1):
        args = ["--cases", declaration["case"], "--output-root", str(output / f"block{block}")]
        if not options.audit:
            preservation.main(["all", "--new-exe", str(BASE.parents[1] / "build/large-bins-overflow/histogram_bench")] + args)
        preservation.main(["analyze"] + args)


if __name__ == "__main__":
    main()
