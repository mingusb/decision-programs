#!/usr/bin/env python3
"""Run a reproducible, serial GPU workload matrix; retain every CSV and command."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from autotune import TuningError, parse_csv, config_key, WORKLOAD, ENVIRONMENT, extract


def cells():
    result = []
    def add(n, bins, input_type, count, distribution, order, cache="warm", launch="graph"):
        item = dict(n=n, bins=bins, input=input_type, counter=count,
                    distribution=distribution, order=order, cache=cache, launch=launch, warmup_ms=0)
        if item not in result:
            result.append(item)
    # Size, skew, ordering, and the CUB non-byte privatization boundary.
    for n, bins, distribution, order, launch in itertools.product(
        (4096, 1 << 20, 1 << 24), (8, 256, 257, 4096),
        ("uniform", "hot99", "single"), ("shuffled", "sorted"), ("stream", "graph")):
        if distribution == "single" and order == "sorted":
            continue
        add(n, bins, "u32", "u32", distribution, order, launch=launch)
    # Published byte-input baseline, and independently varied local/output widths.
    for n, distribution, order, launch in itertools.product(
        (4096, 1 << 20, 1 << 24), ("uniform", "hot99"),
        ("shuffled", "sorted"), ("stream", "graph")):
        add(n, 256, "u8", "u32", distribution, order, launch=launch)
    for n, bins, distribution, launch in itertools.product(
        (1 << 20, 1 << 24), (256, 4096, 8192), ("uniform", "single"), ("stream", "graph")):
        add(n, bins, "u32", "u64", distribution, "shuffled", launch=launch)
    # Explicit cache/launch ablations on identical generated inputs.
    for n, shape, cache, launch in itertools.product(
        (1 << 20, 1 << 24), (("u32", "u32", 8), ("u32", "u32", 4096),
                                 ("u32", "u64", 4096), ("u8", "u32", 256)),
        ("warm", "cold"), ("stream", "graph")):
        kind, count, bins = shape
        add(n, bins, kind, count, "uniform", "shuffled", cache, launch)
    return result


def variants(cell, sms):
    # Predeclared representative policies; this matrix is not a per-cell full search.
    choices = [("cub", 2, 4 * sms, "native"),
               ("global", 2, 4 * sms, "native"),
               ("warp", 2, 4 * sms, "native")]
    for local in ("native", "u32") if cell["counter"] == "u64" else ("native",):
        width = 4 if local == "u32" or cell["counter"] == "u32" else 8
        if cell["bins"] * width <= 48 * 1024:
            choices += [("shared", 3, 2 * sms, local), ("shared", 2, 8 * sms, local),
                        ("shared_rle", 3, 2 * sms, local), ("shared_warp", 2, 4 * sms, local),
                        ("shared_partial", 3, 2 * sms, local)]
    if cell["bins"] <= 256:
        choices.append(("bitplane", 3, 4 * sms, "native"))
    if cell["input"] == "u8" and cell["counter"] == "u32" and cell["bins"] == 256:
        choices.append(("nvidia_sample256", 2, 4 * sms, "native"))
    clear = "kernel" if cell["launch"] == "graph" else "runtime"
    return ",".join(":".join(map(str, (*c, clear))) for c in choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/histogram_bench")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sm-count", type=int, default=48, help="physical SM count; A5000 Laptop has 48")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=991827)
    parser.add_argument("--limit", type=int, help="run only the first N shuffled cells for a smoke check")
    args = parser.parse_args()
    if (args.sm_count <= 0 or args.samples < 3 or args.batch < 1
            or args.seed < 0 or (args.limit is not None and args.limit <= 0)):
        parser.error("invalid SM count or measurement protocol")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = Path(args.exe).resolve()
    binary_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    tasks = cells()
    random.Random(852019).shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    manifest = dict(schema=1, executable=str(executable), sha256=binary_hash,
                    sm_count=args.sm_count, samples=args.samples, batch=args.batch,
                    seed=args.seed, shuffle_seed=852019, cells=tasks,
                    interpretation="Predeclared portfolio; per-cell fastest row is an ex-post oracle, not a deployable selector.")
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing campaign manifest differs; choose a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = None
    all_rows = []
    for i, cell in enumerate(tasks):
        path = output / f"cell-{i:03d}.csv"
        command = [str(executable)]
        for key, value in cell.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        command += ["--seed", str(args.seed), "--samples", str(args.samples), "--batch", str(args.batch),
                    "--variants", variants(cell, args.sm_count)]
        if not path.exists():
            print(f"[{i+1}/{len(tasks)}] {cell}", flush=True)
            started = time.monotonic()
            completed = subprocess.run(command, text=True, capture_output=True, timeout=900)
            (output / f"cell-{i:03d}.log").write_text(completed.stderr)
            (output / f"cell-{i:03d}.command.json").write_text(json.dumps(
                {"command": command, "seconds": time.monotonic() - started, "exit_code": completed.returncode}, indent=2) + "\n")
            if completed.returncode:
                raise RuntimeError(f"cell {i} failed: {completed.stderr}")
            # Parse before publishing the result so an interrupted file is not reused.
            parse_csv(completed.stdout)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(completed.stdout)
            temporary.replace(path)
        fields, rows = parse_csv(path.read_text())
        for row in rows:
            if (extract(row, WORKLOAD) != cell or row["seed"] != args.seed
                    or row["samples"] != args.samples or row["batch"] != args.batch):
                raise RuntimeError(f"cell {i} workload mismatch")
            hardware = {k: v for k, v in extract(row, ENVIRONMENT).items() if k != "eviction_bytes"}
            if environment is None:
                environment = hardware
            if hardware != environment:
                raise RuntimeError("GPU/runtime metadata changed within campaign")
        expected_keys = {tuple((p[0], int(p[1]), int(p[2]), p[3], p[4]))
                         for v in variants(cell, args.sm_count).split(",") for p in [v.split(":")]}
        if {config_key(r) for r in rows} != expected_keys or len(rows) != len(expected_keys):
            raise RuntimeError("candidate set differs from predeclared matrix")
        all_rows.extend(r["csv_row"] for r in rows)
        if hashlib.sha256(executable.read_bytes()).hexdigest() != binary_hash:
            raise RuntimeError("binary changed during campaign")
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    print(f"Completed {len(tasks)} cells and {len(all_rows)} candidate measurements.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TuningError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"campaign: {error}", file=sys.stderr)
        raise SystemExit(1)
