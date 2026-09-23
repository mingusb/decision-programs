#!/usr/bin/env python3
"""Root-only serial native CUDA capability checks before full timing grids."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
import numpy as np
from fixtures import load, write
from campaign import IMPLEMENTATIONS, DATASETS, DATA, HERE, WORKSPACE, command

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--custom-binary", default=str(WORKSPACE / "build/booster-level-batch/ghb_real_bench"))
    p.add_argument("--output", required=True)
    p.add_argument("--implementations", choices=IMPLEMENTATIONS, nargs="+", default=IMPLEMENTATIONS)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(WORKSPACE / "build/benchmark-env/lib/python3.12/site-packages/nvidia/nccl/lib") + ":" + str(WORKSPACE / "build/benchmark-env/lib/python3.12/site-packages/nvidia/cu13/lib") + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env.update(OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    observations = []
    for dataset in DATASETS:
        small = out / dataset
        small.mkdir()
        for split, count in [("train", 512), ("validation", 128)]:
            x, y, header = load(DATA / dataset / f"{split}.ghb")
            # Every feature and every output remains present. Only rows shrink
            # for an untimed capability check, never an accuracy/speed claim.
            rng = np.random.default_rng(2026092208)
            if header["objective"] in (1, 2) and header["targets"] == 1:
                # Include every observed scalar class, then sample remaining
                # rows. Sorted source prefixes can contain just one class.
                required = np.asarray([rng.choice(np.flatnonzero(y[:, 0] == label)) for label in np.unique(y[:, 0])])
                remainder = np.setdiff1d(np.arange(len(x)), required)
                chosen = np.concatenate([required, rng.choice(remainder, size=min(count, len(x)) - len(required), replace=False)])
            else:
                chosen = rng.choice(len(x), size=min(count, len(x)), replace=False)
            chosen = np.sort(chosen)
            write(small / f"{split}.ghb", x[chosen], y[chosen], header["objective"], header["classes"])
            np.asarray(chosen, dtype="<u4").tofile(small / f"{split}-fixture-row-indices.u32")
        for implementation in a.implementations:
            directory = small / implementation
            cmd = command(implementation, {"rounds": 1, "depth": 1, "bins": 32, "learning_rate": .1, "l2": 1}, dataset, "validation", directory, Path(a.custom_binary).resolve())
            cmd[cmd.index("--train") + 1] = str(small / "train.ghb")
            cmd[cmd.index("--evaluation") + 1] = str(small / "validation.ghb")
            with (small / f"{implementation}.stdout").open("w") as stdout, (small / f"{implementation}.stderr").open("w") as stderr:
                process = subprocess.run(cmd, env=env, cwd=WORKSPACE, stdout=stdout, stderr=stderr, check=False)
            record = {"dataset": dataset, "implementation": implementation, "command": cmd, "returncode": process.returncode}
            observations.append(record)
            (out / "capabilities.json").write_text(json.dumps(observations, indent=2) + "\n")
            print(dataset, implementation, "OK" if process.returncode == 0 else "FAILED", flush=True)
    if any(r["returncode"] for r in observations):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
