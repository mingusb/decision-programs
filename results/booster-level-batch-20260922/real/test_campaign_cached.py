#!/usr/bin/env python3
"""Root-only serial selected-test campaign; original validation runner is untouched."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from campaign import HERE, WORKSPACE, DATA, DATASETS, IMPLEMENTATIONS, digest, command, selected

def execute(name, implementation, config, dataset, root, binary):
    directory = root / name
    directory.mkdir()
    cmd = command(implementation, config, dataset, "test", directory / "result", binary)
    sources = ["campaign.py", "test_campaign_cached.py", "framework.py", "fixtures.py", "evaluate.py", "evaluate_cached.py"]
    capture = {"name": name, "implementation": implementation, "dataset": dataset, "split": "test", "config": config,
               "command": cmd, "cwd": str(WORKSPACE), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "binary_sha256": digest(binary) if implementation.startswith("custom-") else None,
               "source_sha256": {name: digest(HERE / name) for name in sources},
               "training_fixture_sha256": digest(DATA / dataset / "train.ghb"), "evaluation_fixture_sha256": digest(DATA / dataset / "test.ghb")}
    capture_path = directory / "capture.json"
    capture_path.write_text(json.dumps(capture, indent=2) + "\n")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    nvidia = WORKSPACE / "build/benchmark-env/lib/python3.12/site-packages/nvidia"
    env["LD_LIBRARY_PATH"] = str(nvidia / "nccl/lib") + ":" + str(nvidia / "cu13/lib") + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    start = time.perf_counter()
    with (directory / "stdout").open("w") as out, (directory / "stderr").open("w") as err:
        process = subprocess.run(cmd, cwd=WORKSPACE, env=env, stdout=out, stderr=err, check=False)
    capture.update(returncode=process.returncode, process_wall_ms=1000 * (time.perf_counter() - start),
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   stdout_sha256=digest(directory / "stdout"), stderr_sha256=digest(directory / "stderr"))
    if implementation.startswith("custom-"):
        capture["binary_unchanged"] = capture["binary_sha256"] == digest(binary)
    capture_path.write_text(json.dumps(capture, indent=2) + "\n")
    if process.returncode or capture.get("binary_unchanged") is False:
        print(name, "FAILED", flush=True)
        return False
    quality_command = [sys.executable, str(HERE / "evaluate_cached.py"), "--train", str(DATA / dataset / "train.ghb"),
                       "--evaluation", str(DATA / dataset / "test.ghb"), "--predictions", str(directory / "result/predictions.f64"),
                       "--output", str(directory / "quality.json"), "--cache-root", str(root / "baseline-cache"),
                       "--provenance", str(directory / "quality-cache.json"), "--capture", str(capture_path)]
    start = time.perf_counter()
    with (directory / "quality.stdout").open("w") as out, (directory / "quality.stderr").open("w") as err:
        quality = subprocess.run(quality_command, cwd=WORKSPACE, env=env, stdout=out, stderr=err, check=False)
    capture.update(quality_command=quality_command, quality_returncode=quality.returncode,
                   quality_process_wall_ms=1000 * (time.perf_counter() - start))
    if quality.returncode == 0:
        capture.update(quality_sha256=digest(directory / "quality.json"), metrics_sha256=digest(directory / "result/metrics.json"),
                       predictions_sha256=digest(directory / "result/predictions.f64"), quality_cache_provenance_sha256=digest(directory / "quality-cache.json"))
        receipt = json.loads((directory / "quality-cache.json").read_text())
        capture.update(baseline_cache_key=receipt["cache_key"], baseline_cache_sha256=receipt["cache_sha256"], baseline_cache_hit=receipt["cache_hit"])
    capture_path.write_text(json.dumps(capture, indent=2) + "\n")
    print(name, "passed" if quality.returncode == 0 else "QUALITY FAILED", flush=True)
    return quality.returncode == 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--custom-binary", default=str(WORKSPACE / "build/booster-level-batch/ghb_real_bench"))
    p.add_argument("--validation-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=IMPLEMENTATIONS)
    p.add_argument("--repetitions", type=int, default=3)
    a = p.parse_args()
    if a.repetitions < 1:
        p.error("positive repetition count required")
    root, binary = Path(a.run_root).resolve(), Path(a.custom_binary).resolve()
    selection = selected(Path(a.validation_root).resolve(), a.datasets, a.implementations)
    root.mkdir(parents=True, exist_ok=False)
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (root / "plan.json").write_text(json.dumps({"argv": sys.argv, "datasets": a.datasets, "implementations": a.implementations,
        "repetitions": a.repetitions, "selection": "unchanged original validation-only selected()", "baseline_cache": "fresh test-only namespace; misses use original complete evaluator"}, indent=2) + "\n")
    (root / ".incomplete").write_text("Serial selected-test jobs pending.\n")
    outcomes = []
    for dataset in a.datasets:
        for repetition in range(a.repetitions):
            order = a.implementations if repetition % 2 == 0 else a.implementations[::-1]
            for implementation in order:
                name = f"test-{dataset}-r{repetition}-{implementation}"
                outcomes.append((name, execute(name, implementation, selection[dataset][implementation]["config"], dataset, root, binary)))
    (root / "summary.json").write_text(json.dumps({"jobs": len(outcomes), "passed": sum(ok for _, ok in outcomes),
        "failed": [name for name, ok in outcomes if not ok]}, indent=2) + "\n")
    (root / ".incomplete").unlink()
    if not all(ok for _, ok in outcomes):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
