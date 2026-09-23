#!/usr/bin/env python3
"""Root-only serial real-data campaign. Import/--help do not touch the GPU."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WORKSPACE = HERE.parents[2]
DATA = EXPERIMENT / "data" / "fixtures"
IMPLEMENTATIONS = ["custom-per-output", "custom-output-batch", "xgboost", "lightgbm", "catboost"]
DATASETS = ["wine", "magic", "letter", "delicious"]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def grid(dataset):
    rounds, depths = ([5, 10], [2, 3]) if dataset == "delicious" else ([25, 75], [3, 5])
    return [{"rounds": r, "depth": d, "bins": 32, "learning_rate": .1, "l2": 1} for r in rounds for d in depths]

def command(implementation, config, dataset, split, output, binary):
    parameters = ["--train", str(DATA / dataset / "train.ghb"), "--evaluation", str(DATA / dataset / f"{split}.ghb"),
                  "--output-dir", str(output)]
    for k, value in config.items():
        parameters += ["--" + k.replace("_", "-"), str(value)]
    if implementation.startswith("custom-"):
        return [str(binary), *parameters, "--tree-build", implementation.removeprefix("custom-"),
                "--tree-execution", "graph", "--output-tile", "16", "--tree-export-batch", "16", "--histogram", "auto"]
    return [sys.executable, str(HERE / "framework.py"), "--framework", implementation, *parameters]

def execute(name, implementation, config, dataset, split, run_root, binary):
    record_dir = run_root / name
    record_dir.mkdir(parents=True, exist_ok=False)
    result_dir = record_dir / "result"
    cmd = command(implementation, config, dataset, split, result_dir, binary)
    capture = {"name": name, "implementation": implementation, "dataset": dataset, "split": split, "config": config,
               "command": cmd, "cwd": str(WORKSPACE), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "binary_sha256": digest(binary) if implementation.startswith("custom-") else None,
               "source_sha256": {p.name: digest(p) for p in [HERE / "campaign.py", HERE / "framework.py", HERE / "fixtures.py", HERE / "evaluate.py"]},
               "training_fixture_sha256": digest(DATA / dataset / "train.ghb"),
               "evaluation_fixture_sha256": digest(DATA / dataset / f"{split}.ghb")}
    (record_dir / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    nccl = WORKSPACE / "build/benchmark-env/lib/python3.12/site-packages/nvidia/nccl/lib"
    environment["LD_LIBRARY_PATH"] = str(nccl) + ":/usr/local/cuda/lib64:" + environment.get("LD_LIBRARY_PATH", "")
    start = time.perf_counter()
    with (record_dir / "stdout").open("w") as out, (record_dir / "stderr").open("w") as err:
        process = subprocess.run(cmd, cwd=WORKSPACE, env=environment, stdout=out, stderr=err, check=False)
    capture.update(returncode=process.returncode, process_wall_ms=1000 * (time.perf_counter() - start),
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   stdout_sha256=digest(record_dir / "stdout"), stderr_sha256=digest(record_dir / "stderr"))
    if implementation.startswith("custom-"):
        capture["binary_unchanged"] = capture["binary_sha256"] == digest(binary)
        if not capture["binary_unchanged"]:
            capture["error"] = "binary changed during run"
    (record_dir / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    if process.returncode or capture.get("binary_unchanged") is False:
        print(name, "FAILED", flush=True)
        return False
    # CPU validation begins after the GPU process exits, never concurrently with
    # a timed training job. The full corpus and all labels remain in scope.
    evaluation_command = [sys.executable, str(HERE / "evaluate.py"), "--train", str(DATA / dataset / "train.ghb"),
                          "--evaluation", str(DATA / dataset / f"{split}.ghb"), "--predictions", str(result_dir / "predictions.f64"),
                          "--output", str(record_dir / "quality.json")]
    with (record_dir / "quality.stdout").open("w") as out, (record_dir / "quality.stderr").open("w") as err:
        validation = subprocess.run(evaluation_command, cwd=WORKSPACE, env=environment, stdout=out, stderr=err, check=False)
    capture["quality_command"] = evaluation_command
    capture["quality_returncode"] = validation.returncode
    if validation.returncode == 0:
        capture["quality_sha256"] = digest(record_dir / "quality.json")
        capture["metrics_sha256"] = digest(result_dir / "metrics.json")
        capture["predictions_sha256"] = digest(result_dir / "predictions.f64")
    (record_dir / "capture.json").write_text(json.dumps(capture, indent=2) + "\n")
    print(name, "passed" if validation.returncode == 0 else "QUALITY FAILED", flush=True)
    return validation.returncode == 0

def selected(validation_root, datasets, implementations):
    selections = {}
    for dataset in datasets:
        selections[dataset] = {}
        for implementation in implementations:
            candidates = []
            for index, config in enumerate(grid(dataset)):
                path = validation_root / f"validation-{dataset}-g{index}-{implementation}"
                capture = json.loads((path / "capture.json").read_text())
                if capture["returncode"] or capture.get("quality_returncode") != 0:
                    raise RuntimeError(f"cannot select from an incomplete four-config budget: {path}")
                if capture["dataset"] != dataset or capture["implementation"] != implementation or capture["config"] != config or capture["split"] != "validation":
                    raise RuntimeError("validation case metadata mismatch")
                q = json.loads((path / "quality.json").read_text())
                m = json.loads((path / "result" / "metrics.json").read_text())
                if digest(path / "quality.json") != capture["quality_sha256"] or digest(path / "result/metrics.json") != capture["metrics_sha256"]:
                    raise RuntimeError("captured validation artifacts changed")
                candidates.append((q["metrics"]["selection_value"], m["timing_ms"]["training_wall"], index, config))
            loss, time_ms, index, config = min(candidates)
            selections[dataset][implementation] = {"config": config, "grid_index": index, "validation_loss": loss,
                                                     "validation_training_ms": time_ms, "validation_case": f"validation-{dataset}-g{index}-{implementation}"}
    return selections

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["validation", "test", "plan"], required=True)
    p.add_argument("--custom-binary", default=str(WORKSPACE / "build/booster-level-batch/ghb_real_bench"))
    p.add_argument("--run-root", required=True)
    p.add_argument("--validation-root")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=IMPLEMENTATIONS)
    p.add_argument("--repetitions", type=int, default=3)
    a = p.parse_args()
    root = Path(a.run_root).resolve()
    binary = Path(a.custom_binary).resolve()
    plan = {"datasets": a.datasets, "implementations": a.implementations, "grids": {d: grid(d) for d in a.datasets},
            "repetitions": a.repetitions, "timing": "serial synchronized jobs, no profiler, fresh process/context initialization excluded", "argv": sys.argv}
    if a.stage == "plan":
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    jobs = []
    if a.stage == "validation":
        for dataset in a.datasets:
            for index, config in enumerate(grid(dataset)):
                order = a.implementations if index % 2 == 0 else a.implementations[::-1]
                for implementation in order:
                    jobs.append((f"validation-{dataset}-g{index}-{implementation}", implementation, config, dataset, "validation"))
    else:
        if not a.validation_root:
            p.error("test requires --validation-root")
        selection = selected(Path(a.validation_root).resolve(), a.datasets, a.implementations)
        (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        for dataset in a.datasets:
            for repetition in range(a.repetitions):
                order = a.implementations if repetition % 2 == 0 else a.implementations[::-1]
                for implementation in order:
                    jobs.append((f"test-{dataset}-r{repetition}-{implementation}", implementation,
                                 selection[dataset][implementation]["config"], dataset, "test"))
    (root / ".incomplete").write_text("Remove only after every recorded job completes.\n")
    successes = []
    for name, implementation, config, dataset, split in jobs:
        successes.append(execute(name, implementation, config, dataset, split, root, binary))
    (root / "summary.json").write_text(json.dumps({"jobs": len(jobs), "passed": sum(successes),
        "failed": [job[0] for job, ok in zip(jobs, successes) if not ok]}, indent=2) + "\n")
    (root / ".incomplete").unlink()
    if not all(successes):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
