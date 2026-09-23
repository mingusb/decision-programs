#!/usr/bin/env python3
"""Test-only exact reuse of immutable CPU training-mean baseline metrics."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import evaluate as reference
from fixtures import load, sha256

HERE = Path(__file__).resolve().parent

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def key_spec(train, evaluation, header):
    manifest_path = HERE / "environment-final.json"
    manifest = json.loads(manifest_path.read_text())
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}
    if manifest["python"] != sys.version or any(manifest["packages"][name]["version"] != version for name, version in versions.items()):
        raise ValueError("metric environment differs from pinned manifest")
    return {"schema": 1, "training_fixture_sha256": sha256(train), "evaluation_fixture_sha256": sha256(evaluation),
            "shape": header, "evaluate_sha256": sha256(HERE / "evaluate.py"), "fixtures_sha256": sha256(HERE / "fixtures.py"),
            "environment_manifest_sha256": sha256(manifest_path), "python": sys.version, "byteorder": sys.byteorder,
            "metric_dependencies": versions}

def check_entry(entry, spec, key):
    if entry["key_spec"] != spec or entry["key"] != key:
        raise ValueError("immutable baseline cache key mismatch")
    origin = Path(entry["origin_quality_path"])
    if sha256(origin) != entry["origin_quality_sha256"]:
        raise ValueError("baseline cache origin quality report changed")
    original = json.loads(origin.read_text())
    if original["training_fixture_sha256"] != spec["training_fixture_sha256"] or original["fixture_sha256"] != spec["evaluation_fixture_sha256"]:
        raise ValueError("baseline cache origin fixture mismatch")
    if canonical(original["training_mean_baseline"]) != canonical(entry["training_mean_baseline"]):
        raise ValueError("baseline cache does not exactly match original full result")

def evaluate_cached(train, evaluation, predictions, output, cache_root, provenance, capture=None):
    output, cache_root, provenance = Path(output).resolve(), Path(cache_root).resolve(), Path(provenance).resolve()
    if output.exists() or provenance.exists():
        raise FileExistsError("quality/provenance output already exists")
    cache_root.mkdir(parents=True, exist_ok=True)
    _, _, train_header = load(train)
    _, y, header = load(evaluation)
    if any(train_header[k] != header[k] for k in ("features", "targets", "objective", "classes")):
        raise ValueError("train/evaluation contract mismatch")
    spec = key_spec(train, evaluation, header)
    key = hashlib.sha256(canonical(spec)).hexdigest()
    path = cache_root / (key + ".json")
    start = time.perf_counter()
    if path.exists():
        entry = json.loads(path.read_text())
        check_entry(entry, spec, key)
        outputs = header["classes"] if header["objective"] == 2 else header["targets"]
        p = np.fromfile(predictions, dtype="<f8").reshape(header["rows"], outputs)
        result = {"reference": "CPU float64 common metric implementation", "fixture_sha256": spec["evaluation_fixture_sha256"],
                  "training_fixture_sha256": spec["training_fixture_sha256"], "predictions_sha256": sha256(predictions),
                  "metrics": reference.metrics(y, p, header["objective"]), "training_mean_baseline": entry["training_mean_baseline"]}
        hit = True
    else:
        # Original complete implementation, including its exact baseline
        # arithmetic, is the sole producer of a new immutable cache value.
        result = reference.evaluate(train, evaluation, predictions)
        hit = False
    elapsed_ms = 1000 * (time.perf_counter() - start)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not hit:
        snapshot = None
        if capture is not None:
            snapshot = json.loads(Path(capture).read_text())
            if snapshot["returncode"] != 0 or snapshot["training_fixture_sha256"] != spec["training_fixture_sha256"] or snapshot["evaluation_fixture_sha256"] != spec["evaluation_fixture_sha256"]:
                raise ValueError("cache origin training receipt is invalid")
        entry = {"schema": 1, "key": key, "key_spec": spec, "training_mean_baseline": result["training_mean_baseline"],
                 "origin_quality_path": str(output), "origin_quality_sha256": sha256(output),
                 "origin_capture_path": str(Path(capture).resolve()) if capture else None,
                 "origin_training_capture_snapshot": snapshot,
                 "producer": "original evaluate.evaluate() unchanged", "producer_wrapper_sha256": sha256(Path(__file__))}
        with path.open("x") as stream:
            json.dump(entry, stream, indent=2, allow_nan=False)
            stream.write("\n")
    receipt = {"schema": 1, "cache_hit": hit, "cache_key": key, "cache_path": str(path), "cache_sha256": sha256(path),
               "origin_quality_path": entry["origin_quality_path"], "origin_quality_sha256": entry["origin_quality_sha256"],
               "quality_path": str(output), "quality_sha256": sha256(output), "wrapper_sha256": sha256(Path(__file__)),
               "key_spec": spec, "metric_evaluation_wall_ms": elapsed_ms,
               "candidate_arithmetic": "original evaluate.metrics() unchanged; miss uses original evaluate.evaluate() unchanged"}
    with provenance.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result, receipt

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "evaluation", "predictions", "output", "cache-root", "provenance"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--capture")
    a = p.parse_args()
    evaluate_cached(a.train, a.evaluation, a.predictions, a.output, a.cache_root, a.provenance, a.capture)
