#!/usr/bin/env python3
"""CPU-only provenance audit; never recomputes candidate metrics or uses the GPU."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_cached import HERE, canonical, check_entry, key_spec
from fixtures import sha256, HEADER

def require(ok, message):
    if not ok:
        raise ValueError(message)

def audit(root):
    entries = {}
    for path in sorted((root / "baseline-cache").glob("*.json")):
        entry = json.loads(path.read_text())
        spec = entry["key_spec"]
        origin_capture = json.loads(Path(entry["origin_capture_path"]).read_text())
        command = origin_capture["quality_command"]
        train = command[command.index("--train") + 1]
        evaluation = command[command.index("--evaluation") + 1]
        with Path(evaluation).open("rb") as stream:
            _, _, rows, features, targets, objective, classes = HEADER.unpack(stream.read(HEADER.size))
        header = dict(rows=rows, features=features, targets=targets, objective=objective, classes=classes)
        require(key_spec(train, evaluation, header) == spec, "current immutable cache inputs differ")
        key = hashlib.sha256(canonical(spec)).hexdigest()
        require(path.name == key + ".json", "cache filename/key mismatch")
        check_entry(entry, spec, key)
        require(origin_capture["returncode"] == 0 and origin_capture["quality_returncode"] == 0, "origin job failed")
        require(origin_capture["quality_sha256"] == entry["origin_quality_sha256"], "origin receipt quality hash differs")
        require(all(origin_capture[k] == v for k, v in entry["origin_training_capture_snapshot"].items()), "origin training capture was changed")
        require(entry["producer_wrapper_sha256"] == sha256(HERE / "evaluate_cached.py"), "producer wrapper changed")
        entries[key] = {"path": str(path), "sha256": sha256(path), "origin_quality_sha256": entry["origin_quality_sha256"], "uses": 0, "misses": 0}
    cases = []
    for path in sorted(root.glob("test-*/quality-cache.json")):
        receipt = json.loads(path.read_text())
        capture = json.loads((path.parent / "capture.json").read_text())
        quality = json.loads((path.parent / "quality.json").read_text())
        entry = json.loads(Path(receipt["cache_path"]).read_text())
        item = entries[receipt["cache_key"]]
        require(Path(receipt["cache_path"]).resolve() == Path(item["path"]).resolve(), "case references another cache file")
        require(capture["split"] == "test" and capture["returncode"] == 0 and capture["quality_returncode"] == 0, "invalid test case")
        require(capture["quality_cache_provenance_sha256"] == sha256(path), "cache provenance hash differs")
        require(capture["quality_sha256"] == sha256(path.parent / "quality.json") == receipt["quality_sha256"], "quality report hash differs")
        require(receipt["cache_sha256"] == item["sha256"] == capture["baseline_cache_sha256"], "immutable cache hash differs")
        require(capture["baseline_cache_key"] == receipt["cache_key"] and capture["baseline_cache_hit"] == receipt["cache_hit"], "capture/cache status differs")
        require(receipt["wrapper_sha256"] == capture["source_sha256"]["evaluate_cached.py"] == sha256(HERE / "evaluate_cached.py"), "executed wrapper identity differs")
        require(capture["source_sha256"]["evaluate.py"] == entry["key_spec"]["evaluate_sha256"] and capture["source_sha256"]["fixtures.py"] == entry["key_spec"]["fixtures_sha256"], "original metric/fixture source identity differs")
        require(receipt["key_spec"] == entry["key_spec"], "per-case key spec differs")
        require(canonical(quality["training_mean_baseline"]) == canonical(entry["training_mean_baseline"]), "baseline payload not exact")
        require(quality["training_fixture_sha256"] == receipt["key_spec"]["training_fixture_sha256"] and quality["fixture_sha256"] == receipt["key_spec"]["evaluation_fixture_sha256"], "quality fixture identity differs")
        command = capture["quality_command"]
        require(Path(command[1]).resolve() == (HERE / "evaluate_cached.py").resolve() and Path(command[command.index("--provenance") + 1]).resolve() == path.resolve(), "actual quality command not recorded")
        item["uses"] += 1
        item["misses"] += not receipt["cache_hit"]
        cases.append({"case": path.parent.name, "cache_key": receipt["cache_key"], "cache_hit": receipt["cache_hit"], "metric_evaluation_wall_ms": receipt["metric_evaluation_wall_ms"]})
    require(entries and all(v["misses"] == 1 for v in entries.values()), "each immutable baseline must have exactly one full-evaluator producer")
    captures = list(root.glob("test-*/capture.json"))
    require(len(cases) == len(captures), "missing per-case cache provenance")
    return {"audit": "exact immutable baseline origin, key, environment, command, source and payload provenance verified", "entries": entries, "cases": cases}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.root.resolve())
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
