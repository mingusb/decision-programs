#!/usr/bin/env python3
"""CPU-only audit of the complete frozen resident campaign; refuses overwrites.

Run only after run_campaign.py has completed. Exit 0 means all zero-allowance
quality comparisons passed; 1 retains metric regressions; 2 is invalid evidence.
No benchmark, CUDA, GPU telemetry, or profiler process is launched here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
EVALUATOR = ROOT / "training/tools/evaluate.py"
IDENTITY = ("schema_version", "kind", "generator_version", "objective", "rows",
            "test_rows", "features", "outputs", "seed", "test_seed", "rounds",
            "max_depth", "max_bins", "output_tile_size", "max_device_bytes",
            "max_histogram_bytes", "histogram", "learning_rate", "l2",
            "min_leaf_rows", "min_child_hessian", "min_gain", "max_leaf_value",
            "instrumentation", "gpu", "cuda_runtime", "cuda_driver")


def planned():
    pairs = [(f"scalar-hybrid-{suffix}", f"scalar-{mode}-{suffix}")
             for suffix in ("a", "b") for mode in ("stream8", "graph8", "graph4")]
    pairs += [(f"{objective}-hybrid", f"{objective}-resident")
              for objective in ("binary", "multiclass")]
    pairs += [(f"outputs{outputs}-hybrid", f"outputs{outputs}-graph")
              for outputs in (129, 1024, 4096)]
    pairs += [("outputs1024-hybrid", "outputs1024-stream")]
    return sorted({name for pair in pairs for name in pair}), pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("campaign_evaluate", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def identity(metadata):
    objective = metadata["objective"]
    if objective == "binary" and metadata["outputs"] > 1:
        objective = "multilabel"
    dataset = (f"ghb-generator-v{metadata['generator_version']}:train-seed={metadata['seed']}:"
               f"test-seed={metadata['test_seed']}:train-rows={metadata['rows']}:"
               f"test-rows={metadata['test_rows']}:features={metadata['features']}:"
               f"objective={objective}:outputs={metadata['outputs']}")
    return objective, dataset, f"heldout:seed={metadata['test_seed']}:rows={metadata['test_rows']}"


def matching(reference, candidate):
    differences = [key for key in IDENTITY if reference[key] != candidate[key]]
    if differences:
        raise ValueError("unmatched training/evaluation protocol: " + ", ".join(differences))


def validate_capture(name):
    directory = CAMPAIGN / name
    if (directory / ".incomplete").exists():
        raise ValueError("incomplete benchmark output")
    result = read_json(directory / "result.json")
    capture = read_json(CAMPAIGN / (name + "-capture.json"))
    if capture["returncode"] != 0 or capture["executable_unchanged"] is not True:
        raise ValueError("capture failed or executable changed during process")
    if result != read_json(CAMPAIGN / (name + "-stdout.json")):
        raise ValueError("stdout and saved result differ")
    executable = Path(capture["command"][0])
    candidates = [executable]
    if executable.is_relative_to(ROOT):
        candidates.append(CAMPAIGN / "first-provenance" / executable.relative_to(ROOT))
    retained_executable = next((path for path in candidates if path.is_file() and sha(path) == capture["executable_sha256"]), None)
    if retained_executable is None:
        raise ValueError("no retained executable matches the captured hash")
    if result["kind"] != "ghb.training" or result["schema_version"] != 1 or result["generator_version"] != 2:
        raise ValueError("unsupported benchmark/generator schema")
    if result["validation"]["serialization_equal"] is not True:
        raise ValueError("model serialization validation failed")
    error = result["validation"]["cpu_gpu_max_abs_error"]
    if isinstance(error, bool) or not math.isfinite(error) or error < 0:
        raise ValueError("invalid CPU/GPU prediction validation")
    if result["trees"] != result["rounds"] * result["outputs"]:
        raise ValueError("tree count differs from rounds times outputs")
    if result["test_seed"] != result["seed"] ^ 0x9E3779B97F4A7C15:
        raise ValueError("unexpected held-out seed")
    for key in IDENTITY:
        if key not in result:
            raise ValueError("missing comparison dimension: " + key)
    files = [directory / filename for filename in
             ("result.json", "targets.csv", "predictions.csv", "baseline.csv", "model.ghb")]
    files += [CAMPAIGN / (name + suffix) for suffix in ("-capture.json", "-stdout.json", "-stderr.txt")]
    # Verify an immutable snapshot when present, even if the working executable
    # is subsequently rebuilt for a separately named experiment.
    frozen = candidates[-1]
    if frozen.is_file() and sha(frozen) == capture["executable_sha256"]:
        retained_executable = frozen
    return result, [artifact(path) for path in files + [retained_executable]]


def prediction_changes(reference, candidate, targets, objective):
    """Unweighted exact binary64 differences and fixed-decision changes."""
    count = differing = changed_rows = decisions = decision_rows = positive_decisions = 0
    maximum = 0.0
    with Path(reference).open(newline="") as old, Path(candidate).open(newline="") as new, Path(targets).open(newline="") as target:
        before, after, truth = csv.reader(old), csv.reader(new), csv.reader(target)
        header, other, target_header = next(before), next(after), next(truth)
        if header != other or not header or header[0] != "row_id":
            raise ValueError("prediction headers differ")
        outputs = len(header) - 1
        per_output = [0] * outputs
        for a, b, t in itertools.zip_longest(before, after, truth):
            if a is None or b is None or t is None or a[0] != b[0] or a[0] != t[0]:
                raise ValueError("prediction/target row identities differ")
            if len(a) != len(header) or len(b) != len(header):
                raise ValueError("prediction widths differ")
            av, bv = list(map(float, a[1:])), list(map(float, b[1:]))
            if any(not math.isfinite(v) for v in av + bv):
                raise ValueError("nonfinite prediction difference input")
            row_different = 0
            for index, (x, y) in enumerate(zip(av, bv)):
                maximum = max(maximum, abs(y - x))
                per_output[index] += x != y
                row_different += x != y
            differing += row_different
            changed_rows += bool(row_different)
            count += outputs
            if objective in ("binary", "multilabel"):
                changed = sum((x >= 0.5) != (y >= 0.5) for x, y in zip(av, bv))
            elif objective == "multiclass":
                changed = int(max(range(outputs), key=av.__getitem__) != max(range(outputs), key=bv.__getitem__))
            else:
                changed = 0
            decisions += changed
            decision_rows += bool(changed)
            weight = float(t[-1]) if target_header[-1] == "weight" else 1.0
            positive_decisions += changed if weight > 0 else 0
    return {"values": count, "differing_values": differing, "max_abs_difference": maximum,
            "rows_with_any_difference": changed_rows, "differing_values_per_output": per_output,
            "decision_rule": ("probability >= 0.5" if objective in ("binary", "multilabel") else
                              "argmax; lowest class index breaks ties" if objective == "multiclass" else None),
            "decision_changes": decisions if objective != "regression" else None,
            "rows_with_decision_changes": decision_rows if objective != "regression" else None,
            "positive_weight_decision_changes": positive_decisions if objective != "regression" else None}


def read_model(path):
    """Read the portable v1 format on CPU; validate bounds and tree structure."""
    data = Path(path).read_bytes()
    offset = 0

    def take(fmt):
        nonlocal offset
        size = struct.calcsize("<" + fmt)
        if offset + size > len(data):
            raise ValueError("truncated model")
        values = struct.unpack_from("<" + fmt, data, offset)
        offset += size
        return values

    magic, version, objective, outputs, feature_count, tree_count = take("8sIIIIQ")
    if magic != b"GHBMODEL" or version != 1 or objective not in (0, 1, 2) or not outputs or not feature_count:
        raise ValueError("invalid model header")
    if outputs > len(data) // 8 or feature_count > len(data) // 12 or tree_count > len(data) // 8:
        raise ValueError("invalid model dimensions")
    bases = take("d" * outputs)
    features = []
    for _ in range(feature_count):
        kind, cuts, categories = take("III")
        if kind not in (0, 1) or cuts > 65534 or categories > 65535 or (categories if kind == 0 else cuts):
            raise ValueError("invalid feature metadata")
        values = take("f" * (cuts + categories))
        if any(not math.isfinite(v) or (i and values[i - 1] >= v) for i, v in enumerate(values)):
            raise ValueError("invalid feature values")
        features.append((kind, values))
    trees = []
    for _ in range(tree_count):
        output, count = take("II")
        if output >= outputs or not count or count > (len(data) - offset) // 28:
            raise ValueError("invalid tree dimensions")
        nodes = [take("iiiIId") for _ in range(count)]
        by_path, seen, pending = {}, set(), [("", 0)]
        while pending:
            path_id, index = pending.pop()
            if index < 0 or index >= count or index in seen:
                raise ValueError("invalid tree traversal")
            seen.add(index)
            feature, left, right, threshold, missing, value = nodes[index]
            if missing > 1 or not math.isfinite(value):
                raise ValueError("invalid tree value")
            by_path[path_id] = nodes[index]
            if feature == -1:
                if left != -1 or right != -1 or threshold != 0:
                    raise ValueError("malformed terminal node")
            else:
                if feature < 0 or feature >= feature_count or len(path_id) >= 31:
                    raise ValueError("invalid tree split")
                kind, values = features[feature]
                if threshold >= len(values) + (2 if kind == 0 else 1):
                    raise ValueError("invalid split threshold")
                pending.extend(((path_id + "L", left), (path_id + "R", right)))
        if len(seen) != count:
            raise ValueError("unreachable tree nodes")
        trees.append((output, by_path))
    if offset != len(data) or any(not math.isfinite(v) for v in bases):
        raise ValueError("invalid model trailer/base score")
    return {"objective": objective, "outputs": outputs, "bases": bases, "features": features, "trees": trees}


def model_changes(reference, candidate):
    a, b = read_model(reference), read_model(candidate)
    if (a["objective"], a["outputs"], len(a["features"]), len(a["trees"])) != (b["objective"], b["outputs"], len(b["features"]), len(b["trees"])):
        raise ValueError("model shapes differ")
    result = {"feature_metadata_equal": a["features"] == b["features"],
              "features_with_changed_cuts_or_categories": sum(x != y for x, y in zip(a["features"], b["features"])),
              "base_score_differing_values": sum(x != y for x, y in zip(a["bases"], b["bases"])),
              "base_score_max_abs_difference": max(abs(x - y) for x, y in zip(a["bases"], b["bases"])),
              "trees": len(a["trees"]), "trees_with_changed_structure_or_split": 0,
              "unmatched_structural_node_paths": 0, "leaf_vs_split_changes": 0,
              "split_feature_changes": 0, "same_feature_split_threshold_changes": 0,
              "split_missing_direction_changes": 0, "matched_leaf_value_changes": 0,
              "matched_leaf_max_abs_difference": 0.0}
    for (old_output, old), (new_output, new) in zip(a["trees"], b["trees"]):
        if old_output != new_output:
            raise ValueError("tree output assignments differ")
        unmatched = len(set(old) ^ set(new))
        result["unmatched_structural_node_paths"] += unmatched
        changed = bool(unmatched)
        for path in set(old) & set(new):
            x, y = old[path], new[path]
            if (x[0] == -1) != (y[0] == -1):
                result["leaf_vs_split_changes"] += 1
                changed = True
            elif x[0] == -1:
                result["matched_leaf_value_changes"] += x[5] != y[5]
                result["matched_leaf_max_abs_difference"] = max(result["matched_leaf_max_abs_difference"], abs(x[5] - y[5]))
            else:
                result["split_feature_changes"] += x[0] != y[0]
                result["same_feature_split_threshold_changes"] += x[0] == y[0] and x[3] != y[3]
                result["split_missing_direction_changes"] += x[4] != y[4]
                changed |= (x[0], x[3], x[4]) != (y[0], y[3], y[4])
        result["trees_with_changed_structure_or_split"] += changed
    return result


def compact_comparison(report):
    return {"status": report["status"], "regressions": report["regressions"],
            "metrics_checked": len(report["metrics"]),
            "applicable_metrics": sum(m["status"] != "not_applicable" for m in report["metrics"].values()),
            "maximum_deterioration_by_metric_family": {
                family: max((metric["deterioration"] for name, metric in report["metrics"].items()
                             if name.split("_output_")[0] == family and metric["status"] != "not_applicable"), default=None)
                for family in sorted({name.split("_output_")[0] for name in report["metrics"]})},
            "aggregate_metrics": {name: metric for name, metric in report["metrics"].items() if "_output_" not in name}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CAMPAIGN / "quality-audit")
    args = parser.parse_args()
    names, pairs = planned()
    # Missing captures are a preflight failure: never label a partial campaign complete.
    missing = [name for name in names if not (CAMPAIGN / name / "result.json").is_file()]
    if missing:
        raise ValueError("campaign is incomplete: " + ", ".join(missing))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".incomplete"
    marker.write_text("CPU quality audit in progress\n")
    evaluator = load_evaluator()
    initial_sources = [artifact(path) for path in (Path(__file__), EVALUATOR, CAMPAIGN / "run_campaign.py")]
    summary = {"schema_version": 1, "kind": "resident_campaign_quality_audit", "gpu_used": False,
               "allowance": 0.0, "started_unix_seconds": time.time(), "sources": initial_sources,
               "expected_cases": names, "expected_resident_comparisons": [list(pair) for pair in pairs],
               "cases": [], "resident_vs_hybrid": [], "errors": [], "raw_artifacts": [],
               "limitations": [
                   "Zero allowance applies independently to every applicable metric, including every output; failures remain failures, even at the last decimal place.",
                   "These deterministic synthetic held-out datasets are not evidence of NLP quality or accuracy on another dataset.",
                   "64 held-out rows in the 4096-output case remain 64 independent examples, not 262144 examples.",
                   "Exact target bytes, row identity, settings, and source hashes are required for comparisons.",
                   "Prediction comparisons use exact parsed binary64 values. Classification decision changes use >=0.5 or lowest-index argmax; regression has no classification threshold.",
                   "Model node comparisons align structural paths, not incidental node-array ordering. Different split features are not counted as same-feature threshold changes.",
                   "A pass on all reported metrics does not prove mathematical equivalence or preservation of quality on unobserved data."]}
    metadata, reports = {}, {}
    for name in names:
        case = {"case": name}
        summary["cases"].append(case)
        try:
            meta, artifacts = validate_capture(name)
            metadata[name] = meta
            summary["raw_artifacts"].extend(artifacts)
            objective, dataset, split = identity(meta)
            case.update(objective=objective, outputs=meta["outputs"], rows=meta["test_rows"])
            directory = CAMPAIGN / name
            target = directory / "targets.csv"
            model_report = evaluator.evaluate(target, directory / "predictions.csv", objective, dataset, split,
                                              meta["outputs"] if objective == "multiclass" else None)
            base_report = evaluator.evaluate(target, directory / "baseline.csv", objective, dataset, split,
                                             meta["outputs"] if objective == "multiclass" else None)
            if model_report["sample_count"] != meta["test_rows"] or model_report["outputs"] != meta["outputs"]:
                raise ValueError("CSV evaluation dimensions differ from benchmark")
            model_path, base_path = destination / (name + "-model.json"), destination / (name + "-base.json")
            write_json(model_path, model_report)
            write_json(base_path, base_report)
            comparison = evaluator.compare(base_path, model_path, 0.0)
            comparison_path = destination / (name + "-model-vs-base.json")
            write_json(comparison_path, comparison)
            case["model_vs_base"] = compact_comparison(comparison)
            case["reports"] = [artifact(path) for path in (model_path, base_path, comparison_path)]
            read_model(directory / "model.ghb")
            reports[name] = model_path
            case["status"] = comparison["status"]
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            case.update(status="invalid_evidence", error=str(error))
            summary["errors"].append({"case": name, "error": str(error)})
        print(name, case["status"], flush=True)
    for reference, candidate in pairs:
        pair = {"reference": reference, "candidate": candidate}
        summary["resident_vs_hybrid"].append(pair)
        try:
            matching(metadata[reference], metadata[candidate])
            before, after = CAMPAIGN / reference, CAMPAIGN / candidate
            if sha(before / "targets.csv") != sha(after / "targets.csv"):
                raise ValueError("exact target/weight CSV hashes differ")
            comparison = evaluator.compare(reports[reference], reports[candidate], 0.0)
            pair.update(compact_comparison(comparison))
            pair["predictions"] = prediction_changes(before / "predictions.csv", after / "predictions.csv",
                                                      before / "targets.csv", identity(metadata[candidate])[0])
            pair["model"] = model_changes(before / "model.ghb", after / "model.ghb")
            output = destination / (candidate + "-vs-" + reference + ".json")
            write_json(output, {"comparison": comparison, "prediction_changes": pair["predictions"], "model_changes": pair["model"]})
            pair["report"] = artifact(output)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            pair.update(status="invalid_evidence", error=str(error))
            summary["errors"].append({"reference": reference, "candidate": candidate, "error": str(error)})
        print(candidate, "vs", reference, pair["status"], flush=True)
    for item in initial_sources + summary["raw_artifacts"]:
        if sha(item["path"]) != item["sha256"]:
            summary["errors"].append({"artifact_changed_during_audit": item["path"]})
    statuses = [item["status"] for item in summary["cases"] + summary["resident_vs_hybrid"]]
    summary["status"] = "invalid_evidence" if summary["errors"] else "regression" if "regression" in statuses else "pass"
    summary["finished_unix_seconds"] = time.time()
    write_json(destination / "summary.json", summary)
    lines = ["# Resident campaign quality audit", "", f"Status: **{summary['status']}**. All comparisons use zero allowance.", "",
             "| Resident case | Matching hybrid | Quality gate | Differing prediction values | Max absolute difference | Classification decision changes | Changed split thresholds |",
             "|---|---|---|---:|---:|---:|---:|"]
    for pair in summary["resident_vs_hybrid"]:
        if "predictions" in pair:
            p, m = pair["predictions"], pair["model"]
            lines.append(f"| {pair['candidate']} | {pair['reference']} | {pair['status']} ({len(pair['regressions'])} regressed metrics) | {p['differing_values']} | {p['max_abs_difference']:.17g} | {p['decision_changes'] if p['decision_changes'] is not None else 'N/A'} | {m['same_feature_split_threshold_changes']} |")
        else:
            lines.append(f"| {pair['candidate']} | {pair['reference']} | invalid evidence | — | — | — | — |")
    lines += ["", "Model against its own constant base scores:", "", "| Case | Zero-allowance gate | Regressed metrics |", "|---|---|---:|"]
    for case in summary["cases"]:
        lines.append(f"| {case['case']} | {case['status']} | {len(case['model_vs_base']['regressions']) if 'model_vs_base' in case else 'N/A'} |")
    lines += ["", "Threshold counts above compare matching-feature split thresholds at matching structural paths. Feature, missing-direction, topology, leaf and base-score changes are retained separately in JSON.", ""]
    for limit in summary["limitations"]:
        lines.append("- " + limit)
    if summary["errors"]:
        lines += ["", "Evidence errors:", "", "```json", json.dumps(summary["errors"], indent=2), "```"]
    with (destination / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    marker.unlink()
    return 2 if summary["status"] == "invalid_evidence" else 1 if summary["status"] == "regression" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
