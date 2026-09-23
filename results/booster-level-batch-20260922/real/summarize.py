#!/usr/bin/env python3
"""CPU-only audit and summary of complete validation-selected test campaigns."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import statistics
import struct
import numpy as np
from campaign import selected, digest, grid, DATA, HERE, WORKSPACE, IMPLEMENTATIONS, DATASETS

ORIGINAL_SOURCES = {"campaign.py", "framework.py", "fixtures.py", "evaluate.py"}
CACHED_TEST_SOURCES = ORIGINAL_SOURCES | {"test_campaign_cached.py", "evaluate_cached.py"}
CAPTURED_ARTIFACTS = {"quality.json": "quality_sha256", "result/metrics.json": "metrics_sha256",
                      "result/predictions.f64": "predictions_sha256", "stdout": "stdout_sha256", "stderr": "stderr_sha256"}

@lru_cache(maxsize=None)
def shared_digest(path):
    """Hash immutable campaign inputs once, after all ranked jobs finish."""
    return digest(path)

def recorded_path(value, cwd):
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()

def command_path(command, flag, cwd):
    if command.count(flag) != 1:
        raise RuntimeError("missing or duplicate captured command argument: " + flag)
    return recorded_path(command[command.index(flag) + 1], cwd)

def audit_case(directory, dataset, implementation, split, config, provenance, binary_hashes):
    """Validate every case, including validation configurations not selected."""
    capture = json.loads((directory / "capture.json").read_text())
    expected = {"name": directory.name, "dataset": dataset, "implementation": implementation, "split": split, "config": config}
    if any(capture.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"captured case scope/config mismatch: {directory}")
    if capture.get("returncode") != 0 or capture.get("quality_returncode") != 0:
        raise RuntimeError(f"incomplete or failed case: {directory}")
    cwd = Path(capture["cwd"]).resolve()
    if cwd != WORKSPACE:
        raise RuntimeError(f"unexpected captured working directory: {directory}")
    train, evaluation = [(DATA / dataset / filename).resolve() for filename in ("train.ghb", f"{split}.ghb")]
    expected_fixtures = {"training_fixture_sha256": shared_digest(train), "evaluation_fixture_sha256": shared_digest(evaluation)}
    if any(capture.get(key) != value for key, value in expected_fixtures.items()):
        raise RuntimeError(f"captured fixture identity changed: {directory}")
    command, quality_command = capture["command"], capture["quality_command"]
    for cmd in (command, quality_command):
        if command_path(cmd, "--train", cwd) != train or command_path(cmd, "--evaluation", cwd) != evaluation:
            raise RuntimeError(f"captured command fixture identity differs: {directory}")
    if command_path(command, "--output-dir", cwd) != (directory / "result").resolve():
        raise RuntimeError(f"captured training output path differs: {directory}")
    if command_path(quality_command, "--predictions", cwd) != (directory / "result/predictions.f64").resolve() or command_path(quality_command, "--output", cwd) != (directory / "quality.json").resolve():
        raise RuntimeError(f"captured evaluator artifact path differs: {directory}")
    sources = ORIGINAL_SOURCES if split == "validation" else CACHED_TEST_SOURCES
    expected_sources = {name: shared_digest((HERE / name).resolve()) for name in sorted(sources)}
    if capture.get("source_sha256") != expected_sources:
        raise RuntimeError(f"captured sources differ from frozen current sources: {directory}")
    evaluator = "evaluate.py" if split == "validation" else "evaluate_cached.py"
    if recorded_path(quality_command[1], cwd) != (HERE / evaluator).resolve():
        raise RuntimeError(f"captured evaluator command differs: {directory}")
    for filename, key in CAPTURED_ARTIFACTS.items():
        if digest(directory / filename) != capture.get(key):
            raise RuntimeError(f"captured artifact changed: {directory / filename}")
    quality = json.loads((directory / "quality.json").read_text())
    measured = json.loads((directory / "result/metrics.json").read_text())
    if quality["fixture_sha256"] != capture["evaluation_fixture_sha256"] or quality["training_fixture_sha256"] != capture["training_fixture_sha256"] or quality["predictions_sha256"] != capture["predictions_sha256"]:
        raise RuntimeError(f"quality reference identity mismatch: {directory}")
    if measured["backend"] != "cuda":
        raise RuntimeError(f"non-CUDA training substituted: {directory}")
    custom = implementation.startswith("custom-")
    if custom:
        binary = recorded_path(command[0], cwd)
        if capture.get("binary_unchanged") is not True or capture.get("binary_sha256") != shared_digest(binary):
            raise RuntimeError(f"custom binary changed during or after campaign: {directory}")
        binary_hashes.add(capture["binary_sha256"])
    elif recorded_path(command[1], cwd) != (HERE / "framework.py").resolve():
        raise RuntimeError(f"captured reference adapter command differs: {directory}")
    if split == "test":
        if digest(directory / "quality-cache.json") != capture.get("quality_cache_provenance_sha256"):
            raise RuntimeError(f"captured cache provenance changed: {directory}")
    counts = provenance[split]
    for key, number in (("cases", 1), ("fixture_hash_checks", 2), ("artifact_hash_checks", len(CAPTURED_ARTIFACTS)),
                        ("source_hash_checks", len(sources)), ("custom_binary_checks", int(custom)),
                        ("cache_provenance_hash_checks", int(split == "test"))):
        counts[key] = counts.get(key, 0) + number
    counts["case_captures"].append({"case": directory.name, "capture_sha256": digest(directory / "capture.json")})
    provenance["frozen_source_sha256"].update(expected_sources)
    return capture, quality, measured

LOWER_METRICS = {"mse", "rmse", "mae", "log_loss", "brier", "hamming_loss"}
HIGHER_METRICS = {"r2", "accuracy", "f1", "roc_auc", "average_precision", "macro_f1", "micro_f1", "exact_match_accuracy",
                  "macro_ap", "micro_ap", "macro_auc", "precision_at_1", "precision_at_3", "precision_at_5"}
METRIC_METADATA = {"selection_metric", "selection_value", "log_clip_epsilon", "macro_ap_labels", "macro_auc_labels"}

def strict_metric_gate(baseline, candidate):
    """Zero-allowance gate over every emitted quality metric, never metadata."""
    if set(baseline) != set(candidate):
        raise ValueError("metric schema differs between custom implementations")
    comparisons = {}
    for name in baseline:
        if name in METRIC_METADATA:
            if name != "selection_value" and baseline[name] != candidate[name]:
                raise ValueError("quality metadata differs between matched custom cases")
            continue
        base_name = name
        for suffix in ("_by_output", "_per_output", "_by_label", "_per_label"):
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        for prefix in ("per_output_", "per_label_"):
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
        if base_name not in LOWER_METRICS | HIGHER_METRICS:
            raise ValueError("unknown emitted quality metric direction: " + name)
        before, after = np.asarray(baseline[name], dtype=np.float64), np.asarray(candidate[name], dtype=np.float64)
        if before.shape != after.shape or not before.size or not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("invalid matched metric shape/finite contract")
        delta = after - before
        deterioration = delta if base_name in LOWER_METRICS else -delta
        comparisons[name] = {"direction": "lower" if base_name in LOWER_METRICS else "higher", "baseline": baseline[name],
            "candidate": candidate[name], "delta_candidate_minus_baseline": delta.tolist(), "checked_values": int(before.size),
            "regressed_values": int(np.count_nonzero(deterioration > 0)), "max_deterioration": max(0.0, float(np.max(deterioration))),
            "passed": bool(np.all(deterioration <= 0))}
    return {"allowance": 0, "passed": all(x["passed"] for x in comparisons.values()), "metrics": comparisons,
            "regressed_metrics": [name for name, value in comparisons.items() if not value["passed"]],
            "per_output_or_label_arrays_emitted": any(np.asarray(baseline[name]).ndim > 0 for name in comparisons),
            "scope": "all quality metrics emitted by the unchanged evaluator; counts, shape/selection metadata are not quality gates"}

def topology(path):
    """Explicit CPU validation reader for the native exported model layout."""
    raw = Path(path).read_bytes()
    position = 0
    def read(pattern):
        nonlocal position
        size = struct.calcsize(pattern)
        result = struct.unpack_from(pattern, raw, position)
        position += size
        return result
    magic, version, objective, outputs, features, trees = read("<8sIIIIQ")
    if magic != b"GHBMODEL" or version != 1:
        raise ValueError("unknown exported model format")
    base = read("<" + "d" * outputs)
    feature_payload = []
    for _ in range(features):
        kind, cuts, categories = read("<III")
        values = read("<" + "f" * (cuts + categories))
        feature_payload.append((kind, cuts, categories, values))
    structures, node_values = [], []
    for _ in range(trees):
        output, nodes = read("<II")
        structure = []
        for _ in range(nodes):
            feature, left, right, threshold, missing, value = read("<iiiIId")
            structure.append((feature, left, right, threshold, missing))
            node_values.append(value)
        structures.append((output, tuple(structure)))
    if position != len(raw):
        raise ValueError("exported model trailing bytes")
    return (objective, outputs, tuple(feature_payload), tuple(structures)), np.asarray(node_values), np.asarray(base)

def compare_custom(left, right):
    lm = json.loads((left / "result/metrics.json").read_text())
    rm = json.loads((right / "result/metrics.json").read_text())
    lc, rc = [json.loads((path / "capture.json").read_text()) for path in (left, right)]
    if lc["config"] != rc["config"] or lc["evaluation_fixture_sha256"] != rc["evaluation_fixture_sha256"] or lc["training_fixture_sha256"] != rc["training_fixture_sha256"] or lc["binary_sha256"] != rc["binary_sha256"]:
        raise ValueError("custom numerical comparison lacks matched config/fixtures/binary")
    lp, rp = [np.fromfile(path / "result/predictions.f64", dtype="<f8").reshape(lm["evaluation_rows"], lm["outputs"]) for path in (left, right)]
    difference = np.abs(lp - rp)
    result = {"per_output_case": str(left), "output_batch_case": str(right), "config": lc["config"],
              "prediction_values": int(lp.size), "nonidentical_values": int(np.count_nonzero(lp != rp)),
              "max_absolute_difference": float(np.max(difference)), "mean_absolute_difference": float(np.mean(difference)),
              "scope": "same binary, configuration, train and held-out rows; floating reordering remains distinct from exact preservation"}
    if lm["objective"] == 1:
        result["binary_decisions_changed"] = int(np.count_nonzero((lp >= .5) != (rp >= .5)))
    elif lm["objective"] == 2:
        result["class_decisions_changed"] = int(np.count_nonzero(np.argmax(lp, axis=1) != np.argmax(rp, axis=1)))
    lt, lv, lb = topology(left / "result/model.ghb")
    rt, rv, rb = topology(right / "result/model.ghb")
    result["topology_and_quantization_identical"] = lt == rt
    result["base_scores_identical"] = bool(np.array_equal(lb, rb))
    if lt == rt:
        result["max_node_value_difference"] = float(np.max(np.abs(lv - rv))) if len(lv) else 0.0
    lq, rq = [json.loads((path / "quality.json").read_text())["metrics"] for path in (left, right)]
    result["metric_delta_output_batch_minus_per_output"] = {key: rq[key] - lq[key] for key in lq.keys() & rq.keys()
        if isinstance(lq[key], (int, float)) and isinstance(rq[key], (int, float))}
    result["strict_loss_nonregression"] = rq["selection_value"] <= lq["selection_value"]
    result["strict_all_metrics_nonregression"] = strict_metric_gate(lq, rq)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-root", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    p.add_argument("--implementations", choices=IMPLEMENTATIONS, nargs="+", default=IMPLEMENTATIONS)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--smoke-root", action="append", default=[])
    p.add_argument("--memory-root", action="append", default=[])
    a = p.parse_args()
    validation, test, output = Path(a.validation_root), Path(a.test_root), Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    expected_selection = selected(validation, a.datasets, a.implementations)
    selection = json.loads((test / "selection.json").read_text())
    if selection != expected_selection:
        raise RuntimeError("test configuration selection disagrees with validation-only selection")
    summaries = {}
    custom_comparisons = {}
    validation_tradeoffs = {}
    binary_hashes = set()
    provenance = {"validation": {"case_captures": []}, "test": {"case_captures": []}, "frozen_source_sha256": {},
                  "summarizer_sha256": digest(Path(__file__).resolve()),
                  "scope": "all requested validation and test cases; five captured artifacts per case; original and cached-test source identities; unchanged custom binary; cache payload audit is separate"}
    for dataset in a.datasets:
        summaries[dataset] = {}
        observations = []
        for implementation in a.implementations:
            for index, config in enumerate(grid(dataset)):
                directory = validation / f"validation-{dataset}-g{index}-{implementation}"
                capture, q, m = audit_case(directory, dataset, implementation, "validation", config, provenance, binary_hashes)
                # Retain every fully audited observation without changing selection.
                observations.append({"implementation": implementation, "grid_index": index, "config": capture["config"],
                                     "validation_metric": q["metrics"]["selection_metric"], "validation_loss": q["metrics"]["selection_value"],
                                     "training_wall_ms": m["timing_ms"]["training_wall"], "case": directory.name})
        for point in observations:
            point["dominated_by"] = [other["case"] for other in observations if other is not point
                and other["training_wall_ms"] <= point["training_wall_ms"] and other["validation_loss"] <= point["validation_loss"]
                and (other["training_wall_ms"] < point["training_wall_ms"] or other["validation_loss"] < point["validation_loss"])]
            point["observed_nondominated"] = not point["dominated_by"]
        validation_tradeoffs[dataset] = {"observations": observations,
            "scope": "single-run validation time/loss observations, no uncertainty adjustment; descriptive only, not a repeated speed ranking or a new test-selection rule"}
        for implementation in a.implementations:
            runs = []
            for repetition in range(a.repetitions):
                directory = test / f"test-{dataset}-r{repetition}-{implementation}"
                capture, quality, measured = audit_case(directory, dataset, implementation, "test", selection[dataset][implementation]["config"], provenance, binary_hashes)
                runs.append({"directory": str(directory), "timing_ms": measured["timing_ms"], "metrics": quality["metrics"],
                             "baseline_metrics": quality["training_mean_baseline"], "model_bytes": measured["model_bytes"],
                             "prediction_backend": measured["prediction_backend"], "memory": measured["memory"], "trees": measured["trees"]})
            if len({r["prediction_backend"] for r in runs}) != 1:
                raise RuntimeError("prediction backend changed between repeats")
            summary = {"selected": selection[dataset][implementation], "runs": runs, "prediction_backend": runs[0]["prediction_backend"]}
            for metric in ("training_wall", "prediction_wall"):
                values = [r["timing_ms"][metric] for r in runs]
                summary[metric + "_ms"] = {"median": statistics.median(values), "min": min(values), "max": max(values), "raw": values}
            values = [r["metrics"]["selection_value"] for r in runs]
            summary["test_loss"] = {"metric": runs[0]["metrics"]["selection_metric"], "median": statistics.median(values), "min": min(values), "max": max(values)}
            summary["model_bytes_median"] = statistics.median([r["model_bytes"] for r in runs])
            summary["process_peak_rss_bytes"] = [r["memory"].get("process_peak_rss_bytes") for r in runs]
            if implementation.startswith("custom-"):
                summary["owned_device_payload_peak_bytes"] = [max(r["memory"]["owned_device_bytes"], r["memory"]["preparation_peak_bytes"]) for r in runs]
            summaries[dataset][implementation] = summary
        if {"custom-per-output", "custom-output-batch"}.issubset(a.implementations):
            comparisons = {"validation_matched_configs": [compare_custom(validation / f"validation-{dataset}-g{i}-custom-per-output",
                                                                          validation / f"validation-{dataset}-g{i}-custom-output-batch") for i in range(4)]}
            if selection[dataset]["custom-per-output"]["config"] == selection[dataset]["custom-output-batch"]["config"]:
                comparisons["test_matched_configs"] = [compare_custom(test / f"test-{dataset}-r{i}-custom-per-output", test / f"test-{dataset}-r{i}-custom-output-batch") for i in range(a.repetitions)]
            else:
                comparisons["test_matched_configs"] = []
                comparisons["test_comparison_note"] = "Validation selected different custom configurations, so selected test predictions are not numerical-preservation evidence. Matched validation comparisons are retained."
            custom_comparisons[dataset] = comparisons
    if any(i.startswith("custom-") for i in a.implementations) and len(binary_hashes) != 1:
        raise RuntimeError("custom validation/test campaign did not use exactly one unchanged binary")
    expected_cases = {"validation": sum(len(grid(d)) for d in a.datasets) * len(a.implementations),
                      "test": len(a.datasets) * len(a.implementations) * a.repetitions}
    for stage, expected in expected_cases.items():
        if provenance[stage]["cases"] != expected:
            raise RuntimeError(f"incomplete {stage} provenance audit")
        provenance[stage]["expected_cases"] = expected
    provenance["totals"] = {key: provenance["validation"][key] + provenance["test"][key] for key in
        ("cases", "fixture_hash_checks", "artifact_hash_checks", "source_hash_checks", "custom_binary_checks", "cache_provenance_hash_checks")}
    smoke_failures = []
    for root in a.smoke_root:
        for path in Path(root).rglob("failure.json"):
            smoke_failures.append({"path": str(path), "sha256": digest(path), "failure": json.loads(path.read_text())})
    memory_observations = []
    for root in a.memory_root:
        for path in Path(root).rglob("memory.json"):
            memory_observations.append({"path": str(path), "sha256": digest(path), "observation": json.loads(path.read_text())})
    result = {"audit": "all validation/test cases: complete budgets, selection, contracts, fixture/artifact/source hashes, unchanged custom binary and CUDA training checked",
              "provenance_audit": provenance,
              "binary_sha256": sorted(binary_hashes), "datasets": summaries, "custom_numerical_comparisons": custom_comparisons,
              "preserved_smoke_failures": smoke_failures, "separate_memory_observations": memory_observations,
              "validation_time_quality_tradeoffs": validation_tradeoffs}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Validation-selected real-data results", "", "All training jobs used GPU backends. Times below include preparation/upload/binning and model fitting from raw host input, in fresh processes with CUDA context setup excluded. Three selected test repetitions are summarized by median and full range. Parameters were selected only on validation; test metrics did not select models.", ""]
    for dataset in a.datasets:
        lines += [f"## {dataset}", "", "| Implementation | Rounds / depth | Training ms (range) | Test loss | Public prediction ms | Prediction backend |", "|---|---:|---:|---:|---:|---|"]
        for implementation in a.implementations:
            s = summaries[dataset][implementation]
            c = s["selected"]["config"]
            t, prediction = s["training_wall_ms"], s["prediction_wall_ms"]
            lines.append(f"| {implementation} | {c['rounds']} / {c['depth']} | {t['median']:.3f} ({t['min']:.3f}–{t['max']:.3f}) | {s['test_loss']['median']:.8g} {s['test_loss']['metric']} | {prediction['median']:.3f} | {s['prediction_backend']} |")
        lines.append("")
    lines += ["Native algorithm and architecture differences remain: CatBoost has symmetric shared vector leaves, LightGBM independent labels use a complete serial binary-relevance loop, and binning/objective/regularization details differ. CPU reference inference entries do not rank GPU prediction kernels. See PROTOCOL.md and every raw metric in summary.json. Custom owned device payload, process RSS and any separate sampled device-wide GPU-memory observation have different scopes and must not be conflated.", "", "These shallow, bounded pilot grids and four datasets do not establish saturated accuracy, large-language-model capability, or universal superiority.", ""]
    counts = provenance["totals"]
    lines += ["## Provenance audit", "", f"Verified every requested case: {provenance['validation']['cases']} validation and {provenance['test']['cases']} test. Checks passed for {counts['fixture_hash_checks']} captured fixture identities, {counts['artifact_hash_checks']} captured quality/metrics/prediction/stdout/stderr hashes, {counts['source_hash_checks']} frozen source hashes, {counts['custom_binary_checks']} unchanged custom-binary identities and {counts['cache_provenance_hash_checks']} cached-evaluator provenance hashes. Captured commands and quality-reference identities agree. Full case capture hashes and source identities are retained in JSON; immutable baseline-cache payloads have a separate audit.", ""]
    if custom_comparisons:
        lines += ["## Matched custom implementation checks", "", "The JSON preserves every matched validation configuration and each test repetition whose selected configurations match. Explicit higher/lower directions gate every emitted quality metric with zero allowance; metadata/counts are excluded. The unchanged evaluator emits aggregate metrics, not per-output metric arrays. A failed gate remains failed; it is not relabeled exact because a difference is small. Loss-only results remain separately recorded.", "", "| Dataset | Compared pairs | Max absolute prediction difference | Topology/quantization changed pairs | Strict loss regressions | Any-metric regressions |", "|---|---:|---:|---:|---:|---:|"]
        for dataset, checks in custom_comparisons.items():
            pairs = checks["validation_matched_configs"] + checks["test_matched_configs"]
            lines.append(f"| {dataset} | {len(pairs)} | {max(x['max_absolute_difference'] for x in pairs):.8g} | {sum(not x['topology_and_quantization_identical'] for x in pairs)} | {sum(not x['strict_loss_nonregression'] for x in pairs)} | {sum(not x['strict_all_metrics_nonregression']['passed'] for x in pairs)} |")
        lines.append("")
    if smoke_failures:
        lines += ["## Preserved capability failures", "", f"{len(smoke_failures)} failed smoke jobs are retained verbatim in summary.json, including adapter/runtime setup failures and unsupported CatBoost multi-output GPU evaluation. Supported training remains on GPU; explicit public CPU reference prediction is labeled above.", ""]
    lines += ["## Observed validation tradeoffs", "", "Each row below is nondominated among the four-config budgets on its dataset: no other observed point has both no-greater training time and no-greater validation loss, with at least one strict improvement. These are single validation-run timings, not repeated speed rankings. They do not change the validation-loss-only test selection. All dominated and nondominated observations remain in JSON.", "", "| Dataset | Implementation | Rounds / depth | Validation loss | Observed training ms |", "|---|---|---:|---:|---:|"]
    for dataset, tradeoffs in validation_tradeoffs.items():
        for point in sorted(tradeoffs["observations"], key=lambda p: p["training_wall_ms"]):
            if point["observed_nondominated"]:
                c = point["config"]
                lines.append(f"| {dataset} | {point['implementation']} | {c['rounds']} / {c['depth']} | {point['validation_loss']:.8g} {point['validation_metric']} | {point['training_wall_ms']:.3f} |")
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
