#!/usr/bin/env python3
"""CPU-only audit of the complete 56-run level-batching synthetic campaign.

No CUDA, benchmark, profiler, or telemetry process is launched. Exit 0 means
all strict quality gates passed, 1 retains regressions, 2 means invalid evidence.
The evaluator and earlier evidence are read only. Existing output is never replaced.
"""
from __future__ import annotations

from array import array
import bisect
import csv
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import statistics
import struct
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPERS = ROOT / "results/booster-resident-20260922/audit_optimized_quality.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


old = module("level_batch_retained_quality", HELPERS)
old.OUT = OUT
a = old.a
campaign = module("level_batch_campaign", OUT / "run_synthetic.py")
SHAPES = {
    "scalar": (65536, 2048, 32, 1, 10, 5, 64, "regression"),
    "33deep": (8192, 256, 16, 33, 3, 5, 32, "regression"),
    "129": (4096, 256, 16, 129, 3, 2, 32, "regression"),
    "1024": (4096, 256, 16, 1024, 2, 2, 16, "binary"),
    "4096": (1024, 64, 16, 4096, 1, 2, 16, "binary"),
    "multiclass17": (4096, 512, 16, 17, 3, 3, 32, "multiclass"),
    "fallback33": (2048, 128, 40, 33, 2, 3, 256, "regression"),
}
SEED = 20260922701
MASK64 = (1 << 64) - 1


def require(condition, text):
    if not condition:
        raise ValueError(text)


def name_for(case, execution, mode, order):
    return f"synthetic-{case}-{execution}-{mode}-{order}"


def plan():
    entries = [(name_for(case, execution, mode, order), case, execution, mode, order)
               for case in SHAPES for execution in ("stream", "graph")
               for order in ("a", "b") for mode in ("per-output", "output-batch")]
    pairs = [(name_for(case, execution, "per-output", order),
              name_for(case, execution, "output-batch", order))
             for case in SHAPES for execution in ("stream", "graph") for order in ("a", "b")]
    repeats = [(name_for(case, execution, mode, "a"), name_for(case, execution, mode, "b"))
               for case in SHAPES for execution in ("stream", "graph")
               for mode in ("per-output", "output-batch")]
    require(len(entries) == 56 and len(pairs) == len(repeats) == 28, "incorrect audit plan")
    require(set(SHAPES) == set(campaign.SHAPES), "campaign/audit shape coverage differs")
    return entries, pairs, repeats


def expected(case, execution, mode):
    rows, test_rows, features, outputs, rounds, depth, bins, objective = SHAPES[case]
    return dict(schema_version=1, kind="ghb.training", generator_version=2,
                rows=rows, test_rows=test_rows, features=features, outputs=outputs,
                rounds=rounds, max_depth=depth, max_bins=bins, objective=objective,
                seed=SEED, test_seed=SEED ^ 0x9E3779B97F4A7C15,
                output_tile_size=32, max_device_bytes=4 << 30,
                max_histogram_bytes=512 << 20, learning_rate=.1, l2=1.,
                min_leaf_rows=10, min_child_hessian=1e-8, min_gain=0., max_leaf_value=0.,
                histogram="auto", effective_deeper_histogram="global",
                tree_execution=execution, tree_build=mode, instrumentation="off",
                root_histogram="batched", split_policy="warp32", root_counts="reuse-global",
                split_batch="root", quantize_policy="radix8", tree_export_batch_requested=0,
                tree_export_batch_effective=0, root_histogram_batch_size=min(32, outputs),
                tree_batch_size=min(32, outputs) if mode == "output-batch" else 1,
                trees=outputs * rounds)


class MT19937_64:
    """Standard 312-word engine used by benchmark generator v2, CPU validation only."""
    def __init__(self, seed):
        self.state = [seed & MASK64]
        for index in range(1, 312):
            previous = self.state[-1]
            self.state.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & MASK64)
        self.index = 312

    def next(self):
        if self.index == 312:
            for index in range(312):
                mixed = (self.state[index] & 0xFFFFFFFF80000000) | (self.state[(index + 1) % 312] & 0x7FFFFFFF)
                self.state[index] = self.state[(index + 156) % 312] ^ (mixed >> 1) ^ (0xB5026F5AA96619E9 if mixed & 1 else 0)
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        return (value ^ (value >> 43)) & MASK64


def generated_holdout(meta):
    rng = MT19937_64(meta["test_seed"])
    weight_rng = MT19937_64(meta["test_seed"] ^ 0xD1B54A32D192ED03)
    categorical = meta["features"] >= 4
    category_count = min(4, meta["max_bins"] - 1)
    features, weights = [], []
    digest = hashlib.sha256()
    for row in range(meta["test_rows"]):
        values = []
        for feature in range(meta["features"]):
            value = 2 * ((rng.next() >> 40) * 2.0 ** -24) - 1
            if categorical and feature + 1 == meta["features"]:
                value = float(rng.next() % category_count)
            if feature % 7 == 0 and rng.next() % 31 == 0:
                value = math.nan
            values.append(value)
            digest.update(struct.pack("<f", value))
        features.append(values)
        weight = 0.0 if row % 97 == 96 else struct.unpack("<f", struct.pack("<f", .5 + (weight_rng.next() >> 40) * 2.0 ** -24))[0]
        weights.append(weight)
    return features, weights, digest.hexdigest()


def transform(row, objective):
    if objective == 1:
        result = []
        for margin in row:
            e = math.exp(-abs(margin))
            result.append(1 / (1 + e) if margin >= 0 else e / (1 + e))
        return result
    if objective == 2:
        maximum = max(row)
        values = [math.exp(value - maximum) for value in row]
        denominator = 0.0
        for value in values:
            denominator += value
        return [value / denominator for value in values]
    return row


def independent_predictions(name, meta, model, destination, holdout):
    features, weights, feature_hash = holdout
    directory = OUT / name
    generated = destination / (name + "-independent-predictions.csv")
    raw = array("d")
    maximum = maximum_base = maximum_relative = 0.0
    differences = decisions = 0
    with (directory / "predictions.csv").open(newline="") as pf, \
         (directory / "baseline.csv").open(newline="") as bf, \
         (directory / "targets.csv").open(newline="") as tf, generated.open("x", newline="") as gf:
        predictions, baseline, targets = csv.reader(pf), csv.reader(bf), csv.reader(tf)
        header, base_header, target_header = next(predictions), next(baseline), next(targets)
        require(header == base_header and len(header) == meta["outputs"] + 1, "prediction header shape mismatch")
        require(target_header[-1] == "weight", "generator v2 target weights missing")
        writer = csv.writer(gf, lineterminator="\n")
        writer.writerow(header)
        base = transform(list(model["bases"]), model["objective"])
        rows = 0
        for index, triple in enumerate(itertools.zip_longest(predictions, baseline, targets)):
            saved, saved_base, target = triple
            require(saved is not None and saved_base is not None and target is not None, "CSV row coverage mismatch")
            require(index < len(features), "extra CSV row")
            require(saved[0] == saved_base[0] == target[0] == str(index), "generator row ID mismatch")
            require(float(target[-1]) == weights[index], "independently generated weight mismatch")
            bins = []
            for value, (kind, values) in zip(features[index], model["features"]):
                if math.isnan(value):
                    bins.append(0)
                elif kind == 0:
                    bins.append(1 + bisect.bisect_left(values, value))
                else:
                    where = bisect.bisect_left(values, value)
                    bins.append(where + 1 if where < len(values) and values[where] == value else 0)
            margins = list(model["bases"])
            for output, nodes in model["trees"]:
                path = ""
                while True:
                    feature, _, _, threshold, missing_left, value = nodes[path]
                    if feature == -1:
                        margins[output] += value
                        break
                    bin_id = bins[feature]
                    left = bool(missing_left) if not bin_id else (bin_id <= threshold if model["features"][feature][0] == 0 else bin_id == threshold)
                    path += "L" if left else "R"
            raw.extend(margins)
            reference = transform(margins, model["objective"])
            actual = list(map(float, saved[1:]))
            actual_base = list(map(float, saved_base[1:]))
            require(len(actual) == len(actual_base) == model["outputs"], "prediction width mismatch")
            for expected_value, actual_value, expected_base, actual_base_value in zip(reference, actual, base, actual_base):
                error = abs(actual_value - expected_value)
                base_error = abs(actual_base_value - expected_base)
                require(math.isfinite(actual_value) and error <= 1e-10 * (1 + abs(expected_value)), "independent CPU/model and saved GPU prediction mismatch")
                require(math.isfinite(actual_base_value) and base_error <= 1e-10 * (1 + abs(expected_base)), "independent base model mismatch")
                maximum = max(maximum, error)
                maximum_base = max(maximum_base, base_error)
                maximum_relative = max(maximum_relative, error / (1 + abs(expected_value)))
                differences += actual_value != expected_value
            if model["objective"] == 1:
                decisions += sum((x >= .5) != (y >= .5) for x, y in zip(reference, actual))
            elif model["objective"] == 2:
                decisions += max(range(len(reference)), key=reference.__getitem__) != max(range(len(actual)), key=actual.__getitem__)
            writer.writerow([index, *reference])
            rows += 1
    require(rows == meta["test_rows"], "independent prediction row count mismatch")
    return generated, raw, dict(status="pass", generated_feature_f32_sha256=feature_hash,
        independently_regenerated_weights=True, max_abs_prediction_error=maximum,
        max_abs_base_error=maximum_base, max_scaled_prediction_error=maximum_relative,
        differing_prediction_values=differences, decision_changes=decisions,
        consistency_tolerance="existing benchmark: abs(error) <= 1e-10 * (1 + abs(reference)); quality allowance remains zero")


def validate(name, case, execution, mode, order):
    meta, capture, artifacts = old.validate_capture(name)
    for key, value in expected(case, execution, mode).items():
        require(meta[key] == value, f"{name}: unexpected {key}: {meta[key]!r}")
    command = [str(campaign.EXE), *campaign.SHAPES[case], "--seed", str(SEED), "--instrumentation", "off",
               "--tree-execution", execution, "--tree-build", mode, "--output-dir", str(OUT / name)]
    require(capture["command"] == command, "capture differs from exact campaign command")
    require(meta["samples"] == [] and meta["tuning"] == [], "ranking run includes recorded/tuning samples")
    require(len(meta["training_loss"]) == meta["rounds"] + 1, "training loss coverage mismatch")
    for value in list(meta["timing"].values()) + meta["training_loss"]:
        require(not isinstance(value, bool) and math.isfinite(value) and value >= 0, "invalid recorded timing/loss")
    model = a.read_model(OUT / name / "model.ghb")
    require(model["objective"] == {"regression": 0, "binary": 1, "multiclass": 2}[meta["objective"]], "serialized objective mismatch")
    require((model["outputs"], len(model["features"]), len(model["trees"])) == (meta["outputs"], meta["features"], meta["trees"]), "serialized model shape mismatch")
    require(all(output == index % meta["outputs"] for index, (output, _) in enumerate(model["trees"])), "serialized tree round/output order mismatch")
    require(meta["memory"]["device_payload_bytes"] <= meta["max_device_bytes"] and
            meta["memory"]["histogram_bytes"] <= meta["max_histogram_bytes"], "recorded memory exceeds budget")
    if mode == "output-batch":
        require(meta["memory"]["root_histogram_bytes"] == meta["memory"]["root_split_bytes"] == 0 and
                meta["memory"]["tree_state_bytes"] > 0, "retained batch storage accounting inconsistent")
    return meta, capture, artifacts, model


def main():
    entries, pairs, repeats = plan()
    if sys.argv[1:] == ["--plan-only"]:
        print("Plan verified: 56 cases, 28 cross-policy and 28 within-policy repeat gates; audit not run.")
        return 0
    require(not sys.argv[1:], "usage: audit_synthetic.py [--plan-only]")
    require(all((OUT / name / "result.json").is_file() for name, *_ in entries), "complete 56-run campaign required")
    destination = OUT / "quality-synthetic"
    destination.mkdir(exist_ok=False)
    marker = destination / ".incomplete"
    marker.touch(exist_ok=False)
    evaluator = a.load_evaluator()
    benchmark = OUT / "final-provenance/training/bench/booster.cpp"
    if not benchmark.is_file():
        benchmark = ROOT / "training/bench/booster.cpp"
    summary = dict(schema_version=1, kind="level_batch_synthetic_audit", gpu_used=False,
        started_unix_seconds=time.time(), allowance=0.0, expected_cases=56, expected_comparisons=28,
        expected_repeat_comparisons=28, cases=[], comparisons=[], repeat_comparisons=[], timings=[], errors=[], artifacts=[],
        sources=[a.artifact(path) for path in (Path(__file__), OUT / "run_synthetic.py", HELPERS, old.HELPERS, a.EVALUATOR, benchmark)],
        limitations=["Two uninstrumented samples per mode/execution/shape; medians are descriptive, not confidence intervals.",
            "Independent CPU model consistency uses the existing benchmark tolerance; every applicable quality metric retains zero allowance.",
            "Within-policy a-to-b repeat gates quantify run variability independently; they cannot change any cross-policy failure into a pass.",
            "All-bin raw-margin bounds compare exact-real serialized model functions and exclude runtime accumulation/transform rounding.",
            "These are synthetic generator-v2 fixtures, not a general speed/accuracy claim or real-data benchmark."])
    metadata, captures, reports, models, raw_margins, holdouts = {}, {}, {}, {}, {}, {}
    for name, case, execution, mode, order in entries:
        row = dict(name=name, shape=case, execution=execution, mode=mode, order=order)
        summary["cases"].append(row)
        try:
            meta, capture, artifacts, model = validate(name, case, execution, mode, order)
            metadata[name], captures[name], models[name] = meta, capture, model
            summary["artifacts"].extend(artifacts)
            if case not in holdouts:
                holdouts[case] = generated_holdout(meta)
            independent, raw_margins[name], row["independent_model_validation"] = independent_predictions(name, meta, model, destination, holdouts[case])
            objective, dataset, split = a.identity(meta)
            paths = {}
            for label, path in (("model", OUT / name / "predictions.csv"),
                                ("base", OUT / name / "baseline.csv"), ("independent", independent)):
                report = evaluator.evaluate(OUT / name / "targets.csv", path, objective, dataset, split,
                    meta["outputs"] if objective == "multiclass" else None)
                require((report["sample_count"], report["outputs"]) == (meta["test_rows"], meta["outputs"]), "evaluator dimensions mismatch")
                paths[label] = destination / (name + "-" + label + ".json")
                a.write_json(paths[label], report)
            reports[name] = paths["model"]
            comparison = evaluator.compare(paths["base"], paths["model"], 0.0)
            a.write_json(destination / (name + "-model-vs-base.json"), comparison)
            row.update(a.compact_comparison(comparison))
            row["generated_artifacts"] = [a.artifact(path) for path in [independent, *paths.values()]]
        except (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append(dict(case=name, error=str(error)))
        print(name, row["status"], flush=True)
    paired_work = [("comparisons", reference, candidate) for reference, candidate in pairs]
    paired_work += [("repeat_comparisons", reference, candidate) for reference, candidate in repeats]
    for category, reference, candidate in paired_work:
        row = dict(reference=reference, candidate=candidate,
                   category="cross_policy" if category == "comparisons" else "within_policy_repeat")
        summary[category].append(row)
        try:
            a.matching(metadata[reference], metadata[candidate])
            for key in ("tree_execution", "quantize_policy", "root_counts", "root_histogram", "split_policy", "split_batch", "tree_export_batch_requested", "tree_export_batch_effective", "effective_deeper_histogram"):
                require(metadata[reference][key] == metadata[candidate][key], "unmatched " + key)
            if category == "repeat_comparisons":
                require(metadata[reference]["tree_build"] == metadata[candidate]["tree_build"], "repeat changes tree-build policy")
            require(captures[reference]["executable_sha256"] == captures[candidate]["executable_sha256"], "paired binary hash differs")
            before, after = OUT / reference, OUT / candidate
            require(a.sha(before / "targets.csv") == a.sha(after / "targets.csv"), "paired target/weight bytes differ")
            require(old.feature_bytes(models[reference]) == old.feature_bytes(models[candidate]), "paired quantization metadata differs")
            comparison = evaluator.compare(reports[reference], reports[candidate], 0.0)
            row.update(a.compact_comparison(comparison))
            row["predictions"] = a.prediction_changes(before / "predictions.csv", after / "predictions.csv", before / "targets.csv", a.identity(metadata[candidate])[0])
            row["model"] = a.model_changes(before / "model.ghb", after / "model.ghb")
            row["encoded_domain"] = old.exact_region_comparison(models[reference], models[candidate])
            raw_before, raw_after = raw_margins[reference], raw_margins[candidate]
            require(len(raw_before) == len(raw_after), "independent raw margin dimensions differ")
            row["independent_raw_margins"] = dict(max_abs_difference=max(abs(x - y) for x, y in zip(raw_before, raw_after)),
                differing_values=sum(x != y for x, y in zip(raw_before, raw_after)), values=len(raw_before))
            a.write_json(destination / (candidate + "-vs-" + reference + ".json"), dict(comparison=comparison, **row))
        except (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append(dict(reference=reference, candidate=candidate, error=str(error)))
        print(row["category"], candidate, row["status"], flush=True)
    if len({capture["executable_sha256"] for capture in captures.values()}) != 1:
        summary["errors"].append(dict(error="campaign uses multiple executable hashes"))
    for case in SHAPES:
        for execution in ("stream", "graph"):
            row = dict(shape=case, execution=execution, samples_per_mode=2, modes={})
            try:
                for mode in ("per-output", "output-batch"):
                    names = [name_for(case, execution, mode, order) for order in ("a", "b")]
                    row["modes"][mode] = dict(cases=names, timing={}, memory=[metadata[name]["memory"] for name in names])
                    for key in metadata[names[0]]["timing"]:
                        samples = [metadata[name]["timing"][key] for name in names]
                        row["modes"][mode]["timing"][key] = dict(samples=samples, median=statistics.median(samples))
                row["speedup_reference_over_candidate"] = {
                    key: row["modes"]["per-output"]["timing"][key]["median"] / row["modes"]["output-batch"]["timing"][key]["median"]
                    for key in ("training_ms", "total_train_ms", "gpu_predict_wall_ms")}
                row["status"] = "valid_descriptive_timings"
            except (KeyError, ZeroDivisionError) as error:
                row.update(status="invalid_evidence", error=str(error))
                summary["errors"].append(dict(timing_case=case, execution=execution, error=str(error)))
            summary["timings"].append(row)
    for artifact in summary["sources"] + summary["artifacts"]:
        try:
            require(a.sha(artifact["path"]) == artifact["sha256"], "artifact changed during audit")
        except (OSError, ValueError) as error:
            summary["errors"].append(dict(artifact=artifact["path"], error=str(error)))
    statuses = [row["status"] for row in summary["cases"] + summary["comparisons"] + summary["repeat_comparisons"]]
    for category, key in (("comparisons", "cross_policy_status"), ("repeat_comparisons", "within_policy_repeat_status")):
        category_status = {row["status"] for row in summary[category]}
        summary[key] = "invalid_evidence" if "invalid_evidence" in category_status else "regression" if "regression" in category_status else "pass"
    summary["status"] = "invalid_evidence" if summary["errors"] else "regression" if "regression" in statuses else "pass"
    summary["finished_unix_seconds"] = time.time()
    a.write_json(destination / "summary.json", summary)
    lines = ["# Synthetic level-batching audit", "", f"Status: **{summary['status']}**. Every applicable quality metric retains zero allowance.", "",
             "| Shape | Submission | Per-output training ms (a, b) | Output-batch training ms (a, b) | Median training speedup | Median total-training speedup |",
             "|---|---|---|---|---:|---:|"]
    for row in summary["timings"]:
        if row["status"] != "valid_descriptive_timings":
            continue
        modes, speed = row["modes"], row["speedup_reference_over_candidate"]
        before = ", ".join(f"{x:.4f}" for x in modes["per-output"]["timing"]["training_ms"]["samples"])
        after = ", ".join(f"{x:.4f}" for x in modes["output-batch"]["timing"]["training_ms"]["samples"])
        lines.append(f"| {row['shape']} | {row['execution']} | {before} | {after} | {speed['training_ms']:.4f}× | {speed['total_train_ms']:.4f}× |")
    for category, label in (("comparisons", "Cross-policy gates"), ("repeat_comparisons", "Within-policy repeat gates: a versus b")):
        lines += ["", "## " + label, "", "| Candidate | Gate | Regressed metrics | Max saved prediction delta | Decision changes | Max independent raw-margin delta | All-bin margin bound |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in summary[category]:
            if "encoded_domain" not in row or "independent_raw_margins" not in row:
                lines.append(f"| {row['candidate']} | invalid evidence | — | — | — | — | — |")
                continue
            p = row["predictions"]
            lines.append(f"| {row['candidate']} | {row['status']} | {len(row['regressions'])} | {p['max_abs_difference']:.17g} | {p['decision_changes']} | {row['independent_raw_margins']['max_abs_difference']:.17g} | {row['encoded_domain']['maximum_raw_margin_bound_upward']:.17g} |")
    lines += ["", *["- " + limitation for limitation in summary["limitations"]], ""]
    with (destination / "summary.md").open("x") as handle:
        handle.write("\n".join(lines))
    marker.unlink()
    return 2 if summary["errors"] else int(summary["status"] == "regression")


if __name__ == "__main__":
    raise SystemExit(main())
