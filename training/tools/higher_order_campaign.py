#!/usr/bin/env python3
"""Serial higher-order experiment orchestration; GPU execution belongs to root.

Fixture loading and signal metrics are explicit offline CPU validation references.
No production training, preprocessing, derivative calculation or prediction runs
on the CPU. Existing sealed experiments and metric implementations are read-only.
"""
from __future__ import annotations
import argparse
import datetime
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
SEALED = WORKSPACE / "results/booster-level-batch-20260922/real"
DATA = SEALED.parent / "data/fixtures"
DATASETS = ("magic", "delicious")
ORDERS = (2, 3, 4)
HEADER = struct.Struct("<8s6I")
SELECTION = {"magic": "average_precision", "delicious": "macro_ap"}
COMMON = {"bins": 32, "learning_rate": .1, "max_leaf_value": 1,
          "tree_build": "output-batch", "tree_execution": "graph", "histogram": "global",
          "output_tile": 16, "tree_export_batch": 16}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    return {str(p): digest(p) for p in (Path(__file__).resolve(), SEALED / "evaluate.py", SEALED / "fixtures.py",
                                      SEALED / "summarize.py", SEALED / "environment-final.json")}


def grid(dataset):
    rounds, depth = ((25, 75), 5) if dataset == "magic" else ((5, 10), 3)
    return [{**COMMON, "rounds": r, "depth": depth, "l2": penalty} for r in rounds for penalty in (1, 10)]


def screen_config(dataset):
    return {**COMMON, "rounds": 25 if dataset == "magic" else 5, "depth": 3 if dataset == "magic" else 2, "l2": 1}


def protocol(binary):
    return {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "purpose": "exploratory closed-form orders 2/3/4: signal quality, complete synchronized training/prediction time and memory",
            "datasets": list(DATASETS), "orders": list(ORDERS), "common": COMMON,
            "screen": {"jobs": 8, "configs": {d: screen_config(d) for d in DATASETS},
                       "extra_reference": "one order-2 max_leaf_value=0 job per dataset; no screening-driven order elimination"},
            "validation": {"jobs": 24, "grids": {d: grid(d) for d in DATASETS},
                           "selection": {"magic": "maximize average_precision", "delicious": "maximize macro_average_precision (unchanged evaluator key macro_ap)"},
                           "tiebreak": ["minimize validation log_loss", "minimize synchronized training_wall_ms", "lowest fixed grid index"]},
            "test": {"jobs": 18, "repetitions": 3, "training_rows": "original training split only",
                     "selection": "one validation-selected config per dataset/order, frozen threshold from that validation model",
                     "status": "reused held-out test: previously inspected during development, exploratory follow-up, not fresh confirmation"},
            "threshold": {"false_positive_rate_limit": .05, "scope": "binary or pooled micro multilabel",
                          "rule": "maximize validation recall under exact 20*FP <= negatives; never divide tied scores; ties minimize FP then maximize threshold",
                          "application": "same validation threshold applied unchanged to all three fresh training repeats; observed test FPR may exceed 5%"},
            "time_to_quality": "shared target is best observed clipped order-2 validation selection metric; earliest sampled round budget reaching it at any offered lambda; independently fitted coarse checkpoints, not a continuous trajectory or time interpolation",
            "comparisons": "zero-allowance all-metric gates against order 2 only at matched config/fixtures; losses and signal changes all preserved; changed objective optimizer is not exact model preservation",
            "execution": "all GPU jobs serial, fresh processes, no profiler; CPU references run only after training exits; context/load/serialization separately reported",
            "limitations": ["two lambda choices and shallow bounded budgets are not exhaustive tuning", "three fits on one split measure training variability, not data uncertainty", "all 983 Delicious outputs retained", "new data are needed for confirmation", "screen failures are preserved, not silently excluded"],
            "fixture_sha256": {f"{d}/{s}": digest(DATA / d / f"{s}.ghb") for d in DATASETS for s in ("train", "validation", "test")},
            "source_sha256": sources(), "binary": str(binary), "binary_sha256": digest(binary) if binary.exists() else None}


def command(binary, config, order, dataset, split, output):
    result = [str(binary), "--train", str(DATA / dataset / "train.ghb"), "--evaluation", str(DATA / dataset / f"{split}.ghb"),
              "--output-dir", str(output), "--optimization-order", str(order)]
    for key, value in config.items():
        result.extend(("--" + key.replace("_", "-"), str(value)))
    return result


def target_view(path):
    import numpy as np
    with path.open("rb") as stream:
        magic, version, rows, features, outputs, objective, classes = HEADER.unpack(stream.read(HEADER.size))
    if magic != b"GHBDS001" or version != 1 or objective != 1 or path.stat().st_size != HEADER.size + 4 * rows * (features + outputs):
        raise ValueError("expected complete independent binary fixture")
    return np.memmap(path, dtype="<f4", mode="r", offset=HEADER.size + rows * features * 4, shape=(rows, outputs))


def operating_point(y, prediction, threshold):
    import numpy as np
    positive, decision = np.asarray(y).reshape(-1) == 1, np.asarray(prediction).reshape(-1) >= threshold
    tp = int(np.count_nonzero(positive & decision))
    fp = int(np.count_nonzero(~positive & decision))
    positives, negatives = int(np.count_nonzero(positive)), int(np.count_nonzero(~positive))
    if not positives or not negatives:
        raise ValueError("operating point requires positive and negative observations")
    return {"threshold": float(threshold), "true_positive": tp, "false_positive": fp,
            "false_negative": positives - tp, "true_negative": negatives - fp,
            "positives": positives, "negatives": negatives, "recall": tp / positives,
            "false_positive_rate": fp / negatives, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "f1": 2 * tp / (positives + tp + fp) if positives + tp + fp else 0.0,
            "meets_five_percent_fpr": 20 * fp <= negatives}


def choose_threshold(y, prediction):
    """Exact tied-score cumulative-count CPU reference; no threshold tuning on test."""
    import numpy as np
    p, truth = np.asarray(prediction).reshape(-1), np.asarray(y).reshape(-1) == 1
    if p.size != truth.size or not p.size or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid threshold prediction contract")
    positives, negatives = int(np.count_nonzero(truth)), int(np.count_nonzero(~truth))
    if not positives or not negatives:
        raise ValueError("threshold selection requires both classes")
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    tp = np.cumsum(truth[order], dtype=np.int64)[ends]
    fp = ends + 1 - tp
    eligible = np.flatnonzero(20 * fp <= negatives)
    threshold = float(np.nextafter(float(sorted_p[0]), math.inf))
    if eligible.size:
        best_tp = int(tp[eligible].max())
        # Earliest tied-score group with maximum TP also has smallest FP and
        # highest threshold; predict-none wins if every eligible TP is zero.
        if best_tp:
            chosen = int(eligible[np.flatnonzero(tp[eligible] == best_tp)[0]])
            threshold = float(sorted_p[ends[chosen]])
    result = operating_point(y, p, threshold)
    if not result["meets_five_percent_fpr"]:
        raise AssertionError("selected threshold violated exact FPR contract")
    return result


def signal_report(dataset, split, predictions, validation_selection=None):
    import numpy as np
    y = target_view(DATA / dataset / f"{split}.ghb")
    if predictions.stat().st_size != y.size * 8:
        raise ValueError("prediction extent differs from fixture")
    p = np.memmap(predictions, dtype="<f8", mode="r", shape=y.shape)
    fixed = operating_point(y, p, .5)
    if split == "validation":
        selected_point = choose_threshold(y, p)
        origin = "this validation case only"
    else:
        selected_point = operating_point(y, p, validation_selection["threshold"])
        origin = validation_selection["validation_case"]
    return {"scope": "binary" if y.shape[1] == 1 else "pooled micro over every row and all outputs",
            "fixed_threshold_0_5": fixed, "validation_selected_threshold": selected_point,
            "threshold_origin": origin, "test_threshold_was_tuned": False,
            "fixture_sha256": digest(DATA / dataset / f"{split}.ghb"), "predictions_sha256": digest(predictions)}


def training_contract(capture, measured):
    if measured["backend"] != "cuda" or measured["prediction_backend"] != "cuda":
        raise RuntimeError("non-CUDA production path")
    expected = {**capture["config"], "optimization_order": capture["optimization_order"]}
    # The pre-existing benchmark does not expose export width in parameters.
    # Its actual CLI argument remains captured; every optimizer requests 16.
    if any(measured["parameters"].get(k) != v for k, v in expected.items() if k != "tree_export_batch"):
        raise RuntimeError("recorded trainer parameters differ from request")


def execute(name, dataset, split, config, order, root, binary, expected_sources, validation_selection=None):
    directory = root / name
    directory.mkdir()
    cmd = command(binary, config, order, dataset, split, directory / "result")
    capture = {"name": name, "dataset": dataset, "split": split, "config": config, "optimization_order": order,
               "command": cmd, "cwd": str(WORKSPACE), "source_sha256": sources(), "binary_sha256": digest(binary),
               "training_fixture_sha256": digest(DATA / dataset / "train.ghb"),
               "evaluation_fixture_sha256": digest(DATA / dataset / f"{split}.ghb"),
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "validation_selection": validation_selection}
    if capture["source_sha256"] != expected_sources:
        raise RuntimeError("benchmark source changed after protocol registration")
    capture_path = directory / "capture.json"
    write(capture_path, capture)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.perf_counter()
    with (directory / "stdout").open("x") as out, (directory / "stderr").open("x") as err:
        process = subprocess.run(cmd, cwd=WORKSPACE, env=env, stdout=out, stderr=err, check=False)
    capture.update(returncode=process.returncode, process_wall_ms=1000 * (time.perf_counter() - start),
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   binary_unchanged=capture["binary_sha256"] == digest(binary),
                   stdout_sha256=digest(directory / "stdout"), stderr_sha256=digest(directory / "stderr"))
    if process.returncode == 0 and capture["binary_unchanged"]:
        try:
            training_contract(capture, read(directory / "result/metrics.json"))
        except Exception as error:
            capture["contract_error"] = f"{type(error).__name__}: {error}"
    if process.returncode == 0 and capture["binary_unchanged"] and "contract_error" not in capture:
        quality_command = [sys.executable, str(SEALED / "evaluate.py"), "--train", str(DATA / dataset / "train.ghb"),
                           "--evaluation", str(DATA / dataset / f"{split}.ghb"), "--predictions", str(directory / "result/predictions.f64"),
                           "--output", str(directory / "quality.json")]
        start = time.perf_counter()
        with (directory / "quality.stdout").open("x") as out, (directory / "quality.stderr").open("x") as err:
            quality = subprocess.run(quality_command, cwd=WORKSPACE, env=env, stdout=out, stderr=err, check=False)
        capture.update(quality_command=quality_command, quality_returncode=quality.returncode,
                       quality_process_wall_ms=1000 * (time.perf_counter() - start))
        if quality.returncode == 0:
            try:
                start = time.perf_counter()
                signal = signal_report(dataset, split, directory / "result/predictions.f64", validation_selection)
                write(directory / "signal.json", signal)
                capture.update(signal_returncode=0, signal_process_wall_ms=1000 * (time.perf_counter() - start))
            except Exception as error:
                capture.update(signal_returncode=1, signal_error=f"{type(error).__name__}: {error}")
            capture["artifact_sha256"] = {str(p.relative_to(directory)): digest(p) for p in
                [directory / "quality.json", directory / "result/metrics.json", directory / "result/predictions.f64", directory / "result/model.ghb",
                 directory / "quality.stdout", directory / "quality.stderr"]}
            if (directory / "signal.json").exists():
                capture["artifact_sha256"]["signal.json"] = digest(directory / "signal.json")
    # A receipt is finalized in place; raw result files are never overwritten.
    capture_path.write_text(json.dumps(capture, indent=2, allow_nan=False) + "\n")
    ok = capture["returncode"] == 0 and capture["binary_unchanged"] and capture.get("quality_returncode") == 0 and capture.get("signal_returncode") == 0
    print(name, "passed" if ok else "FAILED", flush=True)
    return ok


@lru_cache(maxsize=None)
def audited(directory):
    capture = read(directory / "capture.json")
    if capture.get("returncode") != 0 or capture.get("quality_returncode") != 0 or capture.get("signal_returncode") != 0 or capture.get("binary_unchanged") is not True:
        raise RuntimeError(f"incomplete/failed candidate: {directory}")
    if capture["source_sha256"] != sources() or capture["binary_sha256"] != digest(capture["command"][0]):
        raise RuntimeError("source or binary changed since capture")
    if capture["training_fixture_sha256"] != digest(DATA / capture["dataset"] / "train.ghb") or capture["evaluation_fixture_sha256"] != digest(DATA / capture["dataset"] / (capture["split"] + ".ghb")):
        raise RuntimeError("fixture changed since capture")
    for filename, sha in {**capture["artifact_sha256"], "stdout": capture["stdout_sha256"], "stderr": capture["stderr_sha256"]}.items():
        if digest(directory / filename) != sha:
            raise RuntimeError("captured artifact changed: " + str(directory / filename))
    quality, measured, signal = [read(directory / p) for p in ("quality.json", "result/metrics.json", "signal.json")]
    if quality["fixture_sha256"] != capture["evaluation_fixture_sha256"] or quality["training_fixture_sha256"] != capture["training_fixture_sha256"] or quality["predictions_sha256"] != capture["artifact_sha256"]["result/predictions.f64"]:
        raise RuntimeError("quality identity differs from capture")
    training_contract(capture, measured)
    return capture, quality, measured, signal


def completed(root, expected):
    receipt = read(root / "summary.json")
    if (root / ".incomplete").exists() or receipt.get("jobs") != expected or receipt.get("passed") != expected or receipt.get("failed") != []:
        raise RuntimeError(f"campaign lacks its complete {expected}/{expected} receipt: {root}")


def selections(validation):
    completed(validation, 24)
    result = {}
    for dataset in DATASETS:
        result[dataset] = {}
        for order in ORDERS:
            candidates = []
            for index, config in enumerate(grid(dataset)):
                name = f"validation-{dataset}-g{index}-o{order}"
                capture, quality, measured, signal = audited(validation / name)
                if capture["config"] != config or capture["optimization_order"] != order or capture["split"] != "validation" or capture["dataset"] != dataset:
                    raise RuntimeError("validation scope/config differs")
                candidates.append((-quality["metrics"][SELECTION[dataset]], quality["metrics"]["log_loss"],
                                   measured["timing_ms"]["training_wall"], index, name, signal))
            score, loss, timing, index, name, signal = min(candidates)
            result[dataset][str(order)] = {"config": grid(dataset)[index], "grid_index": index,
                "validation_case": name, "selection_metric": SELECTION[dataset], "selection_value": -score,
                "validation_log_loss": loss, "validation_training_wall_ms": timing,
                "threshold": signal["validation_selected_threshold"]["threshold"],
                "validation_operating_point": signal["validation_selected_threshold"]}
    return result


def metric_gate(reference, candidate):
    # Import the sealed gate under a unique name, with its own sibling imports.
    if str(SEALED) not in sys.path:
        sys.path.insert(0, str(SEALED))
    spec = importlib.util.spec_from_file_location("sealed_higher_order_metric_gate", SEALED / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.strict_metric_gate(reference, candidate)


def matched_comparison(reference, candidate):
    rc, rq, rm, rs = audited(reference)
    cc, cq, cm, cs = audited(candidate)
    if rc["optimization_order"] != 2 or any(rc[k] != cc[k] for k in ("config", "dataset", "split", "training_fixture_sha256", "evaluation_fixture_sha256", "binary_sha256")):
        raise RuntimeError("order comparison lacks a matched order-2 control")
    signal_deltas = {}
    for point in ("fixed_threshold_0_5", "validation_selected_threshold"):
        signal_deltas[point] = {key: cs[point][key] - rs[point][key] for key in ("recall", "false_positive_rate", "precision", "f1")}
    return {"reference_case": reference.name, "candidate_case": candidate.name, "comparison": "different optimizers at matched hyperparameters; not an exact-preservation claim",
            "strict_all_metrics_nonregression": metric_gate(rq["metrics"], cq["metrics"]),
            "signal_delta_candidate_minus_reference": signal_deltas,
            "strict_signal_nonregression": all(delta[key] >= 0 for delta in signal_deltas.values() for key in ("recall", "precision", "f1")) and all(delta["false_positive_rate"] <= 0 for delta in signal_deltas.values()),
            "training_time_ratio_reference_over_candidate": rm["timing_ms"]["training_wall"] / cm["timing_ms"]["training_wall"]}


def summarize(validation, test, output, screen=None):
    completed(test, 18)
    if screen is not None:
        completed(screen, 8)
    expected = selections(validation)
    if read(test / "selection.json") != expected:
        raise RuntimeError("test configuration/threshold selection differs from validation")
    result = {"selection": expected, "datasets": {}, "matched_validation_comparisons": [], "matched_test_comparisons": [],
              "coarse_validation_time_to_quality": {}, "interpretation": "exploratory reused-test follow-up; no data-uncertainty intervals or claims of fastest/superior generalization",
              "source_sha256": sources(), "validation_root": str(validation), "test_root": str(test)}
    if screen is not None:
        screening = {"root": str(screen), "cases": [], "matched_clipped_comparisons": [],
                     "unclipped_reference_scope": "order 2 with previous unclipped semantics; not a matched optimizer control for clipped runs"}
        for dataset in DATASETS:
            for suffix in ("o2-clip1", "o3-clip1", "o4-clip1", "o2-unclipped"):
                directory = screen / f"screen-{dataset}-{suffix}"
                capture, quality, measured, signal = audited(directory)
                screening["cases"].append({"case": directory.name, "capture_sha256": digest(directory / "capture.json"),
                    "config": capture["config"], "optimization_order": capture["optimization_order"], "metrics": quality["metrics"],
                    "signal": signal, "timing_ms": measured["timing_ms"], "memory": measured["memory"]})
            for order in (3, 4):
                screening["matched_clipped_comparisons"].append(matched_comparison(screen / f"screen-{dataset}-o2-clip1", screen / f"screen-{dataset}-o{order}-clip1"))
        result["screening"] = screening
    for dataset in DATASETS:
        result["datasets"][dataset] = {}
        target = expected[dataset]["2"]["selection_value"]
        target_record = {"metric": SELECTION[dataset], "target": target, "definition": "best attainable order-2 score in this clipped, bounded validation grid",
                         "limitation": "independent coarse round/lambda checkpoints; no interpolation, extrapolation or continuous-trajectory claim", "orders": {}}
        for index in range(4):
            for order in (3, 4):
                result["matched_validation_comparisons"].append(matched_comparison(validation / f"validation-{dataset}-g{index}-o2", validation / f"validation-{dataset}-g{index}-o{order}"))
        for order in ORDERS:
            observations = []
            for index in range(4):
                capture, quality, measured, signal = audited(validation / f"validation-{dataset}-g{index}-o{order}")
                observations.append({"case": capture["name"], "rounds": capture["config"]["rounds"], "l2": capture["config"]["l2"],
                    "score": quality["metrics"][SELECTION[dataset]], "training_wall_ms": measured["timing_ms"]["training_wall"]})
            achieved = sorted((v for v in observations if v["score"] >= target), key=lambda v: (v["rounds"], v["training_wall_ms"], v["l2"]))
            target_record["orders"][str(order)] = {"observations": observations, "reached": bool(achieved), "earliest_sampled_round": achieved[0] if achieved else None}
            runs = []
            for repetition in range(3):
                directory = test / f"test-{dataset}-r{repetition}-o{order}"
                capture, quality, measured, signal = audited(directory)
                if capture["config"] != expected[dataset][str(order)]["config"] or capture["validation_selection"] != expected[dataset][str(order)]:
                    raise RuntimeError("test selected configuration/threshold differs")
                runs.append({"case": directory.name, "capture_sha256": digest(directory / "capture.json"), "metrics": quality["metrics"],
                    "signal": signal, "timing_ms": measured["timing_ms"], "memory": measured["memory"],
                    "training_loss_initial": measured["training_loss_initial"], "training_loss_final": measured["training_loss_final"]})
            entry = {"runs": runs, "selected": expected[dataset][str(order)]}
            for name in ("training_wall", "prediction_wall"):
                values = [r["timing_ms"][name] for r in runs]
                entry[name + "_ms"] = {"median": statistics.median(values), "min": min(values), "max": max(values), "raw": values}
            entry["metric_medians"] = {key: statistics.median(r["metrics"][key] for r in runs) for key, value in runs[0]["metrics"].items() if isinstance(value, (int, float))}
            entry["signal_medians"] = {point: {key: statistics.median(r["signal"][point][key] for r in runs) for key in ("recall", "precision", "false_positive_rate", "f1")} for point in ("fixed_threshold_0_5", "validation_selected_threshold")}
            result["datasets"][dataset][str(order)] = entry
            if order != 2 and expected[dataset]["2"]["config"] == expected[dataset][str(order)]["config"]:
                for repetition in range(3):
                    result["matched_test_comparisons"].append(matched_comparison(test / f"test-{dataset}-r{repetition}-o2", test / f"test-{dataset}-r{repetition}-o{order}"))
        result["coarse_validation_time_to_quality"][dataset] = target_record
    output.mkdir(parents=True, exist_ok=False)
    write(output / "summary.json", result)
    lines = ["# Higher-order exploratory comparison", "", result["interpretation"], "", "All main runs use leaf clipping 1, including the order-2 control. Selection maximizes validation AP (MAGIC) or macro AP (Delicious), then minimizes log loss and synchronized training wall time. Test thresholds are frozen from the selected validation model; test FPR is measured, not constrained retrospectively.", "",
             "| Dataset | Order | Rounds / lambda | Train ms | Prediction ms | AP / macro AP | Log loss | Recall at validation-selected threshold | Measured test FPR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, entries in result["datasets"].items():
        for order, entry in entries.items():
            c, q, point = entry["selected"]["config"], entry["metric_medians"], entry["signal_medians"]["validation_selected_threshold"]
            lines.append(f"| {dataset} | {order} | {c['rounds']} / {c['l2']} | {entry['training_wall_ms']['median']:.3f} | {entry['prediction_wall_ms']['median']:.3f} | {q[SELECTION[dataset]]:.8g} | {q['log_loss']:.8g} | {point['recall']:.8g} | {point['false_positive_rate']:.8g} |")
    pairs = result["matched_validation_comparisons"] + result["matched_test_comparisons"]
    lines += ["", f"Matched optimizer comparisons: {len(pairs)}; strict all-common-metric failures: {sum(not p['strict_all_metrics_nonregression']['passed'] for p in pairs)}; additional signal-metric failures: {sum(not p['strict_signal_nonregression'] for p in pairs)}. All failures and metric deltas remain in JSON. Unmatched selected configurations are not treated as matched optimizer controls.", "",
              "Coarse time-to-quality observations, complete metric values, threshold confusion counts, timing ranges, raw repeats and memory scopes remain in JSON. These are bounded tuning results on previously inspected held-out data, not fresh confirmation or saturated model-quality rankings.", ""]
    with (output / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "screen", "validation", "test", "summarize"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--screen-root", type=Path)
    parser.add_argument("--custom-binary", type=Path, default=WORKSPACE / "build/booster-higher-order/ghb_real_bench")
    args = parser.parse_args()
    root, binary = args.run_root.resolve(), args.custom_binary.resolve()
    if args.stage == "summarize":
        if not args.validation_root or not args.test_root:
            parser.error("summarize requires --validation-root and --test-root")
        summarize(args.validation_root.resolve(), args.test_root.resolve(), root, args.screen_root.resolve() if args.screen_root else None)
        return
    if args.stage == "test" and not args.validation_root:
        parser.error("test requires --validation-root")
    selected = selections(args.validation_root.resolve()) if args.stage == "test" else None
    registered = protocol(binary)
    registered.update(stage=args.stage, argv=sys.argv, validation_root=str(args.validation_root.resolve()) if args.validation_root else None)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "protocol.json", registered)
    if args.stage == "plan":
        print(json.dumps(registered, indent=2))
        return
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if selected is not None:
        write(root / "selection.json", selected)
    jobs = []
    for dataset in DATASETS:
        if args.stage == "screen":
            for order in ORDERS:
                jobs.append((f"screen-{dataset}-o{order}-clip1", dataset, "validation", screen_config(dataset), order, None))
            jobs.append((f"screen-{dataset}-o2-unclipped", dataset, "validation", {**screen_config(dataset), "max_leaf_value": 0}, 2, None))
        elif args.stage == "validation":
            for index, config in enumerate(grid(dataset)):
                for order in ORDERS if index % 2 == 0 else ORDERS[::-1]:
                    jobs.append((f"validation-{dataset}-g{index}-o{order}", dataset, "validation", config, order, None))
        else:
            for repetition in range(3):
                for order in ORDERS[repetition:] + ORDERS[:repetition]:
                    choice = selected[dataset][str(order)]
                    jobs.append((f"test-{dataset}-r{repetition}-o{order}", dataset, "test", choice["config"], order, choice))
    write(root / "jobs.json", [{"name": name, "command": command(binary, config, order, dataset, split, root / name / "result")} for name, dataset, split, config, order, choice in jobs])
    (root / ".incomplete").write_text("Serial jobs pending; preserve until every receipt completes.\n")
    outcomes = []
    for name, dataset, split, config, order, choice in jobs:
        if digest(binary) != registered["binary_sha256"]:
            raise RuntimeError("binary changed after protocol registration")
        outcomes.append((name, execute(name, dataset, split, config, order, root, binary, registered["source_sha256"], choice)))
    write(root / "summary.json", {"jobs": len(outcomes), "passed": sum(ok for _, ok in outcomes), "failed": [name for name, ok in outcomes if not ok]})
    (root / ".incomplete").unlink()
    if not all(ok for _, ok in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
