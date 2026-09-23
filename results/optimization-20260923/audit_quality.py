#!/usr/bin/env python3
"""Offline, CPU-only paired validation audit; never launches training or CUDA.

Contract/selection recorded before implementation (2026-09-23): accept native
metrics.json/predictions.f64/model.ghb artifacts for matched independent-binary
training on the named train/validation fixtures (default: full Delicious).
Recompute the sealed aggregate metrics, add every label's log loss, Brier,
accuracy and ROC AUC, and use the sealed zero-allowance gate. AUC for a label
with only one class remains null; all its other metrics still participate.
Select the operating threshold once from REFERENCE validation predictions,
then apply that same threshold to candidate and baseline repeats. Also audit
fixed 0.5. Reuse the existing threshold algorithm and portable model reader.

Preserve artifact/fixture/evaluator hashes, exact-bit/numeric comparisons and
all failed gates. Baseline repeats describe run variation; they never subtract
an allowance from candidate regressions. File/shape self-consistency, metric
nonregression, exact equivalence and provenance are separate claims. This
helper cannot attest historical training inputs from metrics metadata alone.
No approximation or tolerance is admitted into a zero-allowance quality gate.

Validation plan: CPU-only self-comparison of saved artifacts, a controlled
prediction degradation that must fail despite unchanged valid metadata, and
small direct checks for undefined AUC, signed zero and threshold reuse.
Output is an exclusive new JSON file. Exit 0: candidate quality gates pass;
1: recorded candidate quality regression; 2: invalid input/evidence. Baseline
control gates remain independent report fields. Exact equivalence is reported
separately and is never implied by exit 0. No artifact is modified.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT / "results/booster-level-batch-20260922/real"
DATA = SEALED.parent / "data/fixtures/delicious"
SOURCE_PATHS = [Path(__file__).resolve(), SEALED / "evaluate.py", SEALED / "summarize.py",
                SEALED / "fixtures.py", SEALED / "campaign.py",
                ROOT / "training/tools/higher_order_campaign.py",
                ROOT / "results/booster-resident-20260922/audit_quality.py"]
sys.path.insert(0, str(SEALED))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


EVALUATE = module("paired_sealed_evaluate", SOURCE_PATHS[1])
SUMMARY = module("paired_sealed_summary", SOURCE_PATHS[2])
SIGNAL = module("paired_signal_reference", SOURCE_PATHS[5])
MODEL = module("paired_model_reference", SOURCE_PATHS[6])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": SIGNAL.digest(path), "bytes": path.stat().st_size}


def load_json(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=MODEL.no_duplicates,
                      parse_constant=invalid_constant)


def fixture(path):
    path = Path(path).resolve()
    # This existing loader checks both extent and the binary-objective header.
    targets = SIGNAL.target_view(path)
    with path.open("rb") as stream:
        _, _, rows, features, outputs, objective, classes = SIGNAL.HEADER.unpack(stream.read(SIGNAL.HEADER.size))
    require(rows > 0 and features > 0 and outputs > 1, "expected nonempty multilabel validation fixture")
    require(np.isfinite(targets).all() and np.isin(targets, [0, 1]).all(), "invalid binary targets")
    return targets, {"rows": rows, "features": features, "outputs": outputs,
                     "objective": objective, "classes": classes, **artifact(path)}


def labels(y, prediction):
    """Same FP64 clipping/formulas as sealed aggregate; AUC keeps undefined nulls."""
    result = []
    for index in range(y.shape[1]):
        truth, p = np.asarray(y[:, index], dtype=np.float64), prediction[:, index]
        positive = int(np.count_nonzero(truth))
        clipped = np.clip(p, EVALUATE.EPSILON, 1 - EVALUATE.EPSILON)
        result.append({"label": index, "positives": positive, "negatives": len(truth) - positive,
                       "log_loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))),
                       "brier": float(np.mean(np.square(p - truth))),
                       "accuracy": float(np.mean((p >= .5) == truth)),
                       "roc_auc": float(EVALUATE.skm.roc_auc_score(truth, p)) if 0 < positive < len(truth) else None})
    return result


def load_case(directory, y, train_info, validation_info):
    directory = Path(directory).resolve()
    artifacts = {name: artifact(directory / name) for name in ("metrics.json", "predictions.f64", "model.ghb")}
    measured = load_json(directory / "metrics.json")
    require(measured.get("implementation") == "custom" and measured.get("backend") == "cuda"
            and measured.get("prediction_backend") == "cuda", "expected custom CUDA training/prediction metadata")
    for key, expected in (("objective", 1), ("train_rows", train_info["rows"]),
                          ("evaluation_rows", validation_info["rows"]), ("features", validation_info["features"]),
                          ("outputs", validation_info["outputs"])):
        require(measured.get(key) == expected, f"{directory}: {key} differs from named fixture")
    require(artifacts["predictions.f64"]["bytes"] == y.size * 8, "prediction byte extent differs")
    prediction = np.memmap(directory / "predictions.f64", dtype="<f8", mode="r", shape=y.shape)
    require(np.isfinite(prediction).all() and (prediction >= 0).all() and (prediction <= 1).all(),
            "prediction probability domain differs")
    model = MODEL.read_model(directory / "model.ghb")
    require(model["objective"] == 1 and model["outputs"] == y.shape[1]
            and len(model["features"]) == validation_info["features"], "saved model shape differs")
    require(len(model["trees"]) == measured["trees"]
            and sum(len(nodes) for _, nodes in model["trees"]) == measured["nodes"], "model tree/node metadata differs")
    require(artifacts["model.ghb"]["bytes"] == measured["model_bytes"], "model byte metadata differs")
    rounds = measured["parameters"]["rounds"]
    history = np.asarray(measured["training_loss"], dtype=np.float64)
    require(type(rounds) is int and rounds > 0 and history.shape == (rounds + 1,)
            and np.isfinite(history).all() and (history >= 0).all(),
            "training history extent/finite/nonnegative contract differs")
    require(history[0] == measured["training_loss_initial"] and history[-1] == measured["training_loss_final"],
            "training loss endpoint metadata differs")
    return {"directory": str(directory), "artifacts": artifacts, "measured": measured,
            "prediction": prediction, "model": model,
            "aggregate": EVALUATE.metrics(y, prediction, 1), "labels": labels(y, prediction),
            "self_consistency": {"passed": True,
                "scope": "artifact extents, finite domains, saved model structure, metadata counts and loss endpoints",
                "independent_saved_model_prediction_check": "not performed by this helper",
                "does_not_establish": "paired equivalence, historical fixture identity, or quality nonregression"}}


def exact_values(before, after):
    before, after = np.asarray(before, dtype="<f8"), np.asarray(after, dtype="<f8")
    require(before.shape == after.shape, "exact array shapes differ")
    require(np.isfinite(before).all() and np.isfinite(after).all(), "exact arrays must be finite")
    bits = before.view("<u8") != after.view("<u8")
    diff = np.abs(after - before)
    indices = np.flatnonzero(bits)
    # Complete originals remain at the recorded, hashed artifact paths.
    examples = [{"flat_index": int(i), "reference_hex": float(before.flat[i]).hex(),
                 "candidate_hex": float(after.flat[i]).hex()} for i in indices[:32]]
    return {"values": int(before.size), "bitwise_equal": not bool(bits.any()),
            "different_bits": int(bits.sum()), "different_numeric_values": int(np.count_nonzero(before != after)),
            "max_absolute_difference": float(diff.max()) if diff.size else 0.,
            "mean_absolute_difference": float(diff.mean()) if diff.size else 0.,
            "first_bit_differences": examples, "examples_truncated": len(indices) > len(examples)}


def model_comparison(reference, candidate):
    a, b = reference["model"], candidate["model"]
    def quantization_bits(model):
        return b"".join(struct.pack("<II", kind, len(values)) + struct.pack("<" + "f" * len(values), *values)
                        for kind, values in model["features"])
    same_trees = len(a["trees"]) == len(b["trees"])
    changed, only_a, only_b, values = [], [], [], []
    tree_details = []
    for index in range(max(len(a["trees"]), len(b["trees"]))):
        if index >= len(a["trees"]) or index >= len(b["trees"]):
            tree_details.append({"tree": index, "status": "missing_reference" if index >= len(a["trees"]) else "missing_candidate"})
            continue
        (old_output, old), (new_output, new) = a["trees"][index], b["trees"][index]
        structure_changes = 0
        for path in sorted(set(old) | set(new)):
            identity = {"tree": index, "path": path, "reference_output": old_output, "candidate_output": new_output}
            if path not in new:
                only_a.append(identity)
            elif path not in old:
                only_b.append(identity)
            else:
                left, right = old[path], new[path]
                if old_output != new_output or left[:5] != right[:5]:
                    changed.append({**identity, "reference": left[:5], "candidate": right[:5]})
                    structure_changes += 1
                if struct.pack("<d", left[5]) != struct.pack("<d", right[5]):
                    values.append({**identity, "reference_hex": float(left[5]).hex(),
                                   "candidate_hex": float(right[5]).hex(), "absolute_difference": abs(left[5] - right[5]),
                                   "matched_node_semantics": old_output == new_output and left[:5] == right[:5]})
        if old_output != new_output or set(old) != set(new) or structure_changes:
            tree_details.append({"tree": index, "reference_nodes": len(old), "candidate_nodes": len(new),
                                 "reference_output": old_output, "candidate_output": new_output,
                                 "changed_structural_nodes": structure_changes})
    topology_equal = same_trees and not tree_details
    return {"serialized_bytes_equal": reference["artifacts"]["model.ghb"]["sha256"] == candidate["artifacts"]["model.ghb"]["sha256"],
            "topology_equal": topology_equal, "quantization_numeric_equal": a["features"] == b["features"],
            "quantization_bitwise_equal": quantization_bits(a) == quantization_bits(b),
            "base_scores": exact_values(a["bases"], b["bases"]),
            "trees_with_structural_changes": tree_details, "structural_node_changes": changed,
            "node_paths_only_in_reference": only_a, "node_paths_only_in_candidate": only_b,
            "matched_node_value_bit_changes": values,
            "node_values_bitwise_equal_on_identical_topology": topology_equal and not values,
            "node_value_comparison_scope": "same tree index and L/R path; semantics flag separates structurally changed nodes"}


def label_gate(reference, candidate):
    output, failures = [], []
    for before, after in zip(reference, candidate, strict=True):
        require(all(before[k] == after[k] for k in ("label", "positives", "negatives")), "label target identity differs")
        names = ["log_loss", "brier", "accuracy"]
        require((before["roc_auc"] is None) == (after["roc_auc"] is None), "AUC applicability differs")
        if before["roc_auc"] is not None:
            names.append("roc_auc")
        gate = SUMMARY.strict_metric_gate({k: before[k] for k in names}, {k: after[k] for k in names})
        if not gate["passed"]:
            failures.append(before["label"])
        output.append({"label": before["label"], "positives": before["positives"], "negatives": before["negatives"],
                       "reference": {k: before[k] for k in ("log_loss", "brier", "accuracy", "roc_auc")},
                       "candidate": {k: after[k] for k in ("log_loss", "brier", "accuracy", "roc_auc")},
                       "auc_applicable": before["roc_auc"] is not None,
                       "undefined_auc_reason": None if before["roc_auc"] is not None else "one target class absent; no invented AUC",
                       "zero_allowance_gate": gate})
    return {"allowance": 0, "passed": not failures, "labels_checked": len(output),
            "auc_labels_checked": sum(row["auc_applicable"] for row in output),
            "regressed_labels": failures, "all_labels": output}


def signal_comparison(y, before, after, threshold):
    result = {}
    for name, value in (("fixed_0_5", .5), ("reference_validation_selected", threshold)):
        old, new = SIGNAL.operating_point(y, before, value), SIGNAL.operating_point(y, after, value)
        delta = {key: new[key] - old[key] for key in ("recall", "false_positive_rate", "precision", "f1")}
        failures = [key for key, difference in delta.items()
                    if key == "false_positive_rate" and difference > 0]
        failures += [key for key, difference in delta.items() if key != "false_positive_rate" and difference < 0]
        result[name] = {"reference": old, "candidate": new, "delta_candidate_minus_reference": delta,
                        "zero_allowance_gate": {"allowance": 0, "passed": not failures, "regressed_metrics": failures}}
    return {"scope": "pooled micro over every row and label", "points": result,
            "passed": all(x["zero_allowance_gate"]["passed"] for x in result.values())}


def compare(reference, candidate, y, threshold, allow_policy_difference):
    old, new = reference["measured"], candidate["measured"]
    require(old["device"] == new["device"], "reported devices differ")
    before_params, after_params = dict(old["parameters"]), dict(new["parameters"])
    policy_before, policy_after = before_params.pop("split_policy", None), after_params.pop("split_policy", None)
    require(before_params == after_params, "training parameters differ beyond split_policy")
    if not allow_policy_difference:
        require(policy_before == policy_after, "baseline-repeat split policies differ")
    aggregate = SUMMARY.strict_metric_gate(reference["aggregate"], candidate["aggregate"])
    by_label = label_gate(reference["labels"], candidate["labels"])
    signal = signal_comparison(y, reference["prediction"], candidate["prediction"], threshold)
    prediction = exact_values(reference["prediction"], candidate["prediction"])
    prediction["per_label"] = [{"label": i,
        "different_bits": int(np.count_nonzero(reference["prediction"][:, i].view("<u8") != candidate["prediction"][:, i].view("<u8"))),
        "max_absolute_difference": float(np.max(np.abs(reference["prediction"][:, i] - candidate["prediction"][:, i])))}
        for i in range(y.shape[1])]
    prediction["fixed_0_5_decisions_changed"] = int(np.count_nonzero((reference["prediction"] >= .5) != (candidate["prediction"] >= .5)))
    prediction["reference_threshold_decisions_changed"] = int(np.count_nonzero((reference["prediction"] >= threshold) != (candidate["prediction"] >= threshold)))
    model = model_comparison(reference, candidate)
    history = exact_values(old["training_loss"], new["training_loss"])
    quality_passed = aggregate["passed"] and by_label["passed"] and signal["passed"]
    return {"reference": reference["directory"], "candidate": candidate["directory"],
            "split_policy": {"reference": policy_before, "candidate": policy_after,
                             "missing_metadata": policy_before is None or policy_after is None},
            "aggregate_zero_allowance_gate": aggregate, "per_label_zero_allowance_gate": by_label,
            "signal": signal, "all_quality_gates_passed": quality_passed,
            "predictions": prediction, "model": model, "training_loss_history": history,
            "exact_saved_artifact_equivalence": {"predictions_and_serialized_model": prediction["bitwise_equal"] and model["serialized_bytes_equal"],
                "including_loss_history": prediction["bitwise_equal"] and model["serialized_bytes_equal"] and history["bitwise_equal"],
                "scope": "this saved pair only; no unobserved intermediate/device state claim"}}


def public_case(case):
    return {key: case[key] for key in ("directory", "artifacts", "measured", "aggregate", "self_consistency")}


def run(args, result):
    y, validation = fixture(args.validation_fixture)
    _, train = fixture(args.train_fixture)
    require(all(train[k] == validation[k] for k in ("features", "outputs", "objective", "classes")),
            "train/validation fixture shapes differ")
    result["fixtures"] = {"train": train, "validation": validation}
    reference = load_case(args.reference, y, train, validation)
    candidate = load_case(args.candidate, y, train, validation)
    selection = SIGNAL.choose_threshold(y, reference["prediction"])
    threshold = selection["threshold"]
    result["threshold_selection"] = {"origin": reference["directory"], "reference_validation_only": True,
        "same_threshold_applied_to_candidate_and_repeats": True, "selection": selection,
        "rule": "maximize pooled reference validation recall subject to exact 20*FP <= negatives; retain tied scores",
        "fixture_role": "caller-declared validation; this helper does not independently attest split-role provenance",
        "default_delicious_validation_path": args.validation_fixture.resolve() == (DATA / "validation.ghb").resolve()}
    result["reference"] = public_case(reference)
    result["candidate"] = public_case(candidate)
    result["comparison"] = compare(reference, candidate, y, threshold, True)
    repeats = []
    for directory in args.baseline_repeat:
        control = load_case(directory, y, train, validation)
        require(Path(control["directory"]) != Path(reference["directory"]), "baseline repeat must be a distinct case directory")
        repeats.append({"case": public_case(control), "comparison": compare(reference, control, y, threshold, False)})
    result["baseline_repeat_controls"] = {"provided": bool(repeats), "repeats": repeats,
        "all_quality_gates_passed": all(x["comparison"]["all_quality_gates_passed"] for x in repeats) if repeats else None,
        "interpretation": "directional comparisons to the same reference, never an allowance or excuse for candidate failures"}
    # Rehash all inputs after reading; mutations invalidate this audit.
    all_artifacts = [train, validation] + list(result["source_files"].values())
    for case in [reference, candidate] + [r["case"] for r in repeats]:
        all_artifacts += list(case["artifacts"].values())
    for entry in all_artifacts:
        require(SIGNAL.digest(entry["path"]) == entry["sha256"], "input changed during audit: " + entry["path"])
    result["inputs_unchanged_during_audit"] = True
    result["status"] = "pass" if result["comparison"]["all_quality_gates_passed"] else "quality_regression"
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-repeat", type=Path, action="append", default=[])
    parser.add_argument("--train-fixture", type=Path, default=DATA / "train.ghb")
    parser.add_argument("--validation-fixture", type=Path, default=DATA / "validation.ghb")
    args = parser.parse_args()
    result = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution": "CPU-only offline validation; no training, GPU, timing ranking or artifact modification",
        "source_files": {str(path.relative_to(ROOT)): artifact(path) for path in SOURCE_PATHS},
        "environment": {"python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "limitations": ["Metadata/shape checks do not attest which fixtures generated historical artifacts; campaign captures provide that provenance.",
                        "No independent saved-model prediction replay is performed here.",
                        "Exactness and quality are separate: matching quality can coexist with different prediction/model bits.",
                        "This validation split and these saved repeats do not establish generalization or universal performance."]}
    with args.output.open("x") as stream:
        try:
            code = run(args, result)
        except Exception as error:
            result["status"] = "invalid_evidence"
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            code = 2
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
