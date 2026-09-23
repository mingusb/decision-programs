#!/usr/bin/env python3
"""CPU evaluation and complete-metric comparison of saved prediction CSV files.

Targets: row_id,target[,weight]. Regression and multilabel also accept contiguous
row_id,target_0,...,target_K-1[,weight]. Predictions use row_id,prediction
for scalar regression/binary/multilabel, matching prediction_0,... columns for multi-output
regression/multilabel, and row_id,p0,...,pK-1 for multiclass. Row IDs must be unique,
nonempty, and in exactly the same order in both files. Binary targets are 0/1;
multiclass targets are integer class indices. All numeric values must be finite.

Weights are nonnegative with a finite, positive total. Regression outputs have
equal weight: overall RMSE is the root of mean squared error over rows and
outputs; overall MAE averages absolute error over rows and outputs. Every output
also has its own metrics. Binary accuracy uses probability >= 0.5. Multiclass
accuracy breaks ties at the lowest class index. AUC uses weighted concordance
with half credit for tied scores and is null if one class has zero total weight.
Multilabel metrics average independent binary outputs equally, with per-output
metrics retained. Multilabel accuracy is weighted elementwise (Hamming) accuracy.
Each multilabel probability is independent; row sums need not equal one.
Its aggregate AUC is null if any output's AUC is undefined; all defined per-output
AUCs still participate in comparison.

Multiclass probabilities must be in [0,1] and sum to 1 within an explicit absolute
tolerance (default 1e-6). Accepted vectors are divided by their row sum before
scoring. Log loss uses natural logs and clips probabilities to [1e-15,1-1e-15].
No output file is overwritten. Compare's allowance is absolute, in EACH metric's
native units: 0.01 accuracy/AUC means one percentage point; 0.01 RMSE means 0.01
target units. All applicable metrics, including each regression and multilabel
output, must meet it.
Comparison uses the decimal representation of recorded values, with no extra
implicit tolerance. Equality at the allowance passes. Exit codes: 0 success/pass,
1 metric regression, 2 invalid input or evidence.
Compare requires both reports' source CSVs to remain available: their hashes and
all metrics are rechecked before a comparison can pass.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

SCHEMA_VERSION = 1
EVALUATOR_VERSION = "1.0"
LOG_CLIP = 1e-15
DEFAULT_PROBABILITY_SUM_TOLERANCE = 1e-6
OBJECTIVES = ("regression", "binary", "multiclass", "multilabel")
LIMITATIONS = [
    "Metrics describe only the supplied predictions, targets, weights, and named split; they do not establish generalization to other data.",
    "This evaluator does not train a model or verify how the supplied predictions were produced.",
    "Calculations use finite double-precision floating-point values; endpoint log loss is explicitly clipped.",
]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
INTEGER = re.compile(r"[+-]?\d+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(value, name: str, *, lower=None, upper=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise EvaluationError(f"{name} is outside finite numeric range") from error
    if not math.isfinite(result) or (lower is not None and result < lower) or (upper is not None and result > upper):
        raise EvaluationError(f"{name} is outside its finite numeric bounds")
    return result


def csv_number(value: str, name: str) -> float:
    text = value.strip()
    if not NUMBER.fullmatch(text):
        raise EvaluationError(f"{name} must be a finite decimal number")
    return number(float(text), name)


def integer(value, name: str, *, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def require_name(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be nonempty")
    return value


def finite_sum(values, name: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise EvaluationError(f"{name} cannot be represented as a finite sum") from error
    return number(result, name)


def csv_table(data: bytes, name: str) -> tuple[list[str], list[list[str]]]:
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as error:
        raise EvaluationError(f"{name}: invalid UTF-8 CSV: {error}") from error
    if len(rows) < 2:
        raise EvaluationError(f"{name}: an exact header and at least one data row are required")
    header, body = rows[0], rows[1:]
    if not header or header[0] != "row_id" or len(header) != len(set(header)):
        raise EvaluationError(f"{name}: duplicate header columns or missing leading row_id")
    seen = set()
    for index, row in enumerate(body, 2):
        if len(row) != len(header):
            raise EvaluationError(f"{name}: row {index} has the wrong number of columns")
        if not row[0].strip() or row[0] in seen:
            raise EvaluationError(f"{name}: empty or duplicate row_id at row {index}")
        seen.add(row[0])
    return header, body


def settings_for(objective: str, tolerance: float) -> dict:
    result = {
        "log_clip_lower": LOG_CLIP if objective != "regression" else None,
        "log_clip_upper": 1.0 - LOG_CLIP if objective != "regression" else None,
        "log_base": "natural" if objective != "regression" else None,
        "binary_decision_threshold": 0.5 if objective in ("binary", "multilabel") else None,
        "binary_threshold_inclusive": True if objective in ("binary", "multilabel") else None,
        "auc_ties": "half_credit" if objective in ("binary", "multilabel") else None,
        "probability_sum_tolerance": tolerance if objective == "multiclass" else None,
        "probability_normalization": "divide_by_row_sum_after_tolerance_check" if objective == "multiclass" else None,
        "multiclass_argmax_ties": "lowest_class_index" if objective == "multiclass" else None,
        "regression_output_weighting": "equal" if objective == "regression" else None,
    }
    if objective == "multilabel":
        result.update(multilabel_output_weighting="equal", multilabel_auc="mean_if_all_outputs_defined")
    return result


def metric_specs(objective: str, outputs: int) -> dict[str, tuple[str, str, float | None]]:
    if objective == "regression":
        names = ["rmse", "mae"] + [f"{metric}_output_{index}" for index in range(outputs) for metric in ("rmse", "mae")]
        return {name: ("minimize", "target_units", None) for name in names}
    result = {"logloss": ("minimize", "nats", -math.log(LOG_CLIP)),
              "accuracy": ("maximize", "proportion", 1.0)}
    if objective in ("binary", "multilabel"):
        result.update(brier=("minimize", "squared_probability", 1.0), auc=("maximize", "proportion", 1.0))
    if objective == "multilabel":
        result.update({f"{name}_output_{index}": spec for index in range(outputs) for name, spec in list(result.items())})
    return result


def weighted_auc(labels: list[int], probabilities: list[float], weights: list[float]) -> float | None:
    positives = finite_sum((weight for label, weight in zip(labels, weights) if label == 1), "positive class weight")
    negatives = finite_sum((weight for label, weight in zip(labels, weights) if label == 0), "negative class weight")
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(probabilities, labels, weights), key=lambda row: row[0])
    terms, negative_before, correction = [], 0.0, 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        group = ordered[start:end]
        positive = finite_sum((weight / positives for _, label, weight in group if label == 1), "normalized positive tie weight")
        negative = finite_sum((weight / negatives for _, label, weight in group if label == 0), "normalized negative tie weight")
        terms.append(positive * (negative_before + 0.5 * negative))
        # Compensated accumulation avoids repeatedly losing small negative groups.
        increment = negative - correction
        updated = negative_before + increment
        correction = (updated - negative_before) - increment
        negative_before = updated
        start = end
    return min(1.0, max(0.0, finite_sum(terms, "AUC")))


def clipped_logloss(probability: float) -> float:
    return -math.log(min(1.0 - LOG_CLIP, max(LOG_CLIP, probability)))


def weighted_error_mean(errors: list[tuple[float, float]], total: float, power: int) -> float:
    """Return mean absolute error or root mean square without intermediate overflow.

    Keep powers of two separate until the final result. Merely computing
    weight/total first can underflow to zero even when a large error makes the
    final contribution representable; squaring the error can likewise overflow.
    """
    denominator, denominator_exponent = math.frexp(total)
    terms = []
    for error, weight in errors:
        if error == 0:
            continue
        magnitude, exponent = math.frexp(error)
        weight_magnitude, weight_exponent = math.frexp(weight)
        terms.append((magnitude ** power * weight_magnitude / denominator,
                      exponent * power + weight_exponent - denominator_exponent))
    if not terms:
        return 0.0
    maximum_exponent = max(exponent for _, exponent in terms)
    scaled = finite_sum((math.ldexp(magnitude, exponent - maximum_exponent) for magnitude, exponent in terms), "scaled weighted error")
    try:
        if power == 2:
            exponent, remainder = divmod(maximum_exponent, 2)
            result = math.ldexp(math.sqrt(math.ldexp(scaled, remainder)), exponent)
        else:
            result = math.ldexp(scaled, maximum_exponent)
    except OverflowError:
        # A convex mean cannot exceed its largest finite input. Rounding at
        # the very top of the floating-point range can cross that boundary.
        result = max(error for error, _ in errors)
    return min(result, max(error for error, _ in errors))


def calculate(target_data: bytes, prediction_data: bytes, objective: str, classes: int | None,
              tolerance: float) -> dict:
    if objective not in OBJECTIVES:
        raise EvaluationError("unsupported objective")
    tolerance = number(tolerance, "probability sum tolerance", lower=0.0)
    if tolerance >= 1:
        raise EvaluationError("probability sum tolerance must be less than one")
    if objective == "regression" and classes is not None:
        raise EvaluationError("--classes is not applicable to regression")
    if objective in ("binary", "multilabel"):
        if classes not in (None, 2) or isinstance(classes, bool):
            raise EvaluationError("binary classification has exactly two classes")
        classes = 2
    if objective == "multiclass":
        integer(classes, "multiclass --classes", minimum=2)
    target_header, target_rows = csv_table(target_data, "targets")
    prediction_header, prediction_rows = csv_table(prediction_data, "predictions")
    weighted = target_header[-1] == "weight"
    target_columns = target_header[1:-1] if weighted else target_header[1:]
    scalar = target_columns == ["target"]
    outputs = len(target_columns)
    if not scalar and (objective not in ("regression", "multilabel") or outputs < 1 or target_columns != [f"target_{i}" for i in range(outputs)]):
        raise EvaluationError("targets must use target or contiguous regression/multilabel target_0,... columns, followed by optional weight")
    if objective == "multiclass":
        expected_prediction_header = ["row_id"] + [f"p{i}" for i in range(classes)]
        outputs = classes
    else:
        expected_prediction_header = ["row_id"] + (["prediction"] if scalar else [f"prediction_{i}" for i in range(outputs)])
    if prediction_header != expected_prediction_header:
        raise EvaluationError("prediction columns do not match the objective and target output count")
    if len(target_rows) != len(prediction_rows) or any(a[0] != b[0] for a, b in zip(target_rows, prediction_rows)):
        raise EvaluationError("target and prediction row_id values must correspond exactly in order")
    weights = [csv_number(row[-1], "weight") if weighted else 1.0 for row in target_rows]
    if any(weight < 0 for weight in weights):
        raise EvaluationError("weights must be nonnegative")
    total = finite_sum(weights, "total weight")
    if total <= 0:
        raise EvaluationError("total weight must be positive")
    predictions = [[csv_number(value, "prediction") for value in row[1:]] for row in prediction_rows]
    if objective == "regression":
        targets = [[csv_number(value, "target") for value in row[1:1 + outputs]] for row in target_rows]
        squared_roots, absolute_means = [], []
        for index in range(outputs):
            errors = [(abs(target[index] - prediction[index]), weight) for target, prediction, weight in zip(targets, predictions, weights) if weight > 0]
            if any(not math.isfinite(error) for error, _ in errors):
                raise EvaluationError("a positive-weight regression error is outside finite numeric range")
            mae = weighted_error_mean(errors, total, 1)
            rmse = weighted_error_mean(errors, total, 2)
            squared_roots.append(number(rmse, "RMSE", lower=0))
            absolute_means.append(mae)
        overall_rmse = weighted_error_mean([(value, 1.0) for value in squared_roots], float(outputs), 2)
        values = {"rmse": overall_rmse, "mae": weighted_error_mean([(value, 1.0) for value in absolute_means], float(outputs), 1)}
        for index, (rmse, mae) in enumerate(zip(squared_roots, absolute_means)):
            values[f"rmse_output_{index}"] = rmse
            values[f"mae_output_{index}"] = mae
        target_summary = {"minimum": [min(row[i] for row in targets) for i in range(outputs)],
                          "maximum": [max(row[i] for row in targets) for i in range(outputs)]}
    elif objective == "multilabel":
        labels = []
        for row in target_rows:
            values = row[1:1 + outputs]
            if any(not INTEGER.fullmatch(value.strip()) or int(value) not in (0, 1) for value in values):
                raise EvaluationError("multilabel targets must be binary integer labels")
            labels.append([int(value) for value in values])
        if any(probability < 0 or probability > 1 for row in predictions for probability in row):
            raise EvaluationError("probabilities must be in [0,1]")
        per_output = []
        counts, class_weights = [], []
        for index in range(outputs):
            y, p = [row[index] for row in labels], [row[index] for row in predictions]
            losses = [clipped_logloss(probability if label else 1 - probability) for label, probability in zip(y, p)]
            per_output.append({
                "logloss": weighted_error_mean([(loss, weight) for loss, weight in zip(losses, weights) if weight > 0], total, 1),
                "accuracy": finite_sum((weight for label, probability, weight in zip(y, p, weights) if label == int(probability >= .5)), "correct weight") / total,
                "brier": weighted_error_mean([((probability - label) ** 2, weight) for label, probability, weight in zip(y, p, weights) if weight > 0], total, 1),
                "auc": weighted_auc(y, p, weights),
            })
            counts.append([y.count(label) for label in (0, 1)])
            class_weights.append([finite_sum((weight for value, weight in zip(y, weights) if value == label), "class weight") for label in (0, 1)])
        values = {name: None if any(metrics[name] is None for metrics in per_output)
                  else weighted_error_mean([(metrics[name], 1.0) for metrics in per_output], float(outputs), 1)
                  for name in ("logloss", "accuracy", "brier", "auc")}
        values.update({f"{name}_output_{index}": value for index, metrics in enumerate(per_output) for name, value in metrics.items()})
        target_summary = {"class_sample_counts": counts, "class_weights": class_weights}
    else:
        labels = []
        for row in target_rows:
            value = row[1].strip()
            if not INTEGER.fullmatch(value) or not 0 <= int(value) < classes:
                raise EvaluationError("classification target must be an integer class index in range")
            labels.append(int(value))
        if any(probability < 0 or probability > 1 for row in predictions for probability in row):
            raise EvaluationError("probabilities must be in [0,1]")
        if objective == "multiclass":
            for row in predictions:
                row_sum = finite_sum(row, "probability row sum")
                if row_sum <= 0 or abs(row_sum - 1.0) > tolerance:
                    raise EvaluationError("multiclass probability row sum exceeds the declared tolerance")
                row[:] = [probability / row_sum for probability in row]
            guesses = [max(range(classes), key=lambda index: row[index]) for row in predictions]
            losses = [clipped_logloss(row[label]) for row, label in zip(predictions, labels)]
        else:
            probabilities = [row[0] for row in predictions]
            guesses = [int(probability >= 0.5) for probability in probabilities]
            losses = [clipped_logloss(probability if label else 1.0 - probability) for probability, label in zip(probabilities, labels)]
        values = {"logloss": weighted_error_mean([(loss, weight) for loss, weight in zip(losses, weights) if weight > 0], total, 1),
                  "accuracy": finite_sum((weight for weight, guess, label in zip(weights, guesses, labels) if guess == label), "correct weight") / total}
        if objective == "binary":
            values.update(brier=weighted_error_mean([((probability - label) ** 2, weight) for weight, probability, label in zip(weights, probabilities, labels) if weight > 0], total, 1),
                          auc=weighted_auc(labels, probabilities, weights))
        target_summary = {"class_sample_counts": [labels.count(index) for index in range(classes)],
                          "class_weights": [finite_sum((weight for weight, label in zip(weights, labels) if label == index), "class weight") for index in range(classes)]}
    specs = metric_specs(objective, outputs)
    metrics = {name: {"value": values[name], "direction": direction, "unit": unit} for name, (direction, unit, _) in specs.items()}
    return {"objective": objective, "sample_count": len(target_rows), "outputs": outputs, "classes": classes,
            "weighting": "column" if weighted else "unit", "total_weight": total,
            "positive_weight_count": sum(weight > 0 for weight in weights),
            "weights_sha256": digest("\n".join(weight.hex() for weight in weights).encode()),
            "target_summary": target_summary, "settings": settings_for(objective, tolerance), "metrics": metrics}


def evaluate(targets: Path, predictions: Path, objective: str, dataset_id: str, split_id: str,
             classes: int | None = None, probability_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE) -> dict:
    dataset_id, split_id = require_name(dataset_id, "dataset ID"), require_name(split_id, "split ID")
    targets, predictions = Path(targets).resolve(), Path(predictions).resolve()
    target_data, prediction_data = targets.read_bytes(), predictions.read_bytes()
    calculated = calculate(target_data, prediction_data, objective, classes, probability_sum_tolerance)
    report = {"schema_version": SCHEMA_VERSION, "evaluator_version": EVALUATOR_VERSION,
              "evaluator_source_sha256": digest(Path(__file__).read_bytes()), "kind": "prediction_evaluation",
              "dataset_id": dataset_id, "split_id": split_id,
              "inputs": {"targets": {"path": str(targets), "sha256": digest(target_data)},
                         "predictions": {"path": str(predictions), "sha256": digest(prediction_data)}},
              **calculated, "limitations": LIMITATIONS}
    validate_report(report)
    return report


def validate_report(report: dict) -> dict:
    expected_keys = {"schema_version", "evaluator_version", "evaluator_source_sha256", "kind", "dataset_id", "split_id",
                     "inputs", "objective", "sample_count", "outputs", "classes", "weighting", "total_weight",
                     "positive_weight_count", "weights_sha256", "target_summary", "settings", "metrics", "limitations"}
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise EvaluationError("evaluation report fields differ from the supported schema")
    if type(report["schema_version"]) is not int or report["schema_version"] != SCHEMA_VERSION or report["evaluator_version"] != EVALUATOR_VERSION or report["kind"] != "prediction_evaluation":
        raise EvaluationError("unsupported evaluation report version or kind")
    for field in ("evaluator_source_sha256", "weights_sha256"):
        if not isinstance(report[field], str) or not SHA256.fullmatch(report[field]):
            raise EvaluationError(f"invalid {field}")
    for field in ("dataset_id", "split_id"):
        require_name(report[field], field)
    objective = report["objective"]
    if objective not in OBJECTIVES:
        raise EvaluationError("unsupported report objective")
    count = integer(report["sample_count"], "sample count", minimum=1)
    outputs = integer(report["outputs"], "output count", minimum=1)
    positive_count = integer(report["positive_weight_count"], "positive weight count", minimum=1)
    total = number(report["total_weight"], "total weight", lower=0)
    if total <= 0 or positive_count > count or report["weighting"] not in ("unit", "column"):
        raise EvaluationError("invalid weight metadata")
    if report["weighting"] == "unit" and (total != count or positive_count != count):
        raise EvaluationError("unit weight metadata is inconsistent")
    classes = report["classes"]
    if objective == "regression":
        if classes is not None:
            raise EvaluationError("regression report cannot declare classes")
    else:
        integer(classes, "class count", minimum=2)
        if (objective == "binary" and (classes != 2 or outputs != 1)) or (objective == "multilabel" and classes != 2) or (objective == "multiclass" and outputs != classes):
            raise EvaluationError("classification output count differs from objective/classes")
    settings = report["settings"]
    if not isinstance(settings, dict):
        raise EvaluationError("settings must be an object")
    tolerance = settings.get("probability_sum_tolerance") if objective == "multiclass" else DEFAULT_PROBABILITY_SUM_TOLERANCE
    tolerance = number(tolerance, "probability sum tolerance", lower=0)
    if tolerance >= 1 or settings != settings_for(objective, tolerance):
        raise EvaluationError("metric settings differ from the declared evaluator contract")
    if report["limitations"] != LIMITATIONS:
        raise EvaluationError("evaluation report omits its interpretation limits")
    summary = report["target_summary"]
    if not isinstance(summary, dict):
        raise EvaluationError("target summary must be an object")
    if objective == "regression":
        if set(summary) != {"minimum", "maximum"} or any(not isinstance(summary[key], list) or len(summary[key]) != outputs for key in summary):
            raise EvaluationError("regression target summary has the wrong dimensions")
        for low, high in zip(summary["minimum"], summary["maximum"]):
            if number(low, "minimum target") > number(high, "maximum target"):
                raise EvaluationError("minimum target exceeds maximum")
    else:
        if objective == "multilabel":
            if set(summary) != {"class_sample_counts", "class_weights"} or any(not isinstance(summary[key], list) or len(summary[key]) != outputs for key in summary):
                raise EvaluationError("multilabel target summary has the wrong output count")
            summaries = [{key: summary[key][index] for key in summary} for index in range(outputs)]
        else:
            summaries = [summary]
        for class_summary in summaries:
            validate_class_summary(class_summary, classes, count, total)
    specs = metric_specs(objective, outputs)
    if not isinstance(report["metrics"], dict) or set(report["metrics"]) != set(specs):
        raise EvaluationError("metric coverage differs from the objective/output contract")
    for name, (direction, unit, upper) in specs.items():
        metric = report["metrics"][name]
        if not isinstance(metric, dict) or set(metric) != {"value", "direction", "unit"} or metric["direction"] != direction or metric["unit"] != unit:
            raise EvaluationError(f"metric {name} direction/unit/schema differs")
        if objective == "multilabel" and (name == "auc" or name.startswith("auc_output_")):
            weights = summary["class_weights"] if name == "auc" else [summary["class_weights"][int(name.removeprefix("auc_output_"))]]
            null_auc = any(weight == 0 for pair in weights for weight in pair)
        else:
            null_auc = name == "auc" and any(weight == 0 for weight in summary["class_weights"])
        if null_auc:
            if metric["value"] is not None:
                raise EvaluationError("AUC must be null when a class has zero total weight")
        else:
            number(metric["value"], name, lower=0, upper=upper)
    inputs = report["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"targets", "predictions"}:
        raise EvaluationError("missing target/prediction artifact records")
    available, verified = {}, []
    for name, artifact in inputs.items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"} or not isinstance(artifact["path"], str) or not Path(artifact["path"]).is_absolute() or not isinstance(artifact["sha256"], str) or not SHA256.fullmatch(artifact["sha256"]):
            raise EvaluationError("invalid input artifact path/hash")
        path = Path(artifact["path"])
        if path.exists():
            if not path.is_file():
                raise EvaluationError("recorded input is not a file: " + str(path))
            data = path.read_bytes()
            if digest(data) != artifact["sha256"]:
                raise EvaluationError("recorded input hash changed: " + str(path))
            available[name] = data
            verified.append(name)
    recomputed = set(available) == {"targets", "predictions"}
    if recomputed:
        expected = calculate(available["targets"], available["predictions"], objective, classes, tolerance)
        if any(report[key] != value for key, value in expected.items()):
            raise EvaluationError("stored metrics or workload metadata disagree with recomputed input artifacts")
    return {"artifact_hashes_verified": sorted(verified), "metrics_recomputed": recomputed,
            "unavailable_artifacts": sorted(set(inputs) - set(available))}


def validate_class_summary(summary: dict, classes: int, count: int, total: float) -> None:
        if set(summary) != {"class_sample_counts", "class_weights"} or any(not isinstance(summary[key], list) or len(summary[key]) != classes for key in summary):
            raise EvaluationError("classification target summary has the wrong dimensions")
        class_counts = [integer(value, "class sample count") for value in summary["class_sample_counts"]]
        class_weights = [number(value, "class weight", lower=0, upper=total) for value in summary["class_weights"]]
        if sum(class_counts) != count or not math.isclose(finite_sum(class_weights, "class weight sum"), total, rel_tol=1e-12, abs_tol=0):
            raise EvaluationError("class counts or weights disagree with report totals")
        if any((class_count == 0 and weight != 0) for class_count, weight in zip(class_counts, class_weights)):
            raise EvaluationError("nonzero weight assigned to an absent class")


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def read_report(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    try:
        report = json.loads(data, object_pairs_hook=no_duplicate_keys,
                            parse_constant=lambda value: (_ for _ in ()).throw(EvaluationError("nonfinite JSON value: " + value)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError("invalid evaluation JSON: " + str(error)) from error
    return report, data


def compare(reference: Path, candidate: Path, max_loss_increase: float = 0.0) -> dict:
    allowance = number(max_loss_increase, "maximum loss increase", lower=0)
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    before, before_bytes = read_report(reference)
    after, after_bytes = read_report(candidate)
    verification = {"reference": validate_report(before), "candidate": validate_report(after)}
    if any(not result["metrics_recomputed"] for result in verification.values()):
        raise EvaluationError("comparison requires both source CSVs for each report; missing artifacts prevent verified metric comparison")
    identity = ("schema_version", "evaluator_version", "evaluator_source_sha256", "dataset_id", "split_id", "objective",
                "sample_count", "outputs", "classes", "weighting", "total_weight", "positive_weight_count",
                "weights_sha256", "target_summary", "settings")
    if any(before[key] != after[key] for key in identity) or before["inputs"]["targets"]["sha256"] != after["inputs"]["targets"]["sha256"]:
        raise EvaluationError("reference and candidate differ in target hash, dataset/split, objective, outputs/classes, weights, evaluator, or settings")
    results, regressions = {}, []
    for name in before["metrics"]:
        old, new = before["metrics"][name], after["metrics"][name]
        if old["direction"] != new["direction"] or old["unit"] != new["unit"]:
            raise EvaluationError("metric direction or unit mismatch: " + name)
        if old["value"] is None or new["value"] is None:
            if old["value"] is not None or new["value"] is not None:
                raise EvaluationError("metric availability mismatch: " + name)
            reason = ("at least one output has a binary class with zero total weight"
                      if before["objective"] == "multilabel" and name == "auc"
                      else "a binary class has zero total weight")
            results[name] = {"reference": None, "candidate": None, "direction": old["direction"],
                             "unit": old["unit"], "status": "not_applicable", "reason": reason}
            continue
        # Binary64's finite exponent range fits comfortably in this precision,
        # including a subtraction spanning its largest and smallest values.
        # Decimal's default precision would silently round some comparisons.
        with localcontext() as context:
            context.prec = 2048
            change = Decimal(str(new["value"])) - Decimal(str(old["value"]))
            deterioration = change if old["direction"] == "minimize" else -change
            regression = deterioration > Decimal(str(allowance))
        if regression:
            regressions.append(name)
        results[name] = {"reference": old["value"], "candidate": new["value"], "direction": old["direction"],
                         "unit": old["unit"], "candidate_minus_reference": float(change),
                         "candidate_minus_reference_decimal": str(change),
                         "deterioration": float(deterioration), "allowed_deterioration": allowance,
                         "deterioration_decimal": str(deterioration),
                         "status": "regression" if regression else "pass"}
    return {"schema_version": SCHEMA_VERSION, "kind": "prediction_evaluation_comparison", "evaluator_version": EVALUATOR_VERSION,
            "status": "regression" if regressions else "pass", "regressions": regressions,
            "dataset_id": before["dataset_id"], "split_id": before["split_id"], "objective": before["objective"],
            "outputs": before["outputs"], "classes": before["classes"], "sample_count": before["sample_count"],
            "targets_sha256": before["inputs"]["targets"]["sha256"], "max_loss_increase": allowance,
            "allowance_interpretation": "Absolute deterioration allowance in each metric's native units; equality passes. Recorded decimal values are compared without an additional tolerance.",
            "inputs": {"reference": {"path": str(reference), "sha256": digest(before_bytes)},
                       "candidate": {"path": str(candidate), "sha256": digest(after_bytes)}},
            "verification": verification, "metrics": results,
            "limitations": LIMITATIONS + ["A pass means only that all applicable metrics meet the declared allowance on this split; source CSV hashes and metrics were rechecked."]}


def write_new(path: Path, report: dict) -> None:
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluation = subparsers.add_parser("evaluate", help="score saved predictions on the declared split",
                                      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    evaluation.add_argument("--objective", choices=OBJECTIVES, required=True)
    evaluation.add_argument("--targets", type=Path, required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True, help="new JSON file; existing files are rejected")
    evaluation.add_argument("--dataset-id", required=True)
    evaluation.add_argument("--split-id", required=True)
    evaluation.add_argument("--classes", type=int, help="required for multiclass; binary/multilabel permit only 2 classes per output")
    evaluation.add_argument("--probability-sum-tolerance", type=float, default=DEFAULT_PROBABILITY_SUM_TOLERANCE,
                            help="multiclass absolute sum tolerance; accepted rows are normalized (default: 1e-6)")
    comparison = subparsers.add_parser("compare", help="check every metric against a saved reference")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, help="new JSON file; otherwise print JSON to stdout")
    comparison.add_argument("--max-loss-increase", type=float, default=0.0,
                            help="absolute deterioration allowed for every metric in its own units; equality passes (default: 0)")
    options = parser.parse_args(argv)
    if options.command == "evaluate":
        report = evaluate(options.targets, options.predictions, options.objective, options.dataset_id, options.split_id,
                          options.classes, options.probability_sum_tolerance)
        write_new(options.output, report)
        return 0
    report = compare(options.reference, options.candidate, options.max_loss_increase)
    if options.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        write_new(options.output, report)
    return 1 if report["status"] == "regression" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, OverflowError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
