#!/usr/bin/env python3
"""Common CPU held-out metric reference, outside production/timed computation."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn import metrics as skm
from fixtures import load, sha256

EPSILON = 1e-15

def metrics(y, prediction, objective):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("nonfinite predictions")
    if objective == 0:
        if p.shape != y.shape:
            raise ValueError("regression shape mismatch")
        mse = np.mean(np.square(p - y), dtype=np.float64)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(p - y))),
                "r2": float(skm.r2_score(y, p)), "selection_metric": "mse", "selection_value": float(mse)}
    if (p < 0).any() or (p > 1).any():
        raise ValueError("classification predictions outside [0,1]")
    if objective == 2:
        target = y[:, 0].astype(np.int64)
        if p.ndim != 2 or p.shape[0] != len(y) or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("invalid multiclass prediction normalization")
        chosen = np.argmax(p, axis=1)
        loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), target], EPSILON, 1))))
        onehot = np.eye(p.shape[1], dtype=np.float64)[target]
        return {"log_loss": loss, "accuracy": float(np.mean(chosen == target)),
                "macro_f1": float(skm.f1_score(target, chosen, labels=np.arange(p.shape[1]), average="macro", zero_division=0)),
                "brier": float(np.mean(np.sum(np.square(p - onehot), axis=1))),
                "selection_metric": "log_loss", "selection_value": loss, "log_clip_epsilon": EPSILON}
    if p.shape != y.shape:
        raise ValueError("independent binary prediction shape mismatch")
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    predicted = p >= .5
    result = {"log_loss": loss, "brier": float(np.mean(np.square(p - y))),
              "hamming_loss": float(np.mean(predicted != y)), "exact_match_accuracy": float(np.mean(np.all(predicted == y, axis=1))),
              "selection_metric": "log_loss", "selection_value": loss, "log_clip_epsilon": EPSILON}
    if y.shape[1] == 1:
        result.update(accuracy=float(np.mean(predicted == y)), f1=float(skm.f1_score(y[:, 0], predicted[:, 0], zero_division=0)),
                      roc_auc=float(skm.roc_auc_score(y[:, 0], p[:, 0])), average_precision=float(skm.average_precision_score(y[:, 0], p[:, 0])))
    else:
        present = y.sum(axis=0) > 0
        varying = present & (y.sum(axis=0) < len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result.update(micro_f1=float(skm.f1_score(y, predicted, average="micro", zero_division=0)),
                          macro_f1=float(skm.f1_score(y, predicted, average="macro", zero_division=0)),
                          micro_ap=float(skm.average_precision_score(y.ravel(), p.ravel())),
                          macro_ap=float(skm.average_precision_score(y[:, present], p[:, present], average="macro")),
                          macro_ap_labels=int(present.sum()),
                          macro_auc=float(skm.roc_auc_score(y[:, varying], p[:, varying], average="macro")),
                          macro_auc_labels=int(varying.sum()))
        # Stable label-index tie break is shared by every implementation.
        order = np.argsort(-p, axis=1, kind="stable")
        for k in (1, 3, 5):
            if k <= y.shape[1]:
                result[f"precision_at_{k}"] = float(np.mean(np.take_along_axis(y, order[:, :k], axis=1)))
    return result

def evaluate(train_path, evaluation_path, predictions_path):
    _, train_y, train_header = load(train_path)
    _, y, header = load(evaluation_path)
    if any(train_header[k] != header[k] for k in ("features", "targets", "objective", "classes")):
        raise ValueError("train/evaluation contract mismatch")
    outputs = header["classes"] if header["objective"] == 2 else header["targets"]
    predictions = np.fromfile(predictions_path, dtype="<f8").reshape(header["rows"], outputs)
    if header["objective"] == 2:
        base = np.bincount(train_y[:, 0].astype(int), minlength=outputs) / len(train_y)
    else:
        base = np.mean(train_y, axis=0, dtype=np.float64)
    return {"reference": "CPU float64 common metric implementation", "fixture_sha256": sha256(evaluation_path),
            "training_fixture_sha256": sha256(train_path), "predictions_sha256": sha256(predictions_path),
            "metrics": metrics(y, predictions, header["objective"]),
            "training_mean_baseline": metrics(y, np.broadcast_to(base, predictions.shape), header["objective"])}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = evaluate(a.train, a.evaluation, a.predictions)
    with Path(a.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
