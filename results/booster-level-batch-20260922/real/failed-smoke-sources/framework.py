#!/usr/bin/env python3
"""One serial GPU reference-framework job; root agent alone invokes this script."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from fixtures import load, sha256

def elapsed(start):
    return 1000 * (time.perf_counter() - start)

def run(a, out):
    start = time.perf_counter()
    x, y, info = load(a.train)
    xe, ye, evaluation_info = load(a.evaluation)
    if any(info[k] != evaluation_info[k] for k in ("features", "targets", "objective", "classes")):
        raise ValueError("train/evaluation contracts differ")
    load_ms = elapsed(start)
    # Import and CUDA context initialization are outside synchronized fit timing,
    # matching ghb_real_bench. No training warmup or test-driven selection.
    import cupy as cp
    if a.framework == "xgboost":
        import xgboost as framework
    elif a.framework == "lightgbm":
        import lightgbm as framework
    else:
        import catboost as framework
    context_start = time.perf_counter()
    cp.cuda.runtime.free(0)
    sync = cp.cuda.runtime.deviceSynchronize
    sync()
    context_ms = elapsed(context_start)
    objective, outputs = info["objective"], info["targets"]
    if objective == 2:
        outputs = info["classes"]
    fit_y = y[:, 0] if info["targets"] == 1 else y
    constant_labels = []
    model_count = 1
    fit_ms = 0
    prepare_ms = 0
    training_start = time.perf_counter()
    if a.framework == "xgboost":
        parameters = dict(tree_method="hist", device="cuda:0", objective=["reg:squarederror", "binary:logistic", "multi:softprob"][objective],
                          max_depth=a.depth, max_bin=a.bins, learning_rate=a.learning_rate, reg_lambda=a.l2,
                          min_child_weight=1e-8, subsample=1, colsample_bytree=1, seed=a.seed,
                          multi_strategy=a.multi_strategy, nthread=a.threads, verbosity=1, disable_default_eval_metric=1)
        if objective == 2:
            parameters["num_class"] = info["classes"]
        dtrain = framework.QuantileDMatrix(cp.asarray(x), label=cp.asarray(fit_y), max_bin=a.bins)
        sync()
        prepare_ms = elapsed(training_start)
        fit_start = time.perf_counter()
        model = framework.train(parameters, dtrain, num_boost_round=a.rounds)
        sync()
        fit_ms = elapsed(fit_start)
        training_wall_ms = elapsed(training_start)
        resolved = json.loads(model.save_config())
        if not resolved["learner"]["generic_parameter"]["device"].startswith("cuda"):
            raise RuntimeError("XGBoost silently selected a non-CUDA device")
        prediction_start = time.perf_counter()
        # GPU array input keeps prediction on the requested device. Includes the
        # evaluation upload and returned host prediction copy, like custom API.
        predictions = cp.asnumpy(model.inplace_predict(cp.asarray(xe)))
        sync()
        prediction_ms = elapsed(prediction_start)
        serialization_start = time.perf_counter()
        model.save_model(out / "model.ubj")
        model_count = a.rounds * (outputs if a.multi_strategy == "one_output_per_tree" else 1)
        architecture = a.multi_strategy
        prediction_backend = "cuda"
    elif a.framework == "lightgbm":
        parameters = dict(device_type="cuda", gpu_device_id=0, num_gpu=1, objective=["regression", "binary", "multiclass"][objective],
                          max_depth=a.depth, num_leaves=2 ** a.depth, max_bin=a.bins, learning_rate=a.learning_rate,
                          lambda_l2=a.l2, min_data_in_leaf=1, min_sum_hessian_in_leaf=1e-8, min_data_in_bin=1,
                          feature_pre_filter=False, bagging_fraction=1, feature_fraction=1, bagging_freq=0,
                          seed=a.seed, data_random_seed=a.seed, num_threads=a.threads, verbosity=1,
                          metric="None", boost_from_average=True)
        if objective == 2:
            parameters["num_class"] = info["classes"]
        # One common bin construction for all binary-relevance labels; set_label
        # changes only training targets. The complete label loop remains timed.
        data = framework.Dataset(x, label=y[:, 0], params=parameters, free_raw_data=False)
        data.construct()
        sync()
        prepare_ms = elapsed(training_start)
        fit_start = time.perf_counter()
        models = []
        independent = info["targets"] if objective != 2 else 1
        for output in range(independent):
            target = y[:, output]
            if objective == 1 and (np.all(target == 0) or np.all(target == 1)):
                constant_labels.append(output)
                models.append(float(target[0]))
                continue
            data.set_label(target)
            models.append(framework.train(parameters, data, num_boost_round=a.rounds))
        sync()
        fit_ms = elapsed(fit_start)
        training_wall_ms = elapsed(training_start)
        resolved = [m.params if not isinstance(m, float) else {"constant": m} for m in models]
        prediction_start = time.perf_counter()
        if objective == 2:
            predictions = models[0].predict(xe, num_threads=a.threads)
        else:
            predictions = np.column_stack([np.full(len(xe), m) if isinstance(m, float) else m.predict(xe, num_threads=a.threads) for m in models])
        sync()
        prediction_ms = elapsed(prediction_start)
        serialization_start = time.perf_counter()
        for index, model in enumerate(models):
            if isinstance(model, float):
                (out / f"model-{index}.constant.json").write_text(json.dumps({"constant": model}) + "\n")
            else:
                model.save_model(out / f"model-{index}.txt")
        model_count = sum(m.num_trees() for m in models if not isinstance(m, float))
        architecture = "native-scalar-or-multiclass" if independent == 1 else "sequential-binary-relevance-reused-dataset"
        prediction_backend = "cpu-public-lightgbm-api"
    else:
        loss = "RMSE" if objective == 0 and outputs == 1 else "MultiRMSE" if objective == 0 else "MultiClass" if objective == 2 else "Logloss" if outputs == 1 else "MultiLogloss"
        parameters = dict(task_type="GPU", devices="0", loss_function=loss, iterations=a.rounds, depth=a.depth,
                          border_count=a.bins, learning_rate=a.learning_rate, l2_leaf_reg=a.l2,
                          boosting_type="Plain", grow_policy="SymmetricTree", bootstrap_type="No", random_strength=0,
                          leaf_estimation_iterations=1, score_function="NewtonL2", random_seed=a.seed,
                          thread_count=a.threads, allow_writing_files=False, verbose=False, allow_const_label=True)
        if objective == 2:
            parameters["classes_count"] = info["classes"]
        if outputs == 1:
            parameters["boost_from_average"] = True
        pool = framework.Pool(x, fit_y, thread_count=a.threads)
        sync()
        prepare_ms = elapsed(training_start)
        fit_start = time.perf_counter()
        model = framework.CatBoost(parameters)
        model.fit(pool)
        sync()
        fit_ms = elapsed(fit_start)
        training_wall_ms = elapsed(training_start)
        resolved = model.get_all_params()
        if resolved.get("task_type") != "GPU":
            raise RuntimeError("CatBoost did not retain the requested GPU backend")
        prediction_start = time.perf_counter()
        predictions = np.asarray(model.predict(xe, prediction_type="RawFormulaVal" if objective == 0 else "Probability", task_type="GPU", thread_count=a.threads))
        if objective == 1 and outputs == 1:
            predictions = predictions[:, 1]
        sync()
        prediction_ms = elapsed(prediction_start)
        serialization_start = time.perf_counter()
        model.save_model(out / "model.cbm")
        model_count = model.tree_count_
        architecture = "native-symmetric-vector-leaf" if outputs > 1 else "native-symmetric-scalar-leaf"
        prediction_backend = "cuda"
    predictions = np.asarray(predictions, dtype="<f8", order="C").reshape(len(xe), outputs)
    if not np.isfinite(predictions).all():
        raise RuntimeError("nonfinite framework predictions")
    predictions.tofile(out / "predictions.f64")
    serialization_ms = elapsed(serialization_start)
    model_bytes = sum(p.stat().st_size for p in out.glob("model*"))
    device = cp.cuda.runtime.getDeviceProperties(0)
    result = {"implementation": a.framework, "version": framework.__version__, "backend": "cuda", "prediction_backend": prediction_backend,
              "architecture": architecture, "device": device["name"].decode(), "objective": objective, "train_rows": len(x),
              "evaluation_rows": len(xe), "features": x.shape[1], "outputs": outputs,
              "parameters": vars(a), "resolved_parameters": resolved, "constant_labels": constant_labels,
              "timing_ms": {"load": load_ms, "context": context_ms, "preparation": prepare_ms, "fit": fit_ms,
                            "training_wall": training_wall_ms, "prediction_wall": prediction_ms, "serialization": serialization_ms},
              "memory": {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                         "device_peak_bytes": None, "device_peak_note": "not measured; do not compare allocator usage to custom owned payload"},
              "trees": model_count, "model_bytes": model_bytes, "fixture_sha256": sha256(a.train),
              "evaluation_fixture_sha256": sha256(a.evaluation), "predictions_sha256": sha256(out / "predictions.f64"),
              "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(), "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
              "environment": {p: importlib.metadata.version(p) for p in ("numpy", "cupy-cuda13x", "xgboost", "lightgbm", "catboost")}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_wall_ms": training_wall_ms, "prediction_wall_ms": prediction_ms}))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["xgboost", "lightgbm", "catboost"], required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=.1)
    p.add_argument("--l2", type=float, default=1)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--multi-strategy", choices=["one_output_per_tree", "multi_output_tree"], default="one_output_per_tree")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(a, out)
    except Exception:
        (out / "failure.json").write_text(json.dumps({"parameters": vars(a), "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise

if __name__ == "__main__":
    main()
