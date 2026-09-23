#!/usr/bin/env python3
"""CPU-only, unchanged-evaluator audit of the 12 default-context pairs.

Exit 0: all strict gates pass; 1: metric regressions retained; 2: invalid evidence.
This audit does not reinterpret a zero-allowance failure as a pass.
"""
from pathlib import Path
import importlib.util
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPERS = ROOT / "results/booster-resident-20260922/audit_optimized_quality.py"
spec = importlib.util.spec_from_file_location("retained_context_helpers", HELPERS)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.OUT = OUT
a = old.a
FROZEN = ROOT / "results/booster-reuse-20260922/final-provenance"
SEED = 20260922605
SHAPES = {
    "scalar": ("regression", 65536, 8192, 32, 1, 10, 5, 64),
    "129": ("regression", 4096, 256, 16, 129, 3, 2, 32),
    "1024": ("binary", 4096, 256, 16, 1024, 2, 2, 16),
    "4096": ("binary", 1024, 64, 16, 4096, 1, 2, 16),
    "multiclass17": ("multiclass", 4096, 512, 16, 17, 3, 2, 32),
    "fallback33": ("regression", 2048, 128, 40, 33, 2, 2, 256),
}
POLICIES = {
    "old": dict(root_histogram="per-tree", split_policy="block256", root_counts="per-output", split_batch="per-tree"),
    "candidate": dict(root_histogram="batched", split_policy="warp32", root_counts="reuse-global", split_batch="root"),
}


def protocol(case, mode, name, meta, capture, model, executable_sha):
    objective, rows, test_rows, features, outputs, rounds, depth, bins = SHAPES[case]
    expected = dict(objective=objective, rows=rows, test_rows=test_rows, features=features, outputs=outputs,
                    rounds=rounds, max_depth=depth, max_bins=bins, seed=SEED, test_seed=SEED ^ 0x9E3779B97F4A7C15,
                    histogram="auto", tree_execution="stream", output_tile_size=32,
                    tree_export_batch_requested=0, tree_export_batch_effective=0, quantize_policy="radix8",
                    max_device_bytes=4 << 30, max_histogram_bytes=512 << 20, learning_rate=.1,
                    l2=1, min_leaf_rows=10, min_child_hessian=1e-8, min_gain=0, max_leaf_value=0,
                    instrumentation="off", **POLICIES[mode])
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"{name}: unexpected {key}: {meta[key]} != {value}")
    if capture["executable_sha256"] != executable_sha:
        raise ValueError("capture does not match the frozen previous executable")
    if Path(capture["command"][0]).resolve() != ROOT / "build/booster-reuse/ghb_bench":
        raise ValueError("unexpected benchmark executable")
    args = capture["command"][1:]
    if len(args) % 2 or len(args[::2]) != len(set(args[::2])):
        raise ValueError("odd or repeated captured command flag")
    flags = dict(zip(args[::2], args[1::2]))
    expected_flags = {"--rows": str(rows), "--test-rows": str(test_rows), "--features": str(features),
                      "--rounds": str(rounds), "--depth": str(depth), "--bins": str(bins),
                      "--seed": str(SEED), "--instrumentation": "off", "--output-dir": str(OUT / name)}
    expected_flags.update({"--" + key.replace("_", "-"): value for key, value in POLICIES[mode].items()})
    if case != "scalar":
        expected_flags["--classes" if objective == "multiclass" else "--outputs"] = str(outputs)
    if objective != "regression":
        expected_flags["--objective"] = objective
    if flags != expected_flags:
        raise ValueError("captured flags do not match the declared shape and policies")
    if (model["objective"], model["outputs"], len(model["features"]), len(model["trees"])) != (
            {"regression": 0, "binary": 1, "multiclass": 2}[objective], outputs, features, outputs * rounds):
        raise ValueError("serialized model does not match declared objective/dimensions")


def main():
    destination = OUT / "context-quality"
    destination.mkdir(exist_ok=False)
    marker = destination / ".incomplete"
    marker.touch(exist_ok=False)
    frozen_exe = FROZEN / "build/booster-reuse/ghb_bench"
    frozen_evaluator = FROZEN / "training/tools/evaluate.py"
    executable_sha = a.sha(frozen_exe)
    if a.sha(a.EVALUATOR) != a.sha(frozen_evaluator):
        raise ValueError("the evaluator changed since the preceding frozen audit")
    evaluator = a.load_evaluator()
    source_paths = (Path(__file__), OUT / "run_context_comparison.py", HELPERS, old.HELPERS,
                    a.EVALUATOR, frozen_evaluator, frozen_exe)
    summary = dict(started_unix_seconds=time.time(), allowance=0.0, executable_sha256=executable_sha,
                   cases=[], comparisons=[], errors=[], artifacts=[], sources=[a.artifact(p) for p in source_paths],
                   limitations=["Held-out synthetic CSV comparisons only; no all-encoded-domain analysis in this audit.",
                                "User acceptance of prior floating-point differences does not change strict gate statuses."])
    metadata, captures, reports, models = {}, {}, {}, {}
    for case in SHAPES:
        for suffix in ("a", "b"):
            for mode in ("old", "candidate"):
                name = f"context-{case}-{mode}-{suffix}"
                row = dict(name=name)
                summary["cases"].append(row)
                try:
                    meta, capture, artifacts = old.validate_capture(name)
                    metadata[name], captures[name] = meta, capture
                    summary["artifacts"].extend(artifacts)
                    models[name] = a.read_model(OUT / name / "model.ghb")
                    protocol(case, mode, name, meta, capture, models[name], executable_sha)
                    objective, dataset, split = a.identity(meta)
                    report = evaluator.evaluate(OUT / name / "targets.csv", OUT / name / "predictions.csv",
                                                objective, dataset, split, meta["outputs"] if objective == "multiclass" else None)
                    if (report["sample_count"], report["outputs"]) != (meta["test_rows"], meta["outputs"]):
                        raise ValueError("CSV sample/output dimensions differ from declared shape")
                    reports[name] = destination / (name + ".json")
                    a.write_json(reports[name], report)
                    row.update(status="valid", targets_sha256=a.sha(OUT / name / "targets.csv"), metrics=len(report["metrics"]))
                except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                    row.update(status="invalid_evidence", error=str(error))
                    summary["errors"].append(dict(case=name, error=str(error)))
                print(name, row["status"], flush=True)
    maximum = {}
    for case in SHAPES:
        for suffix in ("a", "b"):
            reference, candidate = (f"context-{case}-{mode}-{suffix}" for mode in ("old", "candidate"))
            row = dict(reference=reference, candidate=candidate)
            summary["comparisons"].append(row)
            try:
                a.matching(metadata[reference], metadata[candidate])
                if captures[reference]["executable_sha256"] != captures[candidate]["executable_sha256"]:
                    raise ValueError("not a same-binary pair")
                targets_sha = a.sha(OUT / reference / "targets.csv")
                if targets_sha != a.sha(OUT / candidate / "targets.csv"):
                    raise ValueError("paired target/weight CSV bytes differ")
                if old.feature_bytes(models[reference]) != old.feature_bytes(models[candidate]):
                    raise ValueError("paired quantization metadata differs")
                comparison = evaluator.compare(reports[reference], reports[candidate], 0.0)
                row.update(a.compact_comparison(comparison), targets_sha256=targets_sha)
                row["predictions"] = a.prediction_changes(OUT / reference / "predictions.csv", OUT / candidate / "predictions.csv",
                                                           OUT / reference / "targets.csv", a.identity(metadata[candidate])[0])
                for metric in comparison["metrics"].values():
                    if metric["status"] == "not_applicable":
                        continue
                    unit = metric["unit"]
                    maximum[unit] = max(maximum.get(unit, 0.0), metric["deterioration"])
                a.write_json(destination / (candidate + "-vs-old.json"), dict(comparison=comparison, **row))
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                row.update(status="invalid_evidence", error=str(error))
                summary["errors"].append(dict(candidate=candidate, error=str(error)))
            print(candidate, row["status"], flush=True)
    for artifact in summary["sources"] + summary["artifacts"]:
        try:
            if a.sha(artifact["path"]) != artifact["sha256"]:
                summary["errors"].append(dict(artifact_changed=artifact["path"]))
        except OSError as error:
            summary["errors"].append(dict(artifact_missing=artifact["path"], error=str(error)))
    summary["maximum_positive_deterioration_by_unit"] = maximum
    summary["status"] = "invalid_evidence" if summary["errors"] else "regression" if any(
        row["status"] == "regression" for row in summary["comparisons"]) else "pass"
    summary["finished_unix_seconds"] = time.time()
    a.write_json(destination / "summary.json", summary)
    lines = ["# Default-context quality audit", "", f"Status: **{summary['status']}**; allowance remains zero.", "",
             "| Candidate | Strict gate | Regressed metrics | Maximum prediction difference | Decision changes |",
             "|---|---|---:|---:|---:|"]
    for row in summary["comparisons"]:
        if "predictions" in row:
            p = row["predictions"]
            decisions = "not applicable" if p["decision_changes"] is None else str(p["decision_changes"])
            lines.append(f"| {row['candidate']} | {row['status']} | {len(row['regressions'])} | {p['max_abs_difference']:.17g} | {decisions} |")
        else:
            lines.append(f"| {row['candidate']} | invalid evidence | — | — | — |")
    lines += ["", "Maximum positive deterioration by metric unit:", ""]
    lines += [f"- {unit}: {value:.17g}" for unit, value in sorted(maximum.items())]
    lines += ["", "All 24 captures are checked against the same frozen executable, exact declared shapes/seeds/policies, "
              "serialized model dimensions, and the unchanged evaluator. Paired target hashes and feature metadata must match. "
              "These are held-out synthetic results; prior strict failures remain unchanged.", ""]
    (destination / "summary.md").write_text("\n".join(lines))
    marker.unlink()
    return 2 if summary["errors"] else int(summary["status"] == "regression")


if __name__ == "__main__":
    raise SystemExit(main())
