#!/usr/bin/env python3
"""CPU-only post-audit protocol/hash/status verification; does not evaluate CSVs."""
from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location("reuse_quality_review", OUT / "audit_quality.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
a = audit.a


def require(condition, message):
    if not condition:
        raise ValueError(message)


def planned():
    result = {}
    for case in ("scalar", "129", "1024", "4096"):
        for suffix in ("a", "b"):
            for mode in ("base", "counts", "split", "both", "deep-shared"):
                result[f"{case}-{mode}-{suffix}"] = (case, mode, None)
        for suffix in (("a", "b") if case == "scalar" else ("c", "d")):
            mode = "original" if case == "scalar" else "base"
            result[f"{case}-{mode}-{suffix}"] = (case, mode, None)
    for case in ("129", "4096"):
        for suffix in ("a", "b"):
            result[f"{case}-counts-shared-{suffix}"] = (case, "counts-shared", None)
        for mode in ("base", "both", "deep-shared"):
            result[f"confirm-{case}-{mode}"] = (case, mode, 20260922604)
    for mode in ("base", "both", "deep-shared"):
        result[f"multiclass17-{mode}"] = ("multiclass17", mode, None)
    require(len(result) == 61, "review plan must contain 61 cases")
    return result


def expected(case, mode, seed):
    meta = dict(schema_version=1, kind="ghb.training", generator_version=2,
                objective="regression", rows=4096, test_rows=256, features=16,
                outputs=129, seed=2026092201, rounds=3, max_depth=2, max_bins=32,
                output_tile_size=16, max_device_bytes=4 << 30,
                max_histogram_bytes=512 << 20, learning_rate=.1, l2=1,
                min_leaf_rows=10, min_child_hessian=1e-8, min_gain=0,
                max_leaf_value=0, tree_execution="graph", instrumentation="off",
                quantize_policy="radix8", tree_export_batch_requested=16,
                tree_export_batch_effective=16, histogram="global",
                root_histogram="batched", split_policy="warp32",
                root_counts="per-output", split_batch="per-tree")
    if case == "scalar":
        meta.update(rows=65536, test_rows=8192, features=32, outputs=1,
                    seed=20260922601, rounds=10, max_depth=5, max_bins=64,
                    output_tile_size=32, tree_export_batch_effective=1, histogram="shared")
    elif case == "1024":
        meta.update(objective="binary", outputs=1024, rounds=2, max_bins=16)
    elif case == "4096":
        meta.update(objective="binary", outputs=4096, rows=1024,
                    test_rows=64, rounds=1, max_bins=16)
    elif case == "multiclass17":
        meta.update(objective="multiclass", outputs=17, test_rows=512,
                    output_tile_size=5, tree_export_batch_requested=5,
                    tree_export_batch_effective=5)
    elif case != "129":
        raise ValueError("unknown case " + case)
    if seed is not None:
        meta["seed"] = seed
    if mode in ("counts", "both", "deep-shared"):
        meta["root_counts"] = "reuse-global"
    if mode == "counts-shared":
        meta["root_counts"] = "reuse-shared"
    if mode in ("split", "both", "deep-shared"):
        meta["split_batch"] = "root"
    if mode == "deep-shared":
        meta["histogram"] = "shared"
    if mode == "original":
        meta.update(root_histogram="per-tree", split_policy="block256")
    meta["test_seed"] = meta["seed"] ^ 0x9E3779B97F4A7C15
    meta["trees"] = meta["outputs"] * meta["rounds"]
    meta["root_histogram_batch_size"] = (0 if mode == "original" else
                                         min(meta["outputs"], meta["output_tile_size"]))
    return meta


FLAG_FIELDS = {
    "--objective": "objective", "--rows": "rows", "--test-rows": "test_rows",
    "--features": "features", "--outputs": "outputs", "--classes": "outputs",
    "--rounds": "rounds", "--depth": "max_depth", "--bins": "max_bins",
    "--seed": "seed", "--output-tile": "output_tile_size",
    "--max-device-bytes": "max_device_bytes", "--histogram": "histogram",
    "--tree-execution": "tree_execution", "--tree-export-batch": "tree_export_batch_requested",
    "--instrumentation": "instrumentation", "--quantize-policy": "quantize_policy",
    "--root-histogram": "root_histogram", "--split-policy": "split_policy",
    "--root-counts": "root_counts", "--split-batch": "split_batch",
}


def main():
    names = planned()
    planned_names, pairs = audit.plan()
    require(set(planned_names) == set(names) and len(pairs) == 62,
            "campaign and independent 61-case/62-pair review plans differ")
    if sys.argv[1:] == ["--plan-only"]:
        for case, mode, seed in names.values():
            expected(case, mode, seed)
        print("Plan verified: 61 cases, 62 paired gates; no capture or quality work run.")
        return 0
    require(not sys.argv[1:], "usage: verify_metadata.py [--plan-only]")
    quality = OUT / "quality"
    require(not (quality / ".incomplete").exists(), "quality audit is incomplete")
    summary = a.read_json(quality / "summary.json")
    require(summary["allowance"] == 0 and not summary["errors"], "audit allowance/errors invalid")
    require({r["name"] for r in summary["cases"]} == set(names) and len(summary["cases"]) == 61,
            "quality case coverage mismatch")
    require({(r["reference"], r["candidate"]) for r in summary["comparisons"]} == set(pairs)
            and len(summary["comparisons"]) == 62, "quality pair coverage mismatch")
    hashes, artifacts = {}, {}

    def checked_hash(path):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = a.sha(path)
        return hashes[path]

    def verify_artifact(item):
        path = str(Path(item["path"]).resolve())
        require(checked_hash(path) == item["sha256"], "artifact hash changed: " + path)
        if "bytes" in item:
            require(Path(path).stat().st_size == item["bytes"], "artifact size changed: " + path)
        artifacts[path] = dict(path=path, sha256=item["sha256"])

    for item in summary["sources"] + summary["artifacts"]:
        verify_artifact(item)
    evaluator = a.load_evaluator()
    evaluator_hash = checked_hash(a.EVALUATOR)
    binary_hashes, reports = set(), {}
    for name, shape in names.items():
        meta, capture, retained = audit.old.validate_capture(name)
        for key, value in expected(*shape).items():
            require(meta[key] == value, f"{name}: unexpected {key}: {meta[key]!r} != {value!r}")
        command = capture["command"]
        require(len(command) % 2 == 1, name + ": malformed command flag/value pairs")
        flags = dict(zip(command[1::2], command[2::2]))
        require(Path(flags.pop("--output-dir")).resolve() == OUT / name, name + ": output directory mismatch")
        for flag, value in flags.items():
            require(flag in FLAG_FIELDS, name + ": unexpected command flag " + flag)
            require(value == str(meta[FLAG_FIELDS[flag]]), name + ": command/result mismatch for " + flag)
        binary_hashes.add(capture["executable_sha256"])
        for item in retained:
            verify_artifact(item)
        model = a.read_model(OUT / name / "model.ghb")
        require(model["objective"] == {"regression": 0, "binary": 1, "multiclass": 2}[meta["objective"]],
                name + ": serialized model objective mismatch")
        require((model["outputs"], len(model["features"]), len(model["trees"])) ==
                (meta["outputs"], meta["features"], meta["trees"]), name + ": serialized model dimensions mismatch")
        objective, dataset, split = a.identity(meta)
        for kind, csv_name in (("model", "predictions.csv"), ("base", "baseline.csv")):
            path = quality / (name + "-" + kind + ".json")
            report = a.read_json(path)
            require((report["objective"], report["sample_count"], report["outputs"], report["dataset_id"], report["split_id"])
                    == (objective, meta["test_rows"], meta["outputs"], dataset, split), name + ": evaluation identity mismatch")
            require(report["evaluator_source_sha256"] == evaluator_hash, "evaluator source hash mismatch")
            require(set(report["metrics"]) == set(evaluator.metric_specs(objective, meta["outputs"])),
                    name + ": metric coverage mismatch")
            for label, filename in (("targets", "targets.csv"), ("predictions", csv_name)):
                item = report["inputs"][label]
                require(Path(item["path"]).resolve() == OUT / name / filename, name + ": evaluation source mismatch")
                verify_artifact(item)
            reports[str(path)] = report
    require(len(binary_hashes) == 1, "61-case campaign used more than one executable hash")

    def verify_comparison(comparison, row, name):
        require(comparison["max_loss_increase"] == 0, name + ": nonzero quality allowance")
        before = reports[str(Path(comparison["inputs"]["reference"]["path"]).resolve())]
        after = reports[str(Path(comparison["inputs"]["candidate"]["path"]).resolve())]
        for item in comparison["inputs"].values():
            verify_artifact(item)
        require(set(comparison["metrics"]) == set(before["metrics"]) == set(after["metrics"]),
                name + ": comparison omitted metrics")
        regressions = []
        for metric, entry in comparison["metrics"].items():
            old, new = before["metrics"][metric], after["metrics"][metric]
            require((entry["reference"], entry["candidate"], entry["direction"], entry["unit"]) ==
                    (old["value"], new["value"], old["direction"], old["unit"]), name + ": metric values differ")
            if old["value"] is None or new["value"] is None:
                require(old["value"] is None and new["value"] is None and entry["status"] == "not_applicable",
                        name + ": inconsistent unavailable metric")
                continue
            with localcontext() as context:
                context.prec = 2048
                delta = Decimal(str(new["value"])) - Decimal(str(old["value"]))
                deterioration = delta if old["direction"] == "minimize" else -delta
            status = "regression" if deterioration > 0 else "pass"
            require(entry["allowed_deterioration"] == 0 and entry["status"] == status
                    and Decimal(entry["deterioration_decimal"]) == deterioration,
                    name + ": strict metric status differs")
            if status == "regression":
                regressions.append(metric)
        require(comparison["regressions"] == regressions and comparison["status"] ==
                ("regression" if regressions else "pass"), name + ": strict gate status differs")
        compact = a.compact_comparison(comparison)
        require(compact == {key: row[key] for key in compact}, name + ": summary changed comparison outcome")
        for checked in comparison["verification"].values():
            require(checked["metrics_recomputed"] is True and set(checked["artifact_hashes_verified"]) == {"targets", "predictions"},
                    name + ": source metrics were not verified")

    for row in summary["cases"]:
        verify_comparison(a.read_json(quality / (row["name"] + "-model-vs-base.json")), row, row["name"])
    for row in summary["comparisons"]:
        name = row["candidate"] + "-vs-" + row["reference"]
        verify_comparison(a.read_json(quality / (name + ".json"))["comparison"], row, name)
    expected_status = "regression" if any(row["status"] == "regression" for row in summary["cases"] + summary["comparisons"]) else "pass"
    require(summary["status"] == expected_status, "summary conceals strict failures")
    sources = [a.artifact(path) for path in (Path(__file__), OUT / "run_campaign.py", OUT / "audit_quality.py",
                                           ROOT / "training/bench/booster.cpp", a.EVALUATOR)]
    result = dict(status="verified", quality_status=summary["status"], allowance=0,
                  cases=61, paired_gates=62, constant_base_gates=61,
                  executable_sha256=next(iter(binary_hashes)), unique_artifacts_verified=len(artifacts),
                  sources=sources, quality_summary=a.artifact(quality / "summary.json"),
                  limitations="Verifies recorded protocol, artifact identity, coverage and exact zero-allowance arithmetic; the preceding quality audit recomputed the metrics from CSVs.")
    a.write_json(OUT / "metadata-verification.json", result)
    print("Metadata verified: 61 cases, 123 strict gates; retained quality status=" + summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
