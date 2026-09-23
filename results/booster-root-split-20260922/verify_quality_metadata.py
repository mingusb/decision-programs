"""Independent CPU-only metadata verification after the retained quality audit.

This complements, and never edits or reruns, audit_quality.py. Exit 0 means the
metadata checks passed; quality regressions remain in quality_status. Exit 2
means invalid evidence. Existing output is never overwritten.
"""
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPERS = ROOT / "results/booster-resident-20260922/audit_quality.py"
EVALUATOR = ROOT / "training/tools/evaluate.py"
MODES = {"base": ("per-tree", "block256"), "split": ("per-tree", "warp32"),
         "root": ("batched", "block256"), "both": ("batched", "warp32")}
ERRORS = (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def expected_plan():
    cases, pairs = {}, []
    for case in ("scalar", "129", "1024", "4096"):
        for suffix in ("a", "b"):
            for mode in MODES:
                cases[f"{case}-{mode}-{suffix}"] = (case, mode, False)
            pairs.extend((f"{case}-base-{suffix}", f"{case}-{mode}-{suffix}")
                         for mode in ("split", "root", "both"))
    for mode in MODES:
        cases["confirm-129-" + mode] = ("129", mode, True)
    pairs.extend(("confirm-129-base", "confirm-129-" + mode) for mode in ("split", "root", "both"))
    for mode in ("base", "both"):
        cases["multiclass17-" + mode] = ("multiclass17", mode, False)
    pairs.append(("multiclass17-base", "multiclass17-both"))
    require(len(cases) == 38 and len(pairs) == 28, "incorrect independent experiment plan")
    return cases, pairs


def expected_metadata(case, mode, confirmation):
    meta = dict(schema_version=1, kind="ghb.training", generator_version=2,
                tree_execution="graph", quantize_policy="radix8", instrumentation="off",
                max_device_bytes=4 << 30, max_histogram_bytes=512 << 20,
                learning_rate=.1, l2=1, min_leaf_rows=10, min_child_hessian=1e-8,
                min_gain=0, max_leaf_value=0, seed=2026092201)
    if case == "scalar":
        meta.update(objective="regression", rows=65536, test_rows=8192, features=32, outputs=1,
                    rounds=10, max_depth=5, max_bins=64, histogram="shared", output_tile_size=32,
                    seed=20260922601, tree_export_batch_requested=16, tree_export_batch_effective=1)
    elif case == "multiclass17":
        meta.update(objective="multiclass", rows=4096, test_rows=512, features=16, outputs=17,
                    rounds=3, max_depth=2, max_bins=32, histogram="global", output_tile_size=5,
                    tree_export_batch_requested=5, tree_export_batch_effective=5)
    else:
        outputs = int(case)
        meta.update(objective="regression" if outputs == 129 else "binary", outputs=outputs,
                    rows=1024 if outputs == 4096 else 4096, test_rows=64 if outputs == 4096 else 256,
                    features=16, rounds=3 if outputs == 129 else 1 if outputs == 4096 else 2,
                    max_depth=2, max_bins=32 if outputs == 129 else 16, histogram="global",
                    output_tile_size=16, tree_export_batch_requested=16, tree_export_batch_effective=16)
    if confirmation:
        meta["seed"] = 20260922603
    meta["test_seed"] = meta["seed"] ^ 0x9E3779B97F4A7C15
    meta["root_histogram"], meta["split_policy"] = MODES[mode]
    meta["root_histogram_batch_size"] = min(meta["outputs"], meta["output_tile_size"]) if mode in ("root", "both") else 0
    meta["trees"] = meta["rounds"] * meta["outputs"]
    return meta


def command_flags(command):
    require(isinstance(command, list) and command and all(isinstance(x, str) for x in command), "invalid capture command")
    require(len(command) % 2 == 1, "capture command is not option/value pairs")
    values, occurrences = {}, Counter()
    for index in range(1, len(command), 2):
        key, value = command[index:index + 2]
        require(key.startswith("--"), "invalid captured option: " + key)
        values[key] = value  # The benchmark parser uses the last occurrence.
        occurrences[key] += 1
    return values, dict(occurrences)


def check_command(name, capture, meta):
    values, occurrences = command_flags(capture["command"])
    strings = {"--objective": "objective", "--histogram": "histogram", "--tree-execution": "tree_execution",
               "--root-histogram": "root_histogram", "--split-policy": "split_policy",
               "--quantize-policy": "quantize_policy", "--instrumentation": "instrumentation"}
    integers = {"--rows": "rows", "--test-rows": "test_rows", "--features": "features", "--rounds": "rounds",
                "--depth": "max_depth", "--bins": "max_bins", "--output-tile": "output_tile_size",
                "--tree-export-batch": "tree_export_batch_requested", "--max-device-bytes": "max_device_bytes", "--seed": "seed"}
    for key, value in values.items():
        if key in strings:
            require(value == meta[strings[key]], "captured flag disagrees with result: " + key)
        elif key in integers:
            require(value.isdecimal() and int(value) == meta[integers[key]], "captured flag disagrees with result: " + key)
        elif key == "--outputs":
            expected = 1 if meta["objective"] == "multiclass" else meta["outputs"]
            require(value.isdecimal() and int(value) == expected, "captured output count disagrees with result")
        elif key == "--classes":
            require(meta["objective"] == "multiclass" and value.isdecimal() and int(value) == meta["outputs"],
                    "captured class count disagrees with result")
        elif key == "--output-dir":
            require(Path(value).resolve() == OUT / name, "captured output directory differs from case")
        else:
            raise ValueError("unrecognized captured flag: " + key)
    for key in ("--root-histogram", "--split-policy", "--output-dir", "--tree-execution", "--instrumentation"):
        require(key in values, "required explicit experiment flag missing: " + key)
    require(Path(capture["command"][0]).resolve() == ROOT / "build/booster-root-split/ghb_bench", "unexpected captured executable path")
    require(capture["returncode"] == 0 and capture["executable_unchanged"] is True, "failed or changed captured executable")
    return occurrences


def check_comparison(report, objective, outputs, evaluator):
    require(report["max_loss_increase"] == 0 and not isinstance(report["max_loss_increase"], bool), "quality allowance changed")
    require(report["objective"] == objective and report["outputs"] == outputs, "comparison objective/output mismatch")
    specs = evaluator.metric_specs(objective, outputs)
    require(set(report["metrics"]) == set(specs), "comparison omits or adds metrics")
    failures = []
    for name, (direction, unit, _) in specs.items():
        metric = report["metrics"][name]
        require((metric["direction"], metric["unit"]) == (direction, unit), "metric direction/unit changed: " + name)
        if metric["status"] == "not_applicable":
            require(metric["reference"] is None and metric["candidate"] is None, "applicable metric was skipped: " + name)
            require(objective in ("binary", "multilabel") and (name == "auc" or name.startswith("auc_output_")),
                    "metric does not permit not-applicable status: " + name)
            continue
        require(metric["allowed_deterioration"] == 0 and not isinstance(metric["allowed_deterioration"], bool), "per-metric allowance changed: " + name)
        for value in (metric["reference"], metric["candidate"]):
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "invalid metric value: " + name)
        with localcontext() as context:
            context.prec = 2048
            change = Decimal(str(metric["candidate"])) - Decimal(str(metric["reference"]))
            deterioration = change if direction == "minimize" else -change
            expected_status = "regression" if deterioration > 0 else "pass"
            require(Decimal(metric["deterioration_decimal"]) == deterioration, "stored deterioration disagrees with metric values: " + name)
        require(metric["status"] == expected_status, "zero-allowance metric status incorrect: " + name)
        if expected_status == "regression":
            failures.append(name)
    require(set(failures) == set(report["regressions"]) and len(failures) == len(report["regressions"]), "comparison regression list differs from metrics")
    require(report["status"] == ("regression" if failures else "pass"), "comparison status masks metric failure")
    return len(specs), failures


def main():
    destination = OUT / "quality-metadata.json"
    marker = OUT / "quality-metadata.incomplete"
    require(not destination.exists(), "refusing to overwrite " + str(destination))
    marker.open("x").close()
    result = dict(started_unix_seconds=time.time(), status="invalid_evidence", quality_status=None,
                  cases=[], comparisons=[], errors=[], artifacts=[], source_sha256=digest(__file__),
                  scope="Independent metadata, mode, command, model, retained-hash and zero-allowance coverage verification; prior metric computation and quality failures are retained.")
    tracked = {}

    def track(path, expected_sha=None, expected_bytes=None):
        path = Path(path).resolve()
        sha = digest(path)
        size = path.stat().st_size
        require(expected_sha is None or sha == expected_sha, "artifact hash differs: " + str(path))
        require(expected_bytes is None or size == expected_bytes, "artifact size differs: " + str(path))
        record = dict(path=str(path), sha256=sha, bytes=size)
        require(str(path) not in tracked or tracked[str(path)] == record, "artifact changed within verification: " + str(path))
        tracked[str(path)] = record

    def comparison_inputs(report):
        for label in ("reference", "candidate"):
            artifact = report["inputs"][label]
            path = Path(artifact["path"])
            require(path.is_absolute() and path.is_relative_to(OUT / "quality"), "comparison input report is outside this quality audit")
            track(path, artifact["sha256"])

    try:
        track(__file__)
        require(not (OUT / "quality/.incomplete").exists(), "quality audit is still incomplete; run verifier after it finishes")
        require((OUT / "quality/summary.md").is_file(), "quality Markdown output is absent (possible incomplete rendering)")
        track(OUT / "quality/summary.md")
        track(OUT / "quality/summary.json")
        summary = read_json(OUT / "quality/summary.json")
        require(summary["allowance"] == 0 and not isinstance(summary["allowance"], bool), "audit allowance changed")
        require(summary["status"] in ("pass", "regression") and not summary["errors"], "quality audit reports invalid evidence")
        result["quality_status"] = summary["status"]
        cases, pairs = expected_plan()
        require(Counter(row["name"] for row in summary["cases"]) == Counter(cases.keys()), "quality case coverage differs from all 38 planned cases")
        require(Counter((row["reference"], row["candidate"]) for row in summary["comparisons"]) == Counter(pairs), "quality pair coverage differs from all 28 planned comparisons")
        for artifact in summary["sources"] + summary["artifacts"]:
            try:
                require(Path(artifact["path"]).is_absolute(), "audit artifact path is not absolute")
                track(artifact["path"], artifact["sha256"], artifact["bytes"])
            except ERRORS as error:
                result["errors"].append(dict(artifact=artifact.get("path"), error=str(error)))
        source_paths = {Path(item["path"]).resolve() for item in summary["sources"]}
        require(HELPERS in source_paths and EVALUATOR in source_paths and OUT / "audit_quality.py" in source_paths,
                "quality audit omitted verifier dependencies/source provenance")
        helpers = module(HELPERS, "verified_metadata_helpers")
        evaluator = module(EVALUATOR, "verified_metadata_evaluator")
        metadata, captures, statuses = {}, {}, []
        for name, (case, mode, confirmation) in cases.items():
            row = dict(name=name, status="invalid_evidence")
            result["cases"].append(row)
            try:
                require(not (OUT / name / ".incomplete").exists(), "benchmark output incomplete")
                required_artifacts = [OUT / name / filename for filename in ("result.json", "targets.csv", "predictions.csv", "baseline.csv", "model.ghb")]
                required_artifacts += [OUT / (name + suffix) for suffix in ("-capture.json", "-stdout.json", "-stderr.txt")]
                require(all(str(path.resolve()) in tracked for path in required_artifacts), "quality audit omitted required benchmark artifacts")
                meta = read_json(OUT / name / "result.json")
                capture = read_json(OUT / (name + "-capture.json"))
                require(meta == read_json(OUT / (name + "-stdout.json")), "benchmark result differs from captured stdout")
                expected = expected_metadata(case, mode, confirmation)
                for key, value in expected.items():
                    require(meta[key] == value and not isinstance(meta[key], bool), "planned benchmark metadata differs: " + key)
                row["command_option_occurrences"] = check_command(name, capture, meta)
                frozen = OUT / "final-provenance/build/booster-root-split/ghb_bench"
                track(frozen, capture["executable_sha256"])
                model = helpers.read_model(OUT / name / "model.ghb")
                expected_objective = {"regression": 0, "binary": 1, "multiclass": 2}[meta["objective"]]
                require((model["objective"], model["outputs"], len(model["features"]), len(model["trees"])) ==
                        (expected_objective, meta["outputs"], meta["features"], meta["trees"]), "parsed model metadata differs from benchmark")
                require([tree[0] for tree in model["trees"]] == list(range(meta["outputs"])) * meta["rounds"], "model output-tree schedule differs from rounds/outputs")
                bins = sum(len(values) + (2 if kind == 0 else 1) for kind, values in model["features"])
                cache_bytes = 24 * bins * meta["root_histogram_batch_size"]
                require(meta["memory"]["root_histogram_bytes"] == cache_bytes, "reported root cache bytes differ from actual feature/output dimensions")
                require(cache_bytes <= meta["memory"]["histogram_bytes"] <= meta["max_histogram_bytes"], "combined histogram budget inconsistent")
                require(meta["memory"]["owned_device_peak_bytes"] == max(meta["memory"]["device_payload_bytes"], meta["memory"]["preparation_peak_bytes"]) <= meta["max_device_bytes"], "reported device budget inconsistent")
                quality_objective = "multilabel" if meta["objective"] == "binary" and meta["outputs"] > 1 else meta["objective"]
                path = OUT / "quality" / (name + "-model-vs-base.json")
                track(path)
                comparison = read_json(path)
                comparison_inputs(comparison)
                row["metrics_checked"], failures = check_comparison(comparison, quality_objective, meta["outputs"], evaluator)
                original_row = next(x for x in summary["cases"] if x["name"] == name)
                require(original_row["status"] == comparison["status"] and original_row["regressions"] == comparison["regressions"], "case summary masks comparison outcome")
                row.update(status="pass", quality_status=comparison["status"], regressed_metrics=len(failures),
                           root_histogram=meta["root_histogram"], split_policy=meta["split_policy"])
                statuses.append(comparison["status"])
                metadata[name], captures[name] = meta, capture
            except ERRORS as error:
                row["error"] = str(error)
                result["errors"].append(dict(case=name, error=str(error)))
        for reference, candidate in pairs:
            row = dict(reference=reference, candidate=candidate, status="invalid_evidence")
            result["comparisons"].append(row)
            try:
                helpers.matching(metadata[reference], metadata[candidate])
                for key in ("tree_execution", "quantize_policy", "tree_export_batch_requested", "tree_export_batch_effective"):
                    require(metadata[reference][key] == metadata[candidate][key], "paired protocol differs: " + key)
                require(captures[reference]["executable_sha256"] == captures[candidate]["executable_sha256"], "paired executables differ")
                path = OUT / "quality" / (candidate + "-vs-" + reference + ".json")
                track(path)
                report = read_json(path)
                require((report["reference"], report["candidate"]) == (reference, candidate), "comparison file contains another pair")
                for key in ("predictions", "model", "encoded_domain"):
                    require(key in report, "comparison is missing completed analysis: " + key)
                comparison_inputs(report["comparison"])
                objective, _, _ = helpers.identity(metadata[candidate])
                row["metrics_checked"], failures = check_comparison(report["comparison"], objective, metadata[candidate]["outputs"], evaluator)
                original_row = next(x for x in summary["comparisons"] if (x["reference"], x["candidate"]) == (reference, candidate))
                require(report["status"] == original_row["status"] == report["comparison"]["status"], "pair summary masks comparison outcome")
                require(report["regressions"] == original_row["regressions"] == report["comparison"]["regressions"], "pair summary omits metric failures")
                row.update(status="pass", quality_status=report["status"], regressed_metrics=len(failures))
                statuses.append(report["status"])
            except ERRORS as error:
                row["error"] = str(error)
                result["errors"].append(dict(reference=reference, candidate=candidate, error=str(error)))
        require(summary["status"] == ("regression" if "regression" in statuses else "pass"), "overall quality status masks component failures")
    except ERRORS as error:
        result["errors"].append(dict(error=str(error)))
    for artifact in tracked.values():
        try:
            require(digest(artifact["path"]) == artifact["sha256"], "artifact changed during verification: " + artifact["path"])
        except ERRORS as error:
            result["errors"].append(dict(artifact=artifact["path"], error=str(error)))
    result["artifacts"] = list(tracked.values())
    result["status"] = "invalid_evidence" if result["errors"] else "pass"
    result["finished_unix_seconds"] = time.time()
    with destination.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    marker.unlink()
    print(json.dumps({key: result[key] for key in ("status", "quality_status", "errors")}), flush=True)
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ERRORS as error:
        print(json.dumps(dict(status="invalid_evidence", error=str(error))), file=sys.stderr)
        raise SystemExit(2)
