#!/usr/bin/env python3
"""CPU quality audit for run_optimized_campaign.py. Existing evidence is immutable.

Run after the complete 28-case campaign. Exit 0: all learned-model gates pass;
1: strict quality regressions retained; 2: invalid or mismatched evidence.
Preparation-only runs do not participate in learned-model promotion gates.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPERS = OUT / "audit_quality.py"
spec = importlib.util.spec_from_file_location("resident_quality_helpers", HELPERS)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def plan():
    hybrid_pairs = []
    for suffix in ("a", "b"):
        hybrid_pairs += [(f"optimized-scalar-hybrid-{suffix}", f"optimized-scalar-{mode}-{suffix}")
                         for mode in ("stream", "graph")]
    export_pairs = []
    for outputs in (129, 1024):
        candidates = [f"optimized-{outputs}-graph-batch{batch}-{suffix}"
                      for suffix in ("a", "b") for batch in (0, 16)]
        candidates += [f"optimized-{outputs}-graph-batch1", f"optimized-{outputs}-stream-batch16"]
        hybrid_pairs += [(f"outputs{outputs}-hybrid", candidate) for candidate in candidates]
        export_pairs += [(f"optimized-{outputs}-graph-batch0-{suffix}", f"optimized-{outputs}-graph-batch16-{suffix}")
                         for suffix in ("a", "b")]
    hybrid_pairs += [(f"optimized-4096-hybrid-{suffix}", f"optimized-4096-graph-{suffix}") for suffix in ("a", "b")]
    hybrid_pairs += [("confirm-129-hybrid", "confirm-129-graph-batch16")]
    preparation = [(f"quantize-{rows}x{features}-hybrid", f"optimized-quantize-{rows}x{features}-{policy}")
                   for rows, features in ((1048576, 8), (16777216, 1)) for policy in ("radix8", "radix4")]
    all_names = sorted({name for pair in hybrid_pairs + export_pairs + preparation for name in pair})
    campaign_names = [name for name in all_names if name.startswith(("optimized-", "confirm-"))]
    assert len(campaign_names) == 28 and len(hybrid_pairs) == 19 and len(export_pairs) == 4
    return all_names, campaign_names, hybrid_pairs, export_pairs, preparation


def validate_capture(name):
    directory = OUT / name
    if (directory / ".incomplete").exists():
        raise ValueError("incomplete benchmark output")
    result = a.read_json(directory / "result.json")
    capture = a.read_json(OUT / (name + "-capture.json"))
    if capture["returncode"] != 0 or capture["executable_unchanged"] is not True:
        raise ValueError("failed capture or executable changed during capture")
    if result != a.read_json(OUT / (name + "-stdout.json")):
        raise ValueError("saved result and captured stdout differ")
    executable = Path(capture["command"][0])
    choices = []
    if executable.is_relative_to(ROOT):
        choices = [OUT / snapshot / executable.relative_to(ROOT) for snapshot in ("optimized-provenance", "final-provenance", "first-provenance")]
    choices.append(executable)
    retained = next((path for path in choices if path.is_file() and a.sha(path) == capture["executable_sha256"]), None)
    if retained is None:
        raise ValueError("no retained executable matches capture SHA256")
    if result["kind"] != "ghb.training" or result["schema_version"] != 1 or result["generator_version"] != 2:
        raise ValueError("unsupported benchmark/generator schema")
    if result["validation"]["serialization_equal"] is not True:
        raise ValueError("serialization validation failed")
    error = result["validation"]["cpu_gpu_max_abs_error"]
    if isinstance(error, bool) or not math.isfinite(error) or error < 0:
        raise ValueError("invalid CPU/GPU prediction validation")
    if result["trees"] != result["rounds"] * result["outputs"]:
        raise ValueError("tree count differs from rounds times outputs")
    if result["test_seed"] != (result["seed"] ^ 0x9E3779B97F4A7C15):
        raise ValueError("unexpected held-out seed")
    for key in a.IDENTITY:
        if key not in result:
            raise ValueError("missing protocol dimension: " + key)
    paths = [directory / filename for filename in ("result.json", "targets.csv", "predictions.csv", "baseline.csv", "model.ghb")]
    paths += [OUT / (name + suffix) for suffix in ("-capture.json", "-stdout.json", "-stderr.txt")]
    return result, capture, [a.artifact(path) for path in paths + [retained]]


def feature_bytes(model):
    """Keep signed-zero bits as well as exact float32 cut/category values."""
    payload = struct.pack("<I", len(model["features"]))
    for kind, values in model["features"]:
        payload += struct.pack("<II", kind, len(values)) + struct.pack("<" + "f" * len(values), *values)
    return payload


def leaf_regions(model, tree):
    _, nodes = tree
    domains = [(1 << (len(values) + (2 if kind == 0 else 1))) - 1 for kind, values in model["features"]]
    result, pending = [], [("", {})]
    while pending:
        path, allowed = pending.pop()
        feature, _, _, threshold, missing, value = nodes[path]
        if feature == -1:
            result.append((allowed, Fraction.from_float(value)))
            continue
        kind, _ = model["features"][feature]
        present_left = (((1 << (threshold + 1)) - 1) & ~1) if kind == 0 else (1 << threshold if threshold else 0)
        left = present_left | int(bool(missing))
        current = allowed.get(feature, domains[feature])
        for side, part in (("L", left), ("R", domains[feature] ^ left)):
            narrowed = current & part
            if narrowed:
                child = dict(allowed)
                child[feature] = narrowed
                pending.append((path + side, child))
    return result


def upward(value):
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction.from_float(rounded) < value else rounded


def exact_region_comparison(old, new):
    if feature_bytes(old) != feature_bytes(new):
        raise ValueError("feature metadata differs; encoded-domain equivalence cannot be assumed")
    if (old["objective"], old["outputs"], len(old["trees"])) != (new["objective"], new["outputs"], len(new["trees"])):
        raise ValueError("paired model shapes differ")
    bounds = [abs(Fraction.from_float(x) - Fraction.from_float(y)) for x, y in zip(old["bases"], new["bases"])]
    maximum_tree = Fraction(0)
    intersections = 0
    for before, after in zip(old["trees"], new["trees"]):
        if before[0] != after[0]:
            raise ValueError("paired tree output differs")
        tree_max = Fraction(0)
        for region_a, value_a in leaf_regions(old, before):
            for region_b, value_b in leaf_regions(new, after):
                if all(region_a[feature] & region_b[feature] for feature in region_a.keys() & region_b.keys()):
                    intersections += 1
                    tree_max = max(tree_max, abs(value_a - value_b))
        maximum_tree = max(maximum_tree, tree_max)
        bounds[before[0]] += tree_max
    maximum = max(bounds)
    with localcontext() as context:
        context.prec = 2048
        exact_decimal = str(Decimal(maximum.numerator) / Decimal(maximum.denominator))
    return {"nonempty_leaf_region_intersections": intersections,
            "maximum_tree_region_value_difference_upward": upward(maximum_tree),
            "maximum_raw_margin_bound_upward": upward(maximum),
            "maximum_raw_margin_bound_exact_decimal": exact_decimal,
            "per_output_raw_margin_bound_upward": [upward(value) for value in bounds],
            "method": "Intersect all paired leaf regions over every metadata-defined encoded bin, including missing bin0. Subtract serialized binary64 leaf values and sum per-tree maxima plus base-score differences using exact fractions; float bounds round upward. Different thresholds are matched by the encoded regions they actually route.",
            "scope": "Bound on exact-real raw-margin model functions over all encoded-bin combinations; excludes additional runtime accumulation/transform rounding and is not a quality-generalization claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT / "quality-optimized")
    args = parser.parse_args()
    names, campaign_names, hybrid_pairs, export_pairs, preparation_pairs = plan()
    missing = [name for name in names if not (OUT / name / "result.json").is_file()]
    if missing:
        raise ValueError("campaign/reference evidence incomplete: " + ", ".join(missing))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".incomplete"
    marker.write_text("CPU optimized quality audit in progress\n")
    evaluator = a.load_evaluator()
    sources = [a.artifact(path) for path in (Path(__file__), HELPERS, a.EVALUATOR, OUT / "run_campaign.py", OUT / "run_optimized_campaign.py")]
    summary = {"schema_version": 1, "kind": "optimized_resident_campaign_quality_audit", "gpu_used": False,
               "started_unix_seconds": time.time(), "allowance": 0.0, "sources": sources,
               "expected_campaign_cases": campaign_names, "all_evidence_cases": names,
               "cases": [], "optimized_vs_hybrid": [], "batch16_vs_compact0": [], "preparation_only": [],
               "errors": [], "raw_artifacts": [], "limitations": [
                   "Every applicable quality metric and every output receives zero allowance. Last-decimal failures remain failed.",
                   "Preparation-only rounds0 cases are checked for evidence integrity and exact feature metadata, and excluded from learned-model promotion gates.",
                   "The four batch16-versus-compact0 comparisons require identical executable hashes and all training/quantization settings except the intended export-batch setting.",
                   "First-hybrid references for outputs129/1024 are retained historical executions; same-binary export comparisons isolate that change more closely.",
                   "Synthetic held-out results do not establish NLP performance or generalization to other datasets. Large output counts do not increase independent held-out rows.",
                   "All-bin region comparisons cover missing and all retained present bins. Their exact-real raw-margin bounds exclude additional runtime arithmetic rounding."]}
    metadata, captures, reports, models = {}, {}, {}, {}
    for name in names:
        row = {"case": name, "campaign_case": name in campaign_names}
        summary["cases"].append(row)
        try:
            meta, capture, artifacts = validate_capture(name)
            metadata[name], captures[name] = meta, capture
            summary["raw_artifacts"].extend(artifacts)
            objective, dataset, split = a.identity(meta)
            row.update(objective=objective, rows=meta["test_rows"], outputs=meta["outputs"], rounds=meta["rounds"])
            directory = OUT / name
            target = directory / "targets.csv"
            model_report = evaluator.evaluate(target, directory / "predictions.csv", objective, dataset, split,
                                               meta["outputs"] if objective == "multiclass" else None)
            base_report = evaluator.evaluate(target, directory / "baseline.csv", objective, dataset, split,
                                              meta["outputs"] if objective == "multiclass" else None)
            if model_report["sample_count"] != meta["test_rows"] or model_report["outputs"] != meta["outputs"]:
                raise ValueError("CSV dimensions differ from benchmark metadata")
            model_path, base_path = destination / (name + "-model.json"), destination / (name + "-base.json")
            a.write_json(model_path, model_report)
            a.write_json(base_path, base_report)
            models[name] = a.read_model(directory / "model.ghb")
            if len(models[name]["trees"]) != meta["trees"] or models[name]["outputs"] != meta["outputs"]:
                raise ValueError("serialized model dimensions differ from result")
            reports[name] = model_path
            row["reports"] = [a.artifact(path) for path in (model_path, base_path)]
            if meta["rounds"]:
                comparison = evaluator.compare(base_path, model_path, 0.0)
                output = destination / (name + "-model-vs-base.json")
                a.write_json(output, comparison)
                row["model_vs_base"] = a.compact_comparison(comparison)
                row["reports"].append(a.artifact(output))
                row["status"] = comparison["status"]
            else:
                row["status"] = "preparation_only"
                row["learned_model_promotion_gate"] = "excluded: zero trees"
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append({"case": name, "error": str(error)})
        print(name, row["status"], flush=True)
    for category, pairs in (("optimized_vs_hybrid", hybrid_pairs), ("batch16_vs_compact0", export_pairs)):
        for reference, candidate in pairs:
            row = {"reference": reference, "candidate": candidate}
            summary[category].append(row)
            try:
                a.matching(metadata[reference], metadata[candidate])
                if not metadata[candidate]["rounds"]:
                    raise ValueError("zero-round run entered learned-model comparison")
                if category == "batch16_vs_compact0":
                    if captures[reference]["executable_sha256"] != captures[candidate]["executable_sha256"]:
                        raise ValueError("export comparison executable hashes differ")
                    for key in ("tree_execution", "quantize_policy"):
                        if metadata[reference][key] != metadata[candidate][key]:
                            raise ValueError("export comparison differs in " + key)
                    # CLI requests are retained even if a resource cap lowers the effective width.
                    old_command, new_command = captures[reference]["command"], captures[candidate]["command"]
                    old_batch = old_command[old_command.index("--tree-export-batch") + 1]
                    new_batch = new_command[new_command.index("--tree-export-batch") + 1]
                    if (old_batch, new_batch) != ("0", "16"):
                        raise ValueError("incorrect intended compact0/batch16 pair")
                before, after = OUT / reference, OUT / candidate
                if a.sha(before / "targets.csv") != a.sha(after / "targets.csv"):
                    raise ValueError("exact target/weight identity differs")
                comparison = evaluator.compare(reports[reference], reports[candidate], 0.0)
                row.update(a.compact_comparison(comparison))
                row["predictions"] = a.prediction_changes(before / "predictions.csv", after / "predictions.csv", before / "targets.csv", a.identity(metadata[candidate])[0])
                row["model"] = a.model_changes(before / "model.ghb", after / "model.ghb")
                if feature_bytes(models[reference]) != feature_bytes(models[candidate]):
                    raise ValueError("exact serialized feature cuts/categories differ")
                row["encoded_domain"] = exact_region_comparison(models[reference], models[candidate])
                output = destination / (candidate + "-vs-" + reference + ".json")
                a.write_json(output, {"comparison": comparison, "prediction_changes": row["predictions"],
                                      "model_changes": row["model"], "encoded_domain": row["encoded_domain"]})
                row["report"] = a.artifact(output)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                row.update(status="invalid_evidence", error=str(error))
                summary["errors"].append({"reference": reference, "candidate": candidate, "error": str(error)})
            print(category, candidate, row["status"], flush=True)
    for reference, candidate in preparation_pairs:
        row = {"reference": reference, "candidate": candidate, "learned_model_promotion_gate": "excluded: zero trees"}
        summary["preparation_only"].append(row)
        try:
            a.matching(metadata[reference], metadata[candidate])
            if metadata[reference]["rounds"] or metadata[candidate]["rounds"]:
                raise ValueError("nonzero rounds in preparation-only pair")
            if a.sha(OUT / reference / "targets.csv") != a.sha(OUT / candidate / "targets.csv"):
                raise ValueError("preparation target/weight identity differs")
            old, new = feature_bytes(models[reference]), feature_bytes(models[candidate])
            row.update(reference_feature_sha256=hashlib.sha256(old).hexdigest(), candidate_feature_sha256=hashlib.sha256(new).hexdigest(),
                       exact_feature_metadata_equal=old == new)
            if old != new:
                raise ValueError("preparation feature metadata differs from matching frozen hybrid")
            row["status"] = "pass"
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            row.update(status="invalid_evidence", error=str(error))
            summary["errors"].append({"reference": reference, "candidate": candidate, "error": str(error)})
    for item in sources + summary["raw_artifacts"]:
        if a.sha(item["path"]) != item["sha256"]:
            summary["errors"].append({"artifact_changed_during_audit": item["path"]})
    statuses = [row["status"] for category in ("cases", "optimized_vs_hybrid", "batch16_vs_compact0") for row in summary[category]]
    summary["status"] = "invalid_evidence" if summary["errors"] else "regression" if "regression" in statuses else "pass"
    summary["finished_unix_seconds"] = time.time()
    a.write_json(destination / "summary.json", summary)
    lines = ["# Optimized resident quality audit", "", f"Status: **{summary['status']}**. Zero allowance is retained for every applicable metric.", ""]
    for category, label in (("optimized_vs_hybrid", "Optimized versus matching hybrid"), ("batch16_vs_compact0", "Same-binary batch16 versus compact0")):
        lines += ["## " + label, "", "| Candidate | Gate | Regressed metrics | Maximum prediction difference | Decision changes | All-bin raw-margin bound |", "|---|---|---:|---:|---:|---:|"]
        for row in summary[category]:
            if "encoded_domain" in row:
                p = row["predictions"]
                lines.append(f"| {row['candidate']} | {row['status']} | {len(row['regressions'])} | {p['max_abs_difference']:.17g} | {p['decision_changes'] if p['decision_changes'] is not None else 'N/A'} | {row['encoded_domain']['maximum_raw_margin_bound_upward']:.17g} |")
            else:
                lines.append(f"| {row['candidate']} | invalid evidence | — | — | — | — |")
        lines.append("")
    trained = [row for row in summary["cases"] if row.get("rounds", 0)]
    lines += [f"Model-versus-base: {sum(row['status'] == 'pass' for row in trained)}/{len(trained)} trained evidence cases passed.", "",
              "Preparation-only exact feature metadata checks:", ""]
    lines += [f"- {row['candidate']}: {row['status']} against {row['reference']}." for row in summary["preparation_only"]]
    lines.append("")
    lines += ["- " + text for text in summary["limitations"]]
    if summary["errors"]:
        lines += ["", "Evidence errors are retained in summary.json."]
    with (destination / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    marker.unlink()
    return 2 if summary["status"] == "invalid_evidence" else 1 if summary["status"] == "regression" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print("ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
