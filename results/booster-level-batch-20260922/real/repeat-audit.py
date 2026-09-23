#!/usr/bin/env python3
"""CPU-only within-policy repeat audit and concrete matched-policy witness."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from campaign import DATA, DATASETS, HERE, digest
from fixtures import HEADER, MAGIC
from summarize import audit_case, strict_metric_gate, topology


def target_view(path):
    with path.open("rb") as stream:
        magic, version, rows, features, outputs, objective, classes = HEADER.unpack(stream.read(HEADER.size))
    if magic != MAGIC or version != 1 or path.stat().st_size != HEADER.size + 4 * rows * (features + outputs):
        raise ValueError("invalid fixture extent/header")
    return np.memmap(path, mode="r", dtype="<f4", offset=HEADER.size + 4 * rows * features,
                     shape=(rows, outputs)), objective, classes


def decision_differences(reference, candidate, objective):
    if objective == 0:
        return {"kind": "not applicable to regression", "changed_decisions": None, "rows_with_changed_decisions": None}
    changed = (reference >= .5) != (candidate >= .5) if objective == 1 else np.argmax(reference, axis=1) != np.argmax(candidate, axis=1)
    return {"kind": "independent binary outputs at probability >= 0.5" if objective == 1 else "multiclass argmax",
            "changed_decisions": int(np.count_nonzero(changed)),
            "rows_with_changed_decisions": int(np.count_nonzero(np.any(changed, axis=1) if objective == 1 else changed))}


def artifact_identity(path, capture):
    return {"case": path.name, "implementation": capture["implementation"], "capture_sha256": digest(path / "capture.json"),
            "model_sha256": digest(path / "result/model.ghb"), "predictions_sha256": capture["predictions_sha256"],
            "quality_sha256": capture["quality_sha256"], "binary_sha256": capture["binary_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=HERE)
    parser.add_argument("--output-json", type=Path, default=HERE / "repeat-audit.json")
    parser.add_argument("--output-md", type=Path, default=HERE / "repeat-audit.md")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite a previous repeat audit")
    root = args.real_root.resolve()
    summary_path = root / "summary/summary.json"
    summary_digest = digest(summary_path)
    summary = json.loads(summary_path.read_text())
    provenance = {"validation": {"case_captures": []}, "test": {"case_captures": []}, "frozen_source_sha256": {}}
    binaries = set()

    @lru_cache(maxsize=None)
    def case(path):
        raw = json.loads((path / "capture.json").read_text())
        capture, quality, measured = audit_case(path, raw["dataset"], raw["implementation"], raw["split"], raw["config"], provenance, binaries)
        prediction = np.memmap(path / "result/predictions.f64", mode="r", dtype="<f8",
                               shape=(measured["evaluation_rows"], measured["outputs"]))
        return capture, quality, measured, prediction

    pairs = []
    for dataset in DATASETS:
        for implementation in ("custom-per-output", "custom-output-batch"):
            reference_path = root / "test" / f"test-{dataset}-r0-{implementation}"
            rc, rq, rm, rp = case(reference_path)
            for repetition in (1, 2):
                candidate_path = root / "test" / f"test-{dataset}-r{repetition}-{implementation}"
                cc, cq, cm, cp = case(candidate_path)
                for key in ("implementation", "dataset", "split", "config", "training_fixture_sha256", "evaluation_fixture_sha256", "binary_sha256"):
                    if rc[key] != cc[key]:
                        raise ValueError("within-policy repeat contract differs: " + key)
                if rm["objective"] != cm["objective"] or rp.shape != cp.shape:
                    raise ValueError("repeat output contract differs")
                delta = np.abs(rp - cp)
                rt, rv, rb = topology(reference_path / "result/model.ghb")
                ct, cv, cb = topology(candidate_path / "result/model.ghb")
                pair = {"dataset": dataset, "implementation": implementation,
                        "comparison_kind": "within-policy repeated training: reference r0, candidate r" + str(repetition),
                        "reference": artifact_identity(reference_path, rc), "candidate": artifact_identity(candidate_path, cc),
                        "config": rc["config"], "training_fixture_sha256": rc["training_fixture_sha256"],
                        "evaluation_fixture_sha256": rc["evaluation_fixture_sha256"],
                        "prediction_units": "raw regression target units" if rm["objective"] == 0 else "probability",
                        "prediction_values": int(rp.size), "nonidentical_prediction_values": int(np.count_nonzero(rp != cp)),
                        "max_absolute_prediction_difference": float(np.max(delta)), "mean_absolute_prediction_difference": float(np.mean(delta)),
                        "decisions": decision_differences(rp, cp, rm["objective"]),
                        "topology_and_quantization_identical": rt == ct, "base_scores_identical": bool(np.array_equal(rb, cb)),
                        "strict_loss_nonregression": cq["metrics"]["selection_value"] <= rq["metrics"]["selection_value"],
                        "strict_all_metrics_nonregression": strict_metric_gate(rq["metrics"], cq["metrics"])}
                if rt == ct:
                    pair["max_node_value_difference"] = float(np.max(np.abs(rv - cv))) if len(rv) else 0.0
                pairs.append(pair)

    matched = [p for values in summary["custom_numerical_comparisons"].values()
               for key in ("validation_matched_configs", "test_matched_configs") for p in values[key]]
    greatest = max(matched, key=lambda p: p["max_absolute_difference"])
    left_path, right_path = [Path(greatest[key]).resolve() for key in ("per_output_case", "output_batch_case")]
    lc, _, lm, lp = case(left_path)
    rc, _, rm, rp = case(right_path)
    if lc["implementation"] != "custom-per-output" or rc["implementation"] != "custom-output-batch":
        raise ValueError("largest matched-policy comparison labels differ")
    for key in ("dataset", "split", "config", "training_fixture_sha256", "evaluation_fixture_sha256", "binary_sha256"):
        if lc[key] != rc[key]:
            raise ValueError("largest matched-policy comparison contract differs")
    delta = np.abs(lp - rp)
    row, output = [int(v) for v in np.unravel_index(np.argmax(delta), delta.shape)]
    if float(delta[row, output]) != greatest["max_absolute_difference"]:
        raise ValueError("largest matched-policy difference no longer matches preserved summary")
    training_path = DATA / lc["dataset"] / "train.ghb"
    evaluation_path = DATA / lc["dataset"] / (lc["split"] + ".ghb")
    training_targets, objective, classes = target_view(training_path)
    evaluation_targets, eval_objective, eval_classes = target_view(evaluation_path)
    if objective != eval_objective or classes != eval_classes or objective != lm["objective"]:
        raise ValueError("witness target objective contract differs")
    actual_target = float(evaluation_targets[row, 0 if objective == 2 else output])
    positives = int(np.count_nonzero(training_targets[:, 0] == output)) if objective == 2 else int(np.count_nonzero(training_targets[:, output] == 1)) if objective == 1 else None
    label_name = None
    if lc["dataset"] == "delicious":
        labels = [node.attrib["name"] for node in ET.parse(DATA.parent / "delicious-source/delicious.xml").getroot()]
        labels[612] = "TAG_m\\'usica"
        label_name = labels[output]
    witness = {"comparison_kind": "matched-policy comparison, reference per-output and candidate output-batch",
               "matched_pairs_searched": len(matched), "dataset": lc["dataset"], "split": lc["split"], "config": lc["config"],
               "reference": artifact_identity(left_path, lc), "candidate": artifact_identity(right_path, rc),
               "row_index_zero_based": row, "output_index_zero_based": output, "label_name": label_name,
               "actual_target": actual_target, "reference_prediction": float(lp[row, output]),
               "candidate_prediction": float(rp[row, output]), "absolute_difference": float(delta[row, output]),
               "prediction_units": "raw regression target units" if objective == 0 else "probability",
               "decision_difference_at_witness": None if objective == 0 else bool((lp[row, output] >= .5) != (rp[row, output] >= .5)) if objective == 1 else bool(np.argmax(lp[row]) != np.argmax(rp[row])),
               "decisions_across_entire_pair": decision_differences(lp, rp, objective),
               "training_positive_count_for_label": positives, "training_rows": int(training_targets.shape[0]),
               "evaluation_positive_count_for_label": int(np.count_nonzero(evaluation_targets[:, output] == 1)) if objective == 1 else None,
               "training_fixture": {"path": str(training_path), "sha256": digest(training_path)},
               "evaluation_fixture": {"path": str(evaluation_path), "sha256": digest(evaluation_path)}}
    if digest(summary_path) != summary_digest:
        raise RuntimeError("preserved summary changed during repeat audit")
    result = {"scope": "CPU audit of existing completed training observations; no new training, no policy changes, no GPU work",
              "interpretation": "Within-policy differences quantify observed repeat variability. They do not excuse matched-policy regressions or establish a cause for those differences. Zero-allowance failures remain failed. In the unchanged gate function, baseline means the explicitly labeled reference run.",
              "sources_sha256": {p.name: digest(p) for p in [Path(__file__).resolve(), HERE / "summarize.py", HERE / "evaluate.py", HERE / "fixtures.py"]},
              "preserved_summary_sha256": summary_digest, "within_policy_pairs": pairs, "largest_matched_policy_difference_witness": witness}
    lines = ["# Within-policy repeat audit", "", result["interpretation"], "", "Reference is test repetition r0; candidate is r1 or r2 of the same policy, dataset, configuration and binary. These are not comparisons between policies. Regression predictions use target units; binary/multiclass predictions are probabilities.", "",
             "| Dataset | Policy | Pairs | Maximum prediction difference | Changed-topology pairs | Strict loss failures | Any-metric failures |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for implementation in ("custom-per-output", "custom-output-batch"):
            group = [p for p in pairs if p["dataset"] == dataset and p["implementation"] == implementation]
            lines.append(f"| {dataset} | {implementation} | {len(group)} | {max(p['max_absolute_prediction_difference'] for p in group):.10g} | {sum(not p['topology_and_quantization_identical'] for p in group)} | {sum(not p['strict_loss_nonregression'] for p in group)} | {sum(not p['strict_all_metrics_nonregression']['passed'] for p in group)} |")
    lines += ["", "# Largest preserved matched-policy difference", "",
              f"Among {len(matched)} matched-policy comparisons, the largest difference is {witness['absolute_difference']:.17g} {witness['prediction_units']} on {witness['dataset']} {witness['split']}, zero-based row {row}, output {output} ({label_name}). The actual target is {actual_target:g}; per-output predicts {witness['reference_prediction']:.17g}, and output-batch predicts {witness['candidate_prediction']:.17g}.", "",
              f"Reference case: `{left_path.name}`. Candidate case: `{right_path.name}`. Training positives for this label: {positives}/{training_targets.shape[0]}. The decision changes at this witness: {witness['decision_difference_at_witness']}. Across the entire pair, {witness['decisions_across_entire_pair']['changed_decisions']} decisions change across {witness['decisions_across_entire_pair']['rows_with_changed_decisions']} rows. Fixture, model, prediction and capture hashes are retained in JSON.", ""]
    with args.output_json.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with args.output_md.open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"within_policy_pairs": len(pairs), "all_metric_failures": sum(not p["strict_all_metrics_nonregression"]["passed"] for p in pairs),
                      "witness_cases": [left_path.name, right_path.name], "witness_difference": witness["absolute_difference"]}))


if __name__ == "__main__":
    main()
