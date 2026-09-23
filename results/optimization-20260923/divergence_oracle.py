#!/usr/bin/env python3
"""Bounded saved-model CPU reference; see DIVERGENCE_ORACLE_DESIGN.md.

No CUDA, production preprocessing/training, or performance measurement. The
Decimal oracle and exact Fraction sum of host-float derivatives are distinct
from unavailable GPU histogram snapshots. Never reinterpret a quality failure.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "results/booster-level-batch-20260922/data/fixtures/delicious"
CASES = ("pair0-warp32", "pair0-warp-wide", "pair1-warp32")
READER_PATH = ROOT / "results/booster-resident-20260922/audit_quality.py"
spec = importlib.util.spec_from_file_location("saved_divergence_model_reader", READER_PATH)
READER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(READER)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def row_identity(rows):
    data = np.asarray(rows, dtype="<u4")
    return {"rows": len(rows), "sha256_little_endian_u32_row_ids": hashlib.sha256(data.tobytes()).hexdigest()}


def fixture(path):
    with path.open("rb") as stream:
        magic, version, rows, columns, outputs, objective, classes = struct.unpack("<8s6I", stream.read(32))
    require((magic, version, columns, outputs, objective) == (b"GHBDS001", 1, 500, 983, 1), "unexpected Delicious contract")
    require(path.stat().st_size == 32 + 4 * rows * (columns + outputs), "fixture extent differs")
    x = np.memmap(path, mode="r", dtype="<f4", offset=32, shape=(rows, columns))
    y = np.memmap(path, mode="r", dtype="<f4", offset=32 + rows * columns * 4, shape=(rows, outputs))
    require(np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all(), "this bounded oracle requires binary features/targets without missing values")
    return x, y


def decision(node):
    return node[0], node[3], node[4]


def first_difference(before, after):
    for path in sorted(set(before) | set(after), key=lambda p: (len(p), p)):
        if path not in before or path not in after or decision(before[path]) != decision(after[path]):
            return path
    return None


def membership(x, tree, path):
    rows = np.arange(len(x), dtype=np.uint32)
    for depth, branch in enumerate(path):
        node = tree[path[:depth]]
        require(node[0] >= 0 and node[3] == 1, "ancestor is not the expected binary-feature split")
        # Fitted numeric cut=0: lower_bound maps value0->bin1, value1->bin2.
        left = x[rows, node[0]] == 0
        rows = rows[left if branch == "L" else ~left]
    return rows


def probability(margin, digits):
    with localcontext() as context:
        context.prec = digits
        m = Decimal.from_float(margin)
        e = (-m if m >= 0 else m).exp()
        return +(Decimal(1) / (1 + e) if m >= 0 else e / (1 + e))


def stats(n, positive, p):
    return n * p - positive, n * max(p * (1 - p), Decimal.from_float(1e-16))


def score(n, positive, left_n, left_positive, p, penalty):
    g, h = stats(n, positive, p)
    lg, lh = stats(left_n, left_positive, p)
    rg, rh = stats(n - left_n, positive - left_positive, p)
    value = lg * lg / (2 * (lh + penalty)) + rg * rg / (2 * (rh + penalty)) - g * g / (2 * (h + penalty))
    return value, -lg / (lh + penalty), -rg / (rh + penalty), -g / (h + penalty)


def host_derivatives(margin):
    if margin >= 0:
        p = 1 / (1 + math.exp(-margin))
    else:
        e = math.exp(margin)
        p = e / (1 + e)
    return p, p - 1, max(p * (1 - p), 1e-16)


def rational_score(n, positive, left_n, left_positive, derivatives, penalty):
    g0, g1, h0 = map(Fraction.from_float, derivatives)
    penalty = Fraction.from_float(penalty)
    def benefit(rows, positives):
        gradient = (rows - positives) * g0 + positives * g1
        return gradient * gradient / (2 * (rows * h0 + penalty))
    return benefit(left_n, left_positive) + benefit(n - left_n, positive - left_positive) - benefit(n, positive)


def fraction_decimal(value, digits=80):
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def partition_signature(n, positive, left_n, left_positive):
    return tuple(sorted(((left_n, left_positive), (n - left_n, positive - left_positive))))


def node_oracle(x, y, output, path, trees, bases, config):
    rows = membership(x, trees[0], path)
    other_rows = membership(x, trees[1], path)
    require(np.array_equal(rows, other_rows), "first differing node inputs have different memberships")
    require(struct.pack("<d", bases[0]) == struct.pack("<d", bases[1]), "first differing node base margins differ")
    n, positive = len(rows), int(np.count_nonzero(y[rows, output]))
    selected_x, selected_y = x[rows, :], y[rows, output].astype(bool)
    left_n = np.count_nonzero(selected_x == 0, axis=0)
    left_positive = np.count_nonzero(selected_x[selected_y] == 0, axis=0)
    require(int(np.count_nonzero(selected_y)) == positive, "positive count cross-check failed")
    derivatives = host_derivatives(bases[0])
    candidates, groups = [], {}
    with localcontext() as context:
        context.prec = 80
        p = probability(bases[0], 80)
        penalty = Decimal.from_float(config["l2"])
        min_h = Decimal.from_float(config["min_child_hessian"])
        for feature in range(x.shape[1]):
            nl, kl = int(left_n[feature]), int(left_positive[feature])
            _, hl = stats(nl, kl, p)
            _, hr = stats(n - nl, positive - kl, p)
            if min(nl, n - nl) < config["min_leaf_rows"] or min(hl, hr) < min_h:
                continue
            signature = partition_signature(n, positive, nl, kl)
            if signature not in groups:
                gain, _, _, _ = score(n, positive, *signature[0], p, penalty)
                host_gain = rational_score(n, positive, *signature[0], derivatives, config["l2"])
                groups[signature] = (gain, host_gain)
            gain, host_gain = groups[signature]
            for missing in (0, 1):
                candidates.append({"feature": feature, "threshold": 1, "missing_left": missing,
                    "left_rows": nl, "left_positives": kl, "right_rows": n - nl, "right_positives": positive - kl,
                    "signature": signature, "gain": gain, "host_gain": host_gain})
        require(candidates, "no feasible candidate in selected divergence node")
        ordered = sorted(candidates, key=lambda c: (-c["gain"], c["feature"], c["threshold"], c["missing_left"]))
        best = ordered[0]
        best_distinct = next((c for c in ordered if c["signature"] != best["signature"]), None)
        selections = []
        for which, tree in zip(("reference", "comparison"), trees):
            node = tree[path]
            selected = next((c for c in ordered if (c["feature"], c["threshold"], c["missing_left"]) == decision(node)), None)
            require(selected is not None, "saved split is absent from the feasible count-based domain")
            gain, lw, rw, pw = score(n, positive, selected["left_rows"], selected["left_positives"], p, penalty)
            selected_rows = rows[x[rows, selected["feature"]] == 0]
            greater = sum(c["gain"] > selected["gain"] for c in ordered)
            host_greater = sum(c["host_gain"] > selected["host_gain"] for c in ordered)
            with localcontext() as high:
                high.prec = 120
                p120 = probability(bases[0], 120)
                selected120 = score(n, positive, selected["left_rows"], selected["left_positives"], p120, penalty)[0]
                best120 = score(n, positive, best["left_rows"], best["left_positives"], p120, penalty)[0]
                gap120 = best120 - selected120
            selections.append({"case_role": which, "decision": decision(node),
                "left_membership": row_identity(selected_rows), "ideal_gain_80_digits": str(gain),
                "ideal_gain_120_digits": str(selected120), "ideal_gap_to_best_120_digits": str(gap120),
                "rank_ignoring_tie_break": 1 + greater, "host_derivative_exact_rational_rank": 1 + host_greater,
                "same_unordered_child_count_signature_as_best": selected["signature"] == best["signature"],
                "same_ideal_gain_as_best_at_80_digits": selected["gain"] == best["gain"],
                "left_rows": selected["left_rows"], "left_positives": selected["left_positives"],
                "right_rows": selected["right_rows"], "right_positives": selected["right_positives"],
                "ideal_leaf_before_learning_rate": {"parent": str(pw), "left": str(lw), "right": str(rw)},
                "host_derivative_exact_rational_gain": fraction_decimal(selected["host_gain"]),
                "host_derivative_exact_rational_gap_to_best": fraction_decimal(max(c["host_gain"] for c in candidates) - selected["host_gain"])})
        parent_g, parent_h = stats(n, positive, p)
        public = lambda c: {"feature": c["feature"], "threshold": c["threshold"], "missing_left": c["missing_left"],
            "left_rows": c["left_rows"], "left_positives": c["left_positives"], "right_rows": c["right_rows"],
            "right_positives": c["right_positives"], "gain_80_digits": str(c["gain"]),
            "host_derivative_exact_rational_gain": fraction_decimal(c["host_gain"])}
        same_gain = [c for c in ordered if c["gain"] == best["gain"]]
        result = {"output": output, "path": path, "membership": row_identity(rows), "positives": positive,
            "margin_hex": bases[0].hex(), "ideal_probability": str(p), "parent_gradient": str(parent_g), "parent_hessian": str(parent_h),
            "host_derivative_hex": {key: value.hex() for key, value in zip(("g0", "g1", "h"), derivatives)},
            "features_checked": x.shape[1], "threshold_domain": [0, 1, 2],
            "thresholds_0_and_2": "rejected because no missing values and min_leaf_rows=1 make one child empty",
            "feasible_candidates": len(ordered), "positive_gain_candidates": sum(c["gain"] > 0 for c in ordered),
            "distinct_child_count_signatures": len(groups), "best": public(best),
            "runner_up": public(ordered[1]) if len(ordered) > 1 else None,
            "winner_runner_up_gain_margin": str(best["gain"] - ordered[1]["gain"]) if len(ordered) > 1 else None,
            "best_distinct_count_signature": public(best_distinct) if best_distinct else None,
            "margin_to_best_distinct_count_signature": str(best["gain"] - best_distinct["gain"]) if best_distinct else None,
            "cooptimal_candidate_count": len(same_gain), "cooptimal_features": sorted({c["feature"] for c in same_gain}),
            "saved_selections": selections, "all_feasible_candidates": [public(c) for c in ordered],
            "classification": "both saved choices share the ideal optimum's exact integer sufficient statistics"
                if all(s["same_unordered_child_count_signature_as_best"] for s in selections)
                else "at least one saved choice differs from the best sufficient statistics; retain quantified gaps; no GPU input snapshot"}
        left0 = rows[x[rows, selections[0]["decision"][0]] == 0]
        left1 = rows[x[rows, selections[1]["decision"][0]] == 0]
        result["saved_choices_left_membership_symmetric_difference"] = int(len(np.setxor1d(left0, left1)))
        return result


def leaf_checks(x, y, output, tree, margin, config):
    checks = []
    with localcontext() as context:
        context.prec = 80
        p, penalty, rate = probability(margin, 80), Decimal.from_float(config["l2"]), Decimal.from_float(config["learning_rate"])
        for path, node in sorted(tree.items()):
            if node[0] != -1:
                continue
            rows = membership(x, tree, path)
            positive = int(np.count_nonzero(y[rows, output]))
            g, h = stats(len(rows), positive, p)
            expected = -rate * g / (h + penalty)
            saved = Decimal.from_float(node[5])
            checks.append({"path": path, "membership": row_identity(rows), "positives": positive,
                "saved_value_hex": node[5].hex(), "count_based_ideal_value": str(expected),
                "saved_minus_ideal": str(saved - expected), "absolute_difference": float(abs(saved - expected))})
    return {"all_leaves": checks, "maximum_absolute_difference": max(c["absolute_difference"] for c in checks)}


def trace_prediction(x, tree, margin, saved, row):
    path, decisions = "", []
    while tree[path][0] != -1:
        node = tree[path]
        value = float(x[row, node[0]])
        branch = "L" if value == 0 else "R"
        decisions.append({"path": path, "feature": node[0], "threshold": node[3], "missing_left": node[4],
                          "feature_value": value, "encoded_bin": 1 + int(value), "branch": branch})
        path += branch
    leaf = tree[path][5]
    raw = margin + leaf
    host = host_derivatives(raw)[0]
    with localcontext() as context:
        context.prec = 80
        ideal = probability(raw, 80)
        error = Decimal.from_float(saved) - ideal
    return {"row": row, "decisions": decisions, "leaf_path": path, "base_margin_hex": margin.hex(),
            "leaf_increment_hex": leaf.hex(), "leaf_increment": leaf, "host_raw_margin_hex": raw.hex(),
            "saved_gpu_probability_hex": saved.hex(), "saved_gpu_probability": saved,
            "host_probability": host, "host_probability_minus_saved": host - saved,
            "high_precision_sigmoid_of_host_binary64_margin": str(ideal), "saved_minus_high_precision": str(error)}


def run(output):
    source_paths = [Path(__file__), HERE / "DIVERGENCE_ORACLE_DESIGN.md", READER_PATH,
        ROOT / "training/src/kernels.cu", ROOT / "training/src/split_search.cu", ROOT / "training/src/batch_resident.cu",
        ROOT / "training/src/initialization.cu", ROOT / "training/src/booster.cpp", ROOT / "training/src/batch_training.inc",
        ROOT / "training/include/ghb/booster.hpp", ROOT / "training/bench/real_data.cpp"]
    all_inputs = [artifact(p) for p in source_paths] + [artifact(DATA / f"{split}.ghb") for split in ("train", "validation")]
    models, measured, trees, predictions = {}, {}, {}, {}
    x, y = fixture(DATA / "train.ghb")
    vx, vy = fixture(DATA / "validation.ghb")
    for case in CASES:
        directory = HERE / "split-campaign" / case
        all_inputs += [artifact(directory / name) for name in ("model.ghb", "metrics.json", "predictions.f64")]
        models[case] = READER.read_model(directory / "model.ghb")
        measured[case] = json.loads((directory / "metrics.json").read_text())
        config = measured[case]["parameters"]
        require(config["rounds"] == 1 and config["depth"] == 3 and config["optimization_order"] == 2
                and config["max_leaf_value"] == 0 and config["min_leaf_rows"] == 1, "unsupported oracle training contract")
        require(models[case]["features"] == [(0, (0.0,))] * 500, "fitted feature contract differs")
        trees[case] = {out: tree for out, tree in models[case]["trees"]}
        require(len(models[case]["trees"]) == len(trees[case]) == 983, "expected exactly one tree per output")
        predictions[case] = np.memmap(directory / "predictions.f64", mode="r", dtype="<f8", shape=(len(vx), 983))
    before, candidate, repeat = CASES
    baseline = models[before]
    base_differences = {case: [i for i, (a, b) in enumerate(zip(baseline["bases"], models[case]["bases"], strict=True))
                              if struct.pack("<d", a) != struct.pack("<d", b)] for case in CASES}
    require(not any(base_differences.values()), "saved base bits differ; this paired oracle requires common initial derivatives")
    for case in CASES[1:]:
        a, b = dict(measured[before]["parameters"]), dict(measured[case]["parameters"])
        a.pop("split_policy"); b.pop("split_policy")
        require(a == b, "non-policy training parameters differ")
    changes = {case: [{"output": i, "first_differing_path": first_difference(trees[before][i], trees[case][i])}
                    for i in range(983) if first_difference(trees[before][i], trees[case][i]) is not None] for case in CASES[1:]}
    root_first = next(c["output"] for c in changes[candidate] if c["first_differing_path"] == "")
    other_first = next(c["output"] for c in changes[candidate] if c["output"] not in (573, root_first))
    selected = [573, root_first, other_first]
    require(selected == [573, 87, 8], "predeclared bounded selection changed")
    report = {"kind": "independent CPU integer-count/high-precision saved-model diagnostic", "status": "completed",
        "source_and_artifact_identities": all_inputs, "selected_outputs": selected,
        "base_score_bit_differences": base_differences, "first_decision_differences_all_outputs": changes,
        "configuration": measured[before]["parameters"],
        "minimum_gain_contract": "zero, from TrainConfig default and absence of any real_data CLI override; not emitted in metrics",
        "oracle_precision_digits": [80, 120], "comparisons": [], "leaf_cross_checks": {},
        "maximum_difference_row_trace": {},
        "limitations": ["There are no recorded GPU gradients or histogram sums for these training runs.",
                        "Ideal sigmoid/count scores are not the exact inputs consumed by GPU split search.",
                        "The exact-rational auxiliary reference uses host-binary64 derivatives, not asserted CUDA math results.",
                        "Count-derived statistics would change floating-point summation; no production replacement or exact-preservation claim follows.",
                        "All previous zero-allowance quality failures remain failed."]}
    for target in selected:
        for case in CASES[1:]:
            path = first_difference(trees[before][target], trees[case][target])
            if path is None:
                report["comparisons"].append({"reference": before, "comparison": case, "output": target, "no_decision_difference": True})
            else:
                detail = node_oracle(x, y, target, path, (trees[before][target], trees[case][target]),
                    (models[before]["bases"][target], models[case]["bases"][target]), measured[before]["parameters"])
                report["comparisons"].append({"reference": before, "comparison": case, **detail})
        for case in CASES:
            report["leaf_cross_checks"][f"{case}/output{target}"] = leaf_checks(x, y, target, trees[case][target], models[case]["bases"][target], measured[case]["parameters"])
    for case in CASES:
        report["maximum_difference_row_trace"][case] = trace_prediction(vx, trees[case][573], models[case]["bases"][573], float(predictions[case][526, 573]), 526)
    for entry in all_inputs:
        require(digest(entry["path"]) == entry["sha256"], "input changed during diagnostic: " + entry["path"])
    report["inputs_unchanged_during_diagnostic"] = True
    with (output / "findings.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# Bounded saved-model divergence diagnostic", "", "CPU validation only; no GPU execution, production changes or performance ranking.", "",
        "The three runs have bitwise-identical base margins and identical binary feature quantization. All previous strict quality failures remain failed.", "",
        "| Output | Compared run | First differing node | Saved features | Ideal gain | Gap to different count signature | Saved choices share ideal-best counts |", "|---:|---|---|---|---:|---:|---|"]
    for c in report["comparisons"]:
        if c.get("no_decision_difference"):
            continue
        s = c["saved_selections"]
        lines.append(f"| {c['output']} | {c['comparison']} | {c['path'] or 'root'} | {s[0]['decision'][0]} / {s[1]['decision'][0]} | {float(c['best']['gain_80_digits']):.12g} | {float(c['margin_to_best_distinct_count_signature']):.12g} | {all(q['same_unordered_child_count_signature_as_best'] for q in s)} |")
    lines += ["", "For validation row526/output573:", "", "| Run | Leaf path | Leaf increment | Saved probability | CPU probability minus saved |", "|---|---|---:|---:|---:|"]
    for case, trace in report["maximum_difference_row_trace"].items():
        lines.append(f"| {case} | {trace['leaf_path']} | {trace['leaf_increment']:.17g} | {trace['saved_gpu_probability']:.17g} | {trace['host_probability_minus_saved']:.4g} |")
    max_leaf_error = max(value["maximum_absolute_difference"] for value in report["leaf_cross_checks"].values())
    lines += ["", f"All saved leaf values in these three outputs were independently checked from their actual training memberships; maximum absolute difference from the 80-digit count-based reference is {max_leaf_error:.17g}. This is an observed discrepancy, not a new correctness tolerance.", "",
        "Full candidate rankings, exact integer counts, selected partition membership hashes, 80/120-digit scores, exact-rational host-derivative checks, base bits, source hashes and prediction paths are retained in `findings.json`.", "",
        "No GPU histogram snapshot exists. The oracle identifies mathematical ties and score separations; it does not prove the device's actual FP64 sums or blame a particular reduction order. Count-derived first-round statistics are an optional numerical design change, not an exact-preserving optimization.", "",
        "The independent regularized score formula follows [Chen and Guestrin, section2.2](https://arxiv.org/html/1603.02754v3#S2.SS2) and the local documented leaf objective."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "selected_outputs": selected,
                      "node_comparisons": sum(not c.get("no_decision_difference", False) for c in report["comparisons"]),
                      "maximum_saved_leaf_difference_from_ideal": max_leaf_error, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        run(args.output)
    except Exception as error:
        with (args.output / "failure.json").open("x") as stream:
            json.dump({"status": "failed", "error": type(error).__name__, "message": str(error)}, stream, indent=2)
            stream.write("\n")
        raise
