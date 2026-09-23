#!/usr/bin/env python3
"""Independent CPU examples and rejection checks for saved prediction metrics."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "tools/evaluate.py"
spec = importlib.util.spec_from_file_location("training_evaluation_under_test", MODULE)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def score(self, targets, predictions, objective="regression", classes=None, tolerance=1e-6,
              dataset="fixture", split="held_out"):
        self.serial += 1
        target_path = self.root / f"targets-{self.serial}.csv"
        prediction_path = self.root / f"predictions-{self.serial}.csv"
        target_path.write_text(targets)
        prediction_path.write_text(predictions)
        return evaluate.evaluate(target_path, prediction_path, objective, dataset, split, classes, tolerance)

    def save(self, report, name):
        path = self.root / name
        evaluate.write_new(path, report)
        return path

    def value(self, report, name):
        return report["metrics"][name]["value"]

    def test_known_weighted_regression(self):
        result = self.score("row_id,target,weight\na,0,1\nb,4,3\n", "row_id,prediction\na,1\nb,1\n")
        self.assertAlmostEqual(self.value(result, "rmse"), math.sqrt(7))
        self.assertAlmostEqual(self.value(result, "mae"), 2.5)
        self.assertEqual(result["total_weight"], 4)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["outputs"], 1)
        self.assertEqual(result["metrics"]["rmse"]["direction"], "minimize")
        self.assertEqual(self.value(result, "rmse"), self.value(result, "rmse_output_0"))

    def test_multioutput_overall_and_each_output(self):
        result = self.score("row_id,target_0,target_1,weight\na,0,0,1\nb,2,2,3\n",
                            "row_id,prediction_0,prediction_1\na,1,4\nb,3,2\n")
        self.assertEqual(result["outputs"], 2)
        self.assertAlmostEqual(self.value(result, "rmse_output_0"), 1)
        self.assertAlmostEqual(self.value(result, "rmse_output_1"), 2)
        self.assertAlmostEqual(self.value(result, "mae_output_0"), 1)
        self.assertAlmostEqual(self.value(result, "mae_output_1"), 1)
        self.assertAlmostEqual(self.value(result, "rmse"), math.sqrt(2.5))
        self.assertAlmostEqual(self.value(result, "mae"), 1)

    def test_multilabel_weighted_metrics_and_independent_probability_sums(self):
        result = self.score("row_id,target_0,target_1,weight\na,0,1,1\nb,1,0,3\n",
                            "row_id,prediction_0,prediction_1\na,0.1,0.8\nb,0.7,0.4\n", "multilabel")
        self.assertEqual(result["outputs"], 2)
        self.assertEqual(result["classes"], 2)
        first_loss = (-math.log(.9) - 3 * math.log(.7)) / 4
        second_loss = (-math.log(.8) - 3 * math.log(.6)) / 4
        self.assertAlmostEqual(self.value(result, "logloss_output_0"), first_loss)
        self.assertAlmostEqual(self.value(result, "logloss_output_1"), second_loss)
        self.assertAlmostEqual(self.value(result, "logloss"), (first_loss + second_loss) / 2)
        self.assertAlmostEqual(self.value(result, "brier"), (.01 + .04 + 3 * (.09 + .16)) / 8)
        self.assertEqual(self.value(result, "accuracy"), 1)
        self.assertEqual(self.value(result, "auc"), 1)
        self.assertEqual(result["target_summary"]["class_weights"], [[1, 3], [3, 1]])

    def test_multilabel_undefined_auc_is_explicit_and_not_dropped(self):
        result = self.score("row_id,target_0,target_1\na,0,1\nb,1,1\n",
                            "row_id,prediction_0,prediction_1\na,0.1,0.8\nb,0.7,0.4\n", "multilabel")
        self.assertEqual(self.value(result, "auc_output_0"), 1)
        self.assertIsNone(self.value(result, "auc_output_1"))
        self.assertIsNone(self.value(result, "auc"))
        path = self.save(result, "multilabel.json")
        compared = evaluate.compare(path, path)
        self.assertEqual(compared["status"], "pass")
        self.assertEqual(compared["metrics"]["auc_output_1"]["status"], "not_applicable")

    def test_multilabel_subnormal_brier_survives_output_averaging(self):
        smallest = float("5e-324")
        probability = math.sqrt(smallest)
        result = self.score("row_id,target_0,target_1\na,0,0\n",
                            f"row_id,prediction_0,prediction_1\na,{probability},{probability}\n", "multilabel")
        self.assertEqual(self.value(result, "brier_output_0"), smallest)
        self.assertEqual(self.value(result, "brier_output_1"), smallest)
        self.assertEqual(self.value(result, "brier"), smallest)

    def test_multilabel_single_output_matches_binary(self):
        targets = "row_id,target,weight\na,0,1\nb,1,3\nc,0,2\n"
        predictions = "row_id,prediction\na,0.1\nb,0.5\nc,0.5\n"
        binary = self.score(targets, predictions, "binary")
        multilabel = self.score(targets, predictions, "multilabel")
        for metric in ("logloss", "accuracy", "brier", "auc"):
            self.assertEqual(self.value(multilabel, metric), self.value(binary, metric))
            self.assertEqual(self.value(multilabel, metric + "_output_0"), self.value(binary, metric))

    def test_multilabel_defined_auc_checked_when_aggregate_is_undefined(self):
        targets = "row_id,target_0,target_1\na,0,1\nb,1,1\n"
        before = self.save(self.score(targets, "row_id,prediction_0,prediction_1\na,0.1,0.8\nb,0.9,0.8\n", "multilabel"), "defined-before.json")
        after = self.save(self.score(targets, "row_id,prediction_0,prediction_1\na,0.9,0.8\nb,0.1,0.8\n", "multilabel"), "defined-after.json")
        compared = evaluate.compare(before, after)
        self.assertEqual(compared["metrics"]["auc"]["status"], "not_applicable")
        self.assertIn("auc_output_0", compared["regressions"])
        self.assertEqual(compared["status"], "regression")

    def test_multilabel_auc_uses_weighted_class_presence_per_output(self):
        result = self.score("row_id,target_0,target_1,weight\na,0,0,0\nb,0,1,1\nc,1,1,2\n",
                            "row_id,prediction_0,prediction_1\na,0.1,0.1\nb,0.2,0.8\nc,0.9,0.9\n", "multilabel")
        self.assertEqual(result["target_summary"]["class_sample_counts"][1], [1, 2])
        self.assertEqual(result["target_summary"]["class_weights"][1], [0, 3])
        self.assertEqual(self.value(result, "auc_output_0"), 1)
        self.assertIsNone(self.value(result, "auc_output_1"))
        self.assertIsNone(self.value(result, "auc"))

    def test_multilabel_rejects_bad_labels_probabilities_and_shape(self):
        good_targets = "row_id,target_0,target_1\na,0,1\n"
        good_predictions = "row_id,prediction_0,prediction_1\na,0.1,0.8\n"
        for targets, predictions, classes in (
                (good_targets.replace(",0,1", ",0,2"), good_predictions, None),
                (good_targets.replace(",0,1", ",0.5,1"), good_predictions, None),
                (good_targets, good_predictions.replace("0.8", "1.1"), None),
                (good_targets, good_predictions.replace("prediction_1", "prediction_2"), None),
                (good_targets, "row_id,prediction_0\na,0.1\n", None),
                (good_targets, "row_id,prediction_0,prediction_1,prediction_2\na,0.1,0.8,0.4\n", None),
                (good_targets, good_predictions, 3)):
            with self.subTest(targets=targets, predictions=predictions, classes=classes):
                with self.assertRaises(evaluate.EvaluationError):
                    self.score(targets, predictions, "multilabel", classes)

    def test_multilabel_comparison_checks_each_output_not_just_average(self):
        targets = "row_id,target_0,target_1\na,0,1\nb,1,0\n"
        before = self.save(self.score(targets, "row_id,prediction_0,prediction_1\na,0.2,0.6\nb,0.8,0.4\n", "multilabel"), "ml-before.json")
        after = self.save(self.score(targets, "row_id,prediction_0,prediction_1\na,0.3,0.9\nb,0.7,0.1\n", "multilabel"), "ml-after.json")
        compared = evaluate.compare(before, after)
        self.assertEqual(compared["status"], "regression")
        self.assertIn("logloss_output_0", compared["regressions"])
        self.assertNotIn("logloss", compared["regressions"])

    def test_multilabel_report_rejects_tampering_and_malformed_class_summary(self):
        result = self.score("row_id,target_0,target_1\na,0,1\nb,1,0\n",
                            "row_id,prediction_0,prediction_1\na,0.1,0.8\nb,0.7,0.4\n", "multilabel")
        for mutate in (lambda r: r["target_summary"]["class_weights"][0].pop(),
                       lambda r: r["target_summary"]["class_sample_counts"][1].__setitem__(0, 5),
                       lambda r: r["metrics"]["logloss_output_1"].update(value=0.1),
                       lambda r: r["metrics"]["auc_output_1"].update(value=None),
                       lambda r: r["metrics"].pop("auc_output_1"),
                       lambda r: r["metrics"].update(auc_output_2=r["metrics"]["auc_output_1"])):
            changed = copy.deepcopy(result); mutate(changed)
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.validate_report(changed)

    def test_extreme_finite_regression_avoids_squared_overflow(self):
        result = self.score("row_id,target\na,0\n", "row_id,prediction\na,1e200\n")
        self.assertEqual(self.value(result, "rmse"), 1e200)
        self.assertEqual(self.value(result, "mae"), 1e200)

    def test_extreme_weight_ratio_does_not_erase_representable_error(self):
        result = self.score("row_id,target,weight\na,0,5e-324\nb,0,1e308\n",
                            "row_id,prediction\na,1e308\nb,0\n")
        expected = math.sqrt(float("5e-324")) * math.sqrt(1e308)
        self.assertAlmostEqual(self.value(result, "rmse") / expected, 1.0)
        self.assertEqual(self.value(result, "mae"), float("5e-324"))

    def test_subnormal_equal_errors_survive_averaging_rows_and_outputs(self):
        result = self.score("row_id,target_0,target_1\na,0,0\nb,0,0\n",
                            "row_id,prediction_0,prediction_1\na,5e-324,5e-324\nb,5e-324,5e-324\n")
        self.assertEqual(self.value(result, "rmse"), float("5e-324"))
        self.assertEqual(self.value(result, "mae"), float("5e-324"))

    def test_zero_weight_errors_do_not_contribute(self):
        result = self.score("row_id,target,weight\na,1e308,0\nb,1,2\n",
                            "row_id,prediction\na,-1e308\nb,1\n")
        self.assertEqual(self.value(result, "rmse"), 0)
        self.assertEqual(result["positive_weight_count"], 1)

    def test_unrepresentable_positive_weight_error_rejected(self):
        with self.assertRaises(evaluate.EvaluationError):
            self.score("row_id,target\na,1e308\n", "row_id,prediction\na,-1e308\n")

    def test_known_weighted_binary_metrics(self):
        result = self.score("row_id,target,weight\na,0,1\nb,1,2\nc,1,1\n",
                            "row_id,prediction\na,0.1\nb,0.7\nc,0.5\n", "binary")
        self.assertAlmostEqual(self.value(result, "logloss"), (-math.log(.9) - 2 * math.log(.7) - math.log(.5)) / 4)
        self.assertAlmostEqual(self.value(result, "brier"), .11)
        self.assertEqual(self.value(result, "accuracy"), 1)
        self.assertEqual(self.value(result, "auc"), 1)

    def test_auc_weighted_ties(self):
        result = self.score("row_id,target,weight\na,0,1\nb,1,2\nc,0,3\nd,1,4\n",
                            "row_id,prediction\na,0.1\nb,0.5\nc,0.5\nd,0.9\n", "binary")
        self.assertAlmostEqual(self.value(result, "auc"), 21 / 24)
        self.assertAlmostEqual(self.value(result, "accuracy"), .7)

    def test_auc_all_tied_is_half_and_score_order_matters(self):
        labels = "row_id,target\na,0\nb,1\n"
        for predictions, expected in (("a,0.5\nb,0.5\n", .5), ("a,0.8\nb,0.2\n", 0), ("a,0.2\nb,0.8\n", 1)):
            with self.subTest(expected=expected):
                result = self.score(labels, "row_id,prediction\n" + predictions, "binary")
                self.assertEqual(self.value(result, "auc"), expected)

    def test_auc_uses_positive_weight_classes(self):
        result = self.score("row_id,target,weight\na,0,0\nb,1,2\n", "row_id,prediction\na,0.1\nb,0.9\n", "binary")
        self.assertIsNone(self.value(result, "auc"))
        self.assertEqual(result["target_summary"]["class_sample_counts"], [1, 1])
        self.assertEqual(result["target_summary"]["class_weights"], [0, 2])

    def test_auc_extreme_weights_do_not_overflow_pair_products(self):
        result = self.score("row_id,target,weight\na,0,1e308\nb,1,5e-324\n",
                            "row_id,prediction\na,0.1\nb,0.9\n", "binary")
        self.assertEqual(self.value(result, "auc"), 1)

    def test_logloss_endpoint_clip_is_explicit(self):
        result = self.score("row_id,target\na,1\nb,0\n", "row_id,prediction\na,0\nb,1\n", "binary")
        self.assertAlmostEqual(self.value(result, "logloss"), -math.log(1e-15))
        self.assertEqual(result["settings"]["log_clip_lower"], 1e-15)
        self.assertEqual(result["settings"]["log_clip_upper"], 1 - 1e-15)
        self.assertEqual(self.value(result, "brier"), 1)

    def test_known_multiclass_metrics(self):
        result = self.score("row_id,target,weight\na,0,1\nb,2,3\n",
                            "row_id,p0,p1,p2\na,0.5,0.25,0.25\nb,0.1,0.2,0.7\n", "multiclass", 3)
        self.assertAlmostEqual(self.value(result, "logloss"), (-math.log(.5) - 3 * math.log(.7)) / 4)
        self.assertEqual(self.value(result, "accuracy"), 1)
        self.assertEqual(result["outputs"], 3)

    def test_multiclass_fp32_rounding_normalized_after_tolerance_check(self):
        result = self.score("row_id,target\na,2\n", "row_id,p0,p1,p2\na,0.33333334,0.33333334,0.33333334\n", "multiclass", 3)
        self.assertAlmostEqual(self.value(result, "logloss"), math.log(3))
        self.assertEqual(self.value(result, "accuracy"), 0)
        self.assertEqual(result["settings"]["probability_sum_tolerance"], 1e-6)
        self.assertEqual(result["settings"]["probability_normalization"], "divide_by_row_sum_after_tolerance_check")

    def test_multiclass_tolerance_boundary_is_inclusive(self):
        self.score("row_id,target\na,0\n", "row_id,p0,p1\na,0.5,0.375\n", "multiclass", 2, .125)
        with self.assertRaises(evaluate.EvaluationError):
            self.score("row_id,target\na,0\n", "row_id,p0,p1\na,0.5,0.374\n", "multiclass", 2, .125)

    def test_multiclass_rejects_bad_vectors(self):
        for row in ("0.5,0.2,0.2", "0,0,0", "-0.1,0.6,0.5", "1.1,0,0", "0.3,NaN,0.7"):
            with self.subTest(row=row), self.assertRaises(evaluate.EvaluationError):
                self.score("row_id,target\na,0\n", "row_id,p0,p1,p2\na," + row + "\n", "multiclass", 3)

    def test_class_labels_and_probabilities_rejected_outside_contract(self):
        for objective, classes, label, header, row in (
                ("binary", None, "2", "prediction", "0.5"),
                ("binary", None, "0.5", "prediction", "0.5"),
                ("binary", None, "0", "prediction", "1.001"),
                ("multiclass", 2, "2", "p0,p1", "0.5,0.5"),
                ("multiclass", 2, "-1", "p0,p1", "0.5,0.5")):
            with self.subTest(objective=objective, label=label, row=row), self.assertRaises(evaluate.EvaluationError):
                self.score(f"row_id,target\na,{label}\n", f"row_id,{header}\na,{row}\n", objective, classes)

    def test_invalid_objective_class_settings(self):
        for objective, classes, tolerance in (("other", None, 1e-6), ("regression", 2, 1e-6),
                                             ("binary", 3, 1e-6), ("multiclass", None, 1e-6),
                                             ("binary", None, -1), ("binary", None, float("nan"))):
            with self.subTest(objective=objective, classes=classes), self.assertRaises(evaluate.EvaluationError):
                self.score("row_id,target\na,0\n", "row_id,prediction\na,0\n", objective, classes, tolerance)

    def test_row_id_integrity(self):
        for target, prediction in (("a,0\na,1\n", "a,0\nb,1\n"),
                                   ("a,0\nb,1\n", "b,1\na,0\n"),
                                   ("a,0\nb,1\n", "a,0\n"),
                                   (",0\n", ",0\n"),
                                   ("a,0\nb,1\n", "a,0\na,1\n")):
            with self.subTest(target=target, prediction=prediction), self.assertRaises(evaluate.EvaluationError):
                self.score("row_id,target\n" + target, "row_id,prediction\n" + prediction)

    def test_reject_empty_malformed_and_noncontiguous_columns(self):
        for targets, predictions in (("row_id,target\n", "row_id,prediction\n"),
                                    ("row_id,target\na,0\n\n", "row_id,prediction\na,0\n"),
                                    ("row_id,target_0,target_2\na,0,0\n", "row_id,prediction_0,prediction_1\na,0,0\n"),
                                    ("row_id,target,target\na,0,0\n", "row_id,prediction\na,0\n"),
                                    ("row_id,target_0,target_1\na,0,0\n", "row_id,prediction\na,0\n")):
            with self.subTest(targets=targets), self.assertRaises(evaluate.EvaluationError):
                self.score(targets, predictions)

    def test_reject_zero_negative_and_overflowing_weights(self):
        for weights in (("0", "0"), ("1", "-1"), ("1e308", "1e308"), ("NaN", "1")):
            with self.subTest(weights=weights), self.assertRaises(evaluate.EvaluationError):
                self.score(f"row_id,target,weight\na,0,{weights[0]}\nb,1,{weights[1]}\n", "row_id,prediction\na,0\nb,1\n")

    def test_reject_nonfinite_targets_and_predictions(self):
        for value in ("NaN", "inf", "-inf", "1e309"):
            for field in ("target", "prediction"):
                with self.subTest(value=value, field=field), self.assertRaises(evaluate.EvaluationError):
                    self.score("row_id,target\na," + (value if field == "target" else "0") + "\n",
                               "row_id,prediction\na," + (value if field == "prediction" else "0") + "\n")

    def test_inputs_unchanged_and_overwrite_forbidden(self):
        report = self.score("row_id,target\na,0\n", "row_id,prediction\na,1\n")
        snapshots = {name: Path(value["path"]).read_bytes() for name, value in report["inputs"].items()}
        output = self.save(report, "report.json")
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            evaluate.write_new(output, {"bad": "overwrite"})
        with self.assertRaises(FileExistsError):
            evaluate.write_new(Path(report["inputs"]["targets"]["path"]), report)
        self.assertEqual(output.read_bytes(), original)
        for name, artifact in report["inputs"].items():
            self.assertEqual(Path(artifact["path"]).read_bytes(), snapshots[name])
            self.assertEqual(evaluate.digest(snapshots[name]), artifact["sha256"])

    def test_compare_all_regression_outputs_even_if_overall_improves(self):
        targets = "row_id,target_0,target_1\na,0,0\n"
        before = self.score(targets, "row_id,prediction_0,prediction_1\na,0,4\n")
        after = self.score(targets, "row_id,prediction_0,prediction_1\na,1,0\n")
        result = evaluate.compare(self.save(before, "before.json"), self.save(after, "after.json"))
        self.assertEqual(result["status"], "regression")
        self.assertEqual(set(result["regressions"]), {"rmse_output_0", "mae_output_0"})
        self.assertEqual(result["metrics"]["rmse"]["status"], "pass")

    def test_compare_allowance_boundaries_and_default_zero(self):
        targets = "row_id,target\na,0\n"
        before = self.save(self.score(targets, "row_id,prediction\na,0\n"), "before.json")
        after = self.save(self.score(targets, "row_id,prediction\na,0.1\n"), "after.json")
        self.assertEqual(evaluate.compare(before, after)["status"], "regression")
        self.assertEqual(evaluate.compare(before, after, .1)["status"], "pass")
        self.assertEqual(evaluate.compare(before, after, math.nextafter(.1, 0))["status"], "regression")
        self.assertEqual(evaluate.compare(before, before)["status"], "pass")

    def test_large_allowance_preserves_exact_recorded_decimal_boundary(self):
        targets = "row_id,target\na,0\n"
        old_value = math.nextafter(2e292, 0)
        new_value = 1e308
        allowance = math.nextafter(new_value, 0)
        before = self.save(self.score(targets, f"row_id,prediction\na,{old_value}\n"), "before.json")
        after = self.save(self.score(targets, f"row_id,prediction\na,{new_value}\n"), "after.json")
        result = evaluate.compare(before, after, allowance)
        self.assertEqual(result["status"], "regression")
        self.assertEqual(set(result["regressions"]), {"rmse", "mae", "rmse_output_0", "mae_output_0"})
        # Default Decimal precision would round this deterioration to allowance.
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 2048
            excess = Decimal(result["metrics"]["rmse"]["deterioration_decimal"]) - Decimal(str(allowance))
        self.assertEqual(excess, Decimal("4e276"))

    def test_compare_respects_maximize_direction(self):
        targets = "row_id,target\na,0\nb,1\n"
        before = self.save(self.score(targets, "row_id,prediction\na,0.1\nb,0.9\n", "binary"), "before.json")
        after = self.save(self.score(targets, "row_id,prediction\na,0.9\nb,0.1\n", "binary"), "after.json")
        result = evaluate.compare(before, after)
        self.assertIn("accuracy", result["regressions"])
        self.assertIn("auc", result["regressions"])
        self.assertEqual(result["metrics"]["accuracy"]["deterioration"], 1)

    def test_maximize_allowance_boundary_is_inclusive(self):
        targets = "row_id,target\n" + "".join(f"{i},1\n" for i in range(10))
        before_predictions = "row_id,prediction\n0,0.51\n" + "".join(f"{i},1\n" for i in range(1, 10))
        after_predictions = before_predictions.replace("0,0.51", "0,0.49")
        before = self.save(self.score(targets, before_predictions, "binary"), "before.json")
        after = self.save(self.score(targets, after_predictions, "binary"), "after.json")
        self.assertEqual(evaluate.compare(before, after, .1)["status"], "pass")
        result = evaluate.compare(before, after, math.nextafter(.1, 0))
        self.assertEqual(result["regressions"], ["accuracy"])

    def test_compare_null_auc_explicitly_not_applicable(self):
        report = self.score("row_id,target\na,1\n", "row_id,prediction\na,0.5\n", "binary")
        path = self.save(report, "single-class.json")
        result = evaluate.compare(path, path)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["metrics"]["auc"]["status"], "not_applicable")

    def test_compare_dataset_split_and_unchanged_target_requirements(self):
        before = self.score("row_id,target\na,0\n", "row_id,prediction\na,0\n")
        before_path = self.save(before, "before.json")
        for change in ("dataset", "split", "target", "weights"):
            with self.subTest(change=change):
                targets = "row_id,target\na,1\n" if change == "target" else "row_id,target,weight\na,0,2\n" if change == "weights" else "row_id,target\na,0\n"
                after = self.score(targets, "row_id,prediction\na,0\n", dataset="different" if change == "dataset" else "fixture", split="different" if change == "split" else "held_out")
                path = self.save(after, change + ".json")
                with self.assertRaises(evaluate.EvaluationError):
                    evaluate.compare(before_path, path)

    def test_compare_settings_must_match(self):
        targets, predictions = "row_id,target\na,0\n", "row_id,p0,p1\na,0.5,0.5\n"
        before = self.save(self.score(targets, predictions, "multiclass", 2, 1e-6), "before.json")
        after = self.save(self.score(targets, predictions, "multiclass", 2, 1e-5), "after.json")
        with self.assertRaises(evaluate.EvaluationError):
            evaluate.compare(before, after)

    def test_report_rejects_missing_metrics_directions_and_numeric_bounds(self):
        report = self.score("row_id,target\na,0\nb,1\n", "row_id,prediction\na,0.2\nb,0.8\n", "binary")
        for mutate in (lambda r: r["metrics"].pop("auc"),
                       lambda r: r["metrics"]["accuracy"].update(direction="minimize"),
                       lambda r: r["metrics"]["accuracy"].update(value=1.1),
                       lambda r: r["metrics"]["accuracy"].update(value=True),
                       lambda r: r["metrics"]["logloss"].update(value=float("nan")),
                       lambda r: r["metrics"]["auc"].update(value=None)):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.validate_report(changed)

    def test_available_input_hash_changes_are_rejected(self):
        report = self.score("row_id,target\na,0\n", "row_id,prediction\na,0\n")
        path = self.save(report, "report.json")
        Path(report["inputs"]["predictions"]["path"]).write_text("row_id,prediction\na,1\n")
        with self.assertRaisesRegex(evaluate.EvaluationError, "hash changed"):
            evaluate.compare(path, path)

    def test_available_artifacts_recompute_and_reject_plausible_fabricated_metric(self):
        report = self.score("row_id,target\na,0\n", "row_id,prediction\na,1\n")
        self.assertTrue(evaluate.validate_report(report)["metrics_recomputed"])
        report["metrics"]["mae"]["value"] = .5
        with self.assertRaisesRegex(evaluate.EvaluationError, "recomputed"):
            evaluate.validate_report(report)

    def test_missing_artifacts_prevent_a_verified_comparison(self):
        report = self.score("row_id,target\na,0\n", "row_id,prediction\na,1\n")
        path = self.save(report, "report.json")
        Path(report["inputs"]["predictions"]["path"]).unlink()
        check = evaluate.validate_report(report)
        self.assertEqual(check["artifact_hashes_verified"], ["targets"])
        self.assertFalse(check["metrics_recomputed"])
        self.assertEqual(check["unavailable_artifacts"], ["predictions"])
        with self.assertRaisesRegex(evaluate.EvaluationError, "requires both source CSVs"):
            evaluate.compare(path, path)

    def test_json_duplicate_keys_and_nonfinite_values_rejected(self):
        for content in ('{"x":1,"x":2}', '{"x":NaN}'):
            path = self.root / "invalid.json"
            path.write_text(content)
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.read_report(path)

    def test_cli_evaluate_compare_and_overwrite_exit_contract(self):
        target_path, prediction_path, output = self.root / "t.csv", self.root / "p.csv", self.root / "evaluation.json"
        target_path.write_text("row_id,target\na,0\n")
        prediction_path.write_text("row_id,prediction\na,0\n")
        args = ["evaluate", "--objective", "regression", "--targets", str(target_path), "--predictions", str(prediction_path),
                "--output", str(output), "--dataset-id", "fixture", "--split-id", "held_out"]
        self.assertEqual(evaluate.main(args), 0)
        with self.assertRaises(FileExistsError):
            evaluate.main(args)
        comparison_output = self.root / "comparison.json"
        self.assertEqual(evaluate.main(["compare", "--reference", str(output), "--candidate", str(output), "--output", str(comparison_output)]), 0)
        self.assertEqual(json.loads(comparison_output.read_text())["status"], "pass")
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(evaluate.main(["compare", "--reference", str(output), "--candidate", str(output)]), 0)
        self.assertEqual(json.loads(capture.getvalue())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
