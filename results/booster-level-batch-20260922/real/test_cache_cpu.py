"""Tiny CPU fixtures prove cached results equal the untouched full evaluator."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate import evaluate
from evaluate_cached import canonical, evaluate_cached
from fixtures import write

class ExactBaselineCache(unittest.TestCase):
    def test_all_objectives_miss_hit_and_candidate_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = np.arange(12, dtype=np.float32).reshape(6, 2)
            for name, objective, classes, y, first, second in [
                ("reg", 0, 0, np.arange(6)[:, None], np.arange(6)[:, None] + .25, np.arange(6)[:, None] - .5),
                ("binary", 1, 2, np.asarray([0, 1, 0, 1, 0, 1])[:, None], np.asarray([.2, .8, .3, .7, .4, .6])[:, None], np.full((6, 1), .5)),
                ("multiclass", 2, 3, np.asarray([0, 1, 2, 0, 1, 2])[:, None], np.eye(3)[[0, 1, 2, 0, 1, 2]] * .7 + .1, np.full((6, 3), 1 / 3)),
                ("multilabel", 1, 2, np.asarray([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0]] * 2),
                 np.asarray([[.8, .2, .8, .1, .1], [.2, .8, .8, .1, .1], [.8, .8, .2, .1, .1]] * 2), np.full((6, 5), .5)),
            ]:
                directory = root / name
                directory.mkdir()
                train, evaluation = directory / "train.ghb", directory / "test.ghb"
                write(train, x, y, objective, classes)
                write(evaluation, x + 1, y, objective, classes)
                receipts = []
                for index, prediction in enumerate((first, second)):
                    p = directory / f"p{index}.f64"
                    np.asarray(prediction, dtype="<f8").tofile(p)
                    expected = evaluate(train, evaluation, p)
                    actual, receipt = evaluate_cached(train, evaluation, p, directory / f"q{index}.json", directory / "cache", directory / f"proof{index}.json")
                    self.assertEqual(canonical(expected), canonical(actual))
                    self.assertEqual(receipt["cache_hit"], index == 1)
                    receipts.append(receipt)
                self.assertEqual(receipts[0]["cache_key"], receipts[1]["cache_key"])
                self.assertEqual(receipts[0]["cache_sha256"], receipts[1]["cache_sha256"])
                # Reusing the same cache with another evaluation payload must
                # create another key, even when targets and shape are unchanged.
                other = directory / "other.ghb"
                write(other, x + 2, y, objective, classes)
                _, receipt = evaluate_cached(train, other, p, directory / "other.json", directory / "cache", directory / "other-proof.json")
                self.assertFalse(receipt["cache_hit"])
                self.assertNotEqual(receipt["cache_key"], receipts[0]["cache_key"])

if __name__ == "__main__":
    unittest.main()
