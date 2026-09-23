"""CPU-only validity checks for benchmark input and independent quality evidence."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from fixtures import write, load
from evaluate import metrics

class BenchmarkReferences(unittest.TestCase):
    def test_roundtrip_and_trailing_payload_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.ghb"
            x = np.asarray([[1, np.nan], [0, 3]], dtype=np.float32)
            y = np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
            write(p, x, y, 1, 2)
            actual_x, actual_y, header = load(p)
            np.testing.assert_equal(actual_x, x)
            np.testing.assert_equal(actual_y, y)
            self.assertEqual(header["targets"], 3)
            with p.open("ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                load(p)

    def test_binary_reference_and_invalid_probability(self):
        y = np.asarray([[0.], [1.]])
        result = metrics(y, np.asarray([[.25], [.75]]), 1)
        self.assertAlmostEqual(result["log_loss"], -np.log(.75))
        self.assertEqual(result["roc_auc"], 1)
        self.assertEqual(result["f1"], 1)
        self.assertEqual(result["brier"], .0625)
        with self.assertRaises(ValueError):
            metrics(y, np.asarray([[-.1], [.75]]), 1)

    def test_multiclass_normalization_and_targets(self):
        y = np.asarray([[0.], [1.], [2.]])
        result = metrics(y, np.eye(3), 2)
        self.assertEqual(result["log_loss"], 0)
        self.assertEqual(result["macro_f1"], 1)
        with self.assertRaises(ValueError):
            metrics(y, np.eye(3) * .5, 2)

    def test_multilabel_all_labels_and_tie_order(self):
        y = np.asarray([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=float)
        p = .1 + .8 * y
        result = metrics(y, p, 1)
        self.assertEqual(result["macro_ap_labels"], 3)
        self.assertEqual(result["micro_f1"], 1)
        self.assertEqual(result["hamming_loss"], 0)
        self.assertEqual(result["precision_at_1"], 1)
        self.assertAlmostEqual(result["precision_at_3"], 2 / 3)
        self.assertAlmostEqual(result["precision_at_5"], .4)

if __name__ == "__main__":
    unittest.main()
