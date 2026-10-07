import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.calibration import apply_calibrator, calibration_metrics, fit_calibrator
from src.models import FEATURE_GROUPS
from src.phase10 import select_model_for_calibration


class TestCalibration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(202)
        self.y_val = rng.binomial(1, .22, 600)
        raw = rng.uniform(.05, .9, 600)
        self.p_val = np.clip(.15 + .55 * raw + .2 * self.y_val, 1e-4, 1 - 1e-4)
        self.p_test = np.linspace(.03, .95, 80)

    def test_sigmoid_and_isotonic_keep_prediction_shape_and_bounds(self):
        for method in ("sigmoid", "isotonic"):
            calibrator = fit_calibrator(method, self.p_val, self.y_val)
            predicted = apply_calibrator(method, calibrator, self.p_test)
            self.assertEqual(predicted.shape, self.p_test.shape)
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())

    def test_sigmoid_preserves_ranking_when_fitted_slope_is_positive(self):
        calibrator = fit_calibrator("sigmoid", self.p_val, self.y_val)
        self.assertGreater(calibrator.coef_[0, 0], 0)
        calibrated = apply_calibrator("sigmoid", calibrator, self.p_test)
        y = np.tile([0, 1], len(self.p_test) // 2)
        self.assertEqual(roc_auc_score(y, self.p_test), roc_auc_score(y, calibrated))
        self.assertEqual(average_precision_score(y, self.p_test), average_precision_score(y, calibrated))

    def test_calibration_diagnostics_report_bins_and_proper_metrics(self):
        y = np.tile([0, 1], 50)
        p = np.linspace(.01, .99, len(y))
        metrics, bins = calibration_metrics(y, p, n_bins=10)
        self.assertEqual(metrics["n"], len(y))
        self.assertIn("brier", metrics)
        self.assertIn("log_loss", metrics)
        self.assertIn("calibration_intercept", metrics)
        self.assertIn("calibration_slope", metrics)
        self.assertIn("ece_equal_width", metrics)
        self.assertEqual(int(bins.n.sum()), len(y))

    def test_phase10_model_selection_is_invariant_to_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp) / "tables"
            tables.mkdir()

            def write_metrics(test_value):
                split_rows, cohort_rows = [], []
                for model_idx, model_name in enumerate(FEATURE_GROUPS):
                    for split, value in (("train", .74 + model_idx * .01),
                                         ("validation", .72 + model_idx * .01),
                                         ("test", test_value)):
                        row = {"split": split, "model": model_name,
                               "roc_auc": value, "pr_auc": value / 3}
                        split_rows.append(row)
                        cohort_rows.append({"split": split, "cohort": "Thin-file", **row})
                pd.DataFrame(split_rows).to_csv(tables / "phase8_split_metrics.csv", index=False)
                pd.DataFrame(split_rows).to_csv(tables / "phase9_nonlinear_metrics.csv", index=False)
                pd.DataFrame(cohort_rows).to_csv(tables / "phase8_cohort_metrics.csv", index=False)
                pd.DataFrame(cohort_rows).to_csv(tables / "phase9_nonlinear_cohort_metrics.csv", index=False)

            write_metrics(.999)
            first = select_model_for_calibration(tmp)
            write_metrics(.001)
            second = select_model_for_calibration(tmp)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
