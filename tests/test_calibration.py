import unittest

import numpy as np

from src.calibration import apply_calibrator, calibration_metrics, fit_calibrator


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


if __name__ == "__main__":
    unittest.main()
