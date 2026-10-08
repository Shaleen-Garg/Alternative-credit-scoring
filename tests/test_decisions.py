import unittest

import numpy as np

from src.decision_analysis import _select_cost_threshold, decision_metrics
from src.stability_analysis import _psi


class TestDecisionAndStabilityDiagnostics(unittest.TestCase):
    def test_expected_cost_and_decision_counts(self):
        y = np.array([1, 1, 0, 0, 1])
        p = np.array([.8, .2, .7, .1, .3])
        result = decision_metrics(y, p, .5, fn_cost=5, fp_cost=1)
        self.assertEqual((result["TP"], result["TN"], result["FP"], result["FN"]), (1, 1, 1, 2))
        self.assertEqual(result["expected_cost"], 11)
        self.assertAlmostEqual(result["normalized_expected_cost"], 2.2)
        self.assertAlmostEqual(result["approval_rate"], .6)
        self.assertAlmostEqual(result["default_rate_among_approved"], 2 / 3)

    def test_cost_search_includes_thresholds_above_one_half(self):
        result = _select_cost_threshold(
            np.array([0, 0, 1, 1]), np.array([.2, .51, .52, .9]), fn_cost=1,
        )
        self.assertEqual(result["threshold"], .52)
        self.assertEqual(result["normalized_expected_cost"], 0)

    def test_decision_metrics_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            decision_metrics([0, 1], [.2], .5)
        with self.assertRaises(ValueError):
            decision_metrics([0, 2], [.2, .8], .5)
        with self.assertRaises(ValueError):
            decision_metrics([0, 1], [.2, 1.2], .5)

    def test_psi_is_zero_for_identical_distribution_and_positive_when_shifted(self):
        train = np.arange(100, dtype=float)
        self.assertAlmostEqual(_psi(train, train), 0.0)
        self.assertGreater(_psi(train, train + 100), 0.01)


if __name__ == "__main__":
    unittest.main()
