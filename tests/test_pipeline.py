import unittest

import numpy as np
import pandas as pd

from src.nonlinear import _new_model, _tree_features
from src.explainability import _reason


class TestNonlinearPipeline(unittest.TestCase):
    def test_local_explanation_distinguishes_zero_effect_from_direction(self):
        self.assertIn("unchanged", _reason("APP_INCOME_TOTAL", 0.0))
        self.assertIn("raises", _reason("APP_INCOME_TOTAL", 0.01))
        self.assertIn("lowers", _reason("APP_INCOME_TOTAL", -0.01))

    def test_tree_inputs_preserve_nan_and_convert_employment_sentinel(self):
        frame = pd.DataFrame({"APP_DAYS_EMPLOYED": [365243, -100, np.nan],
                              "BUREAU_DEBT_RATIO": [np.nan, 0.2, 0.4]})
        result = _tree_features(frame, list(frame.columns))
        self.assertTrue(np.isnan(result.APP_DAYS_EMPLOYED.iloc[0]))
        self.assertTrue(np.isnan(result.APP_DAYS_EMPLOYED.iloc[2]))
        self.assertTrue(np.isnan(result.BUREAU_DEBT_RATIO.iloc[0]))
        self.assertTrue(all(dtype == np.dtype("float64") for dtype in result.dtypes))

    def test_hist_gradient_boosting_accepts_missing_values_and_predicts_probabilities(self):
        rng = np.random.default_rng(73)
        X = pd.DataFrame(rng.normal(size=(240, 3)), columns=["x1", "x2", "x3"])
        y = (X.x1 + .4 * X.x2 > 0).astype(int)
        X.loc[:12, "x2"] = np.nan
        model = _new_model({"learning_rate": .1, "max_iter": 8, "max_leaf_nodes": 7,
                            "min_samples_leaf": 10, "l2_regularization": 1.0})
        model.fit(X.iloc[:180], y.iloc[:180])
        p = model.predict_proba(X.iloc[180:])[:, 1]
        self.assertEqual(p.shape, (60,))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
