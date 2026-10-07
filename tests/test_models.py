import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import (
    APPLICATION_FEATURES, BUREAU_FEATURES, ALTERNATIVE_FEATURES,
    FEATURE_GROUPS, load_and_split_data, build_pipeline,
    thin_file_mask,
)


class TestFeatureGroupsAndModels(unittest.TestCase):
    def test_feature_groups_are_exact_and_disjoint(self):
        self.assertEqual(len(APPLICATION_FEATURES), 10)
        self.assertEqual(len(BUREAU_FEATURES), 6)
        self.assertEqual(len(ALTERNATIVE_FEATURES), 7)
        self.assertEqual(len(set(APPLICATION_FEATURES + BUREAU_FEATURES + ALTERNATIVE_FEATURES)), 23)
        self.assertEqual(FEATURE_GROUPS["Application"], APPLICATION_FEATURES)
        self.assertEqual(FEATURE_GROUPS["Application + Bureau"], APPLICATION_FEATURES + BUREAU_FEATURES)
        self.assertEqual(FEATURE_GROUPS["Application + Bureau + Alternative"],
                         APPLICATION_FEATURES + BUREAU_FEATURES + ALTERNATIVE_FEATURES)
        for features in FEATURE_GROUPS.values():
            self.assertNotIn("TARGET", features)
            self.assertNotIn("SK_ID_CURR", features)

    def _dataset(self, n=600):
        rng = np.random.default_rng(17)
        frame = pd.DataFrame({"SK_ID_CURR": np.arange(n), "TARGET": np.tile([0, 1], n // 2)})
        for feature in APPLICATION_FEATURES + BUREAU_FEATURES + ALTERNATIVE_FEATURES:
            frame[feature] = rng.normal(size=n)
        frame["BUREAU_CREDIT_COUNT"] = rng.integers(0, 8, n)
        frame["INST_TOTAL_COUNT"] = rng.integers(0, 15, n)
        frame["INST_LATE_PAYMENT_COUNT"] = rng.integers(0, 5, n)
        frame["APP_EXT_SOURCE_1"] = rng.normal(size=n)
        frame.loc[:80, "APP_EXT_SOURCE_1"] = np.nan
        frame["BUREAU_DEBT_RATIO"] = rng.uniform(0, 2, n)
        frame.loc[:20, "BUREAU_DEBT_RATIO"] = np.nan
        frame["APP_DAYS_EMPLOYED"] = rng.integers(-10000, 0, n)
        frame.loc[10, "APP_DAYS_EMPLOYED"] = 365243
        return frame

    def test_split_repeats_phase_7a_two_stage_stratified_indices(self):
        frame = self._dataset()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            frame.to_csv(path, index=False)
            result = load_and_split_data(path)
        expected = train_test_split(frame[["SK_ID_CURR", *APPLICATION_FEATURES]], frame.TARGET,
                                    test_size=.15, stratify=frame.TARGET, random_state=42)
        X_train_val, X_test, y_train_val, y_test = expected
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=.15 / .85, stratify=y_train_val, random_state=42
        )
        for actual, expected_frame in zip(result[:3], [X_train, X_val, X_test]):
            np.testing.assert_array_equal(actual.SK_ID_CURR, expected_frame.SK_ID_CURR)
        for actual, expected_target in zip(result[3:], [y_train, y_val, y_test]):
            np.testing.assert_array_equal(actual.to_numpy(), expected_target.to_numpy())

    def test_model_preprocessing_is_fitted_on_train_and_outputs_probabilities(self):
        frame = self._dataset()
        features = FEATURE_GROUPS["Application + Bureau + Alternative"]
        train, valid = frame.iloc[:450], frame.iloc[450:]
        model = build_pipeline(features)
        model.fit(train[features], train.TARGET)
        ext_imputer = model.named_steps["preprocessor"].named_transformers_["ext"].named_steps["imputer"]
        expected_mean = train.APP_EXT_SOURCE_1.mean()
        self.assertAlmostEqual(ext_imputer.statistics_[0], expected_mean)
        bureau_imputer = model.named_steps["preprocessor"].named_transformers_["bureau"].named_steps["imputer"]
        self.assertAlmostEqual(bureau_imputer.statistics_[BUREAU_FEATURES.index("BUREAU_DEBT_RATIO")],
                               train.BUREAU_DEBT_RATIO.median())
        transformed = model.named_steps["preprocessor"].transform(valid[features])
        self.assertTrue(np.isfinite(transformed).all())
        probabilities = model.predict_proba(valid[features])[:, 1]
        self.assertEqual(probabilities.shape, (len(valid),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_invalid_leakage_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            build_pipeline(APPLICATION_FEATURES + ["TARGET"])
        with self.assertRaises(ValueError):
            build_pipeline(APPLICATION_FEATURES + ["SK_ID_CURR"])

    def test_locked_thin_file_mask_uses_zero_bureau_count(self):
        frame = pd.DataFrame({"BUREAU_CREDIT_COUNT": [0, 1, 0, 4]})
        np.testing.assert_array_equal(thin_file_mask(frame), [True, False, True, False])
        with self.assertRaises(ValueError):
            thin_file_mask(pd.DataFrame({"PREV_APP_COUNT": [0]}))


if __name__ == "__main__":
    unittest.main()
