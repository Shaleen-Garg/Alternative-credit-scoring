import unittest
import pandas as pd
import numpy as np
from src.models import load_and_split_data, build_baseline_pipeline

class TestBaselineModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filepath = r"C:\ujjivan_project\data\processed\feature_master.csv"
        cls.X_train, cls.X_val, cls.X_test, cls.y_train, cls.y_val, cls.y_test = load_and_split_data(cls.filepath)
        cls.pipeline = build_baseline_pipeline()
        
        cls.X_train_features = cls.X_train.drop(columns=['SK_ID_CURR'])
        cls.X_val_features = cls.X_val.drop(columns=['SK_ID_CURR'])
        cls.X_test_features = cls.X_test.drop(columns=['SK_ID_CURR'])
        
        # Fit on a tiny sample for speed in tests
        sample_idx = np.random.choice(len(cls.X_train_features), 1000, replace=False)
        cls.pipeline.fit(cls.X_train_features.iloc[sample_idx], cls.y_train.iloc[sample_idx])

    def test_split_sizes(self):
        """Verify sizes are approx 70/15/15"""
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        self.assertAlmostEqual(len(self.X_train) / total, 0.70, delta=0.01)
        self.assertAlmostEqual(len(self.X_val) / total, 0.15, delta=0.01)
        self.assertAlmostEqual(len(self.X_test) / total, 0.15, delta=0.01)

    def test_stratification(self):
        """Verify TARGET stratification is preserved."""
        overall_rate = (self.y_train.sum() + self.y_val.sum() + self.y_test.sum()) / (len(self.y_train) + len(self.y_val) + len(self.y_test))
        self.assertAlmostEqual(self.y_train.mean(), overall_rate, delta=0.005)
        self.assertAlmostEqual(self.y_val.mean(), overall_rate, delta=0.005)
        self.assertAlmostEqual(self.y_test.mean(), overall_rate, delta=0.005)

    def test_no_id_or_target_in_predictors(self):
        """Verify SK_ID_CURR and TARGET are not predictors."""
        self.assertNotIn('SK_ID_CURR', self.X_train_features.columns)
        self.assertNotIn('TARGET', self.X_train_features.columns)

    def test_predictions_valid(self):
        """Verify predictions are probabilities [0,1] and no NaNs."""
        preds = self.pipeline.predict_proba(self.X_val_features[:100])[:, 1]
        self.assertEqual(len(preds), 100)
        self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))
        self.assertFalse(np.isnan(preds).any())

if __name__ == '__main__':
    unittest.main()
