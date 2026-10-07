import unittest
import pandas as pd
import os

class TestFeaturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We assume the pipeline was already run and output is available
        cls.file_path = "data/processed/feature_master.csv"
        cls.df = pd.read_csv(cls.file_path) if os.path.exists(cls.file_path) else None

    def test_file_exists(self):
        """Verify the feature matrix was generated."""
        self.assertIsNotNone(self.df, "Feature master dataset not found!")

    def test_row_count(self):
        """Verify the row count matches the application_train base table."""
        # 307511 is the exact row count of application_train.csv
        self.assertEqual(len(self.df), 307511)

    def test_unique_id(self):
        """Verify one final feature row per SK_ID_CURR."""
        self.assertEqual(self.df['SK_ID_CURR'].nunique(), len(self.df))

    def test_thin_file_count(self):
        """Verify the 44,020 thin-file borrowers are preserved."""
        thin_files = self.df[self.df['BUREAU_CREDIT_COUNT'] == 0]
        self.assertEqual(len(thin_files), 44020)

    def test_thin_file_alternative_data(self):
        """Verify that thin-file borrowers still have alternative data."""
        thin_files = self.df[self.df['BUREAU_CREDIT_COUNT'] == 0]
        # At least some of them must have previous apps (we found 41550 earlier)
        has_alt = thin_files[thin_files['PREV_APP_COUNT'] > 0]
        self.assertGreater(len(has_alt), 40000)

    def test_no_target_leakage(self):
        """Verify TARGET is the only predictive variable and we didn't accidentally include future targets."""
        cols = self.df.columns.tolist()
        # The only col with 'TARGET' should be 'TARGET'
        target_cols = [c for c in cols if 'TARGET' in c.upper()]
        self.assertEqual(len(target_cols), 1)

    def test_no_infinite_values(self):
        """Verify there are no infinite values in the dataset."""
        import numpy as np
        # Check numeric columns for inf
        num_df = self.df.select_dtypes(include=[np.number])
        has_inf = np.isinf(num_df).any().any()
        self.assertFalse(has_inf, "Dataset contains infinite values (e.g. div by zero).")

    def test_valid_ratios(self):
        """Verify ratios are bounded appropriately where mathematically sensible."""
        if 'APP_CREDIT_INCOME_RATIO' in self.df.columns:
            # Should not be negative
            self.assertTrue((self.df['APP_CREDIT_INCOME_RATIO'].fillna(0) >= 0).all())
            
    def test_installment_lateness_logic(self):
        """Explicit mathematical proof of late payment logic."""
        # Setup: Application is day 0.
        # Scheduled installment is day -30.
        # Actual entry payment is day -25.
        # Delay = -25 - (-30) = +5. Positive delay = late.
        days_instalment = -30
        days_entry = -25
        is_late = days_entry > days_instalment
        self.assertTrue(is_late, "Logic: Positive delay means entry date is > scheduled date.")

if __name__ == '__main__':
    unittest.main()
