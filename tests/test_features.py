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

if __name__ == '__main__':
    unittest.main()
