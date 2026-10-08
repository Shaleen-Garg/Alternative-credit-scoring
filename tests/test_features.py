import sqlite3
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_TABLE = Path("data/processed/feature_master.csv")


class TestSqlFeaturePipeline(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(Path("sql/schema.sql").read_text(encoding="utf-8"))
        self.conn.executemany(
            "INSERT INTO application_train "
            "(SK_ID_CURR, TARGET, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY) "
            "VALUES (?, ?, ?, ?, ?)",
            [(1, 0, 1000, 500, 100), (2, 1, 0, 500, 100)],
        )
        self.conn.executemany(
            "INSERT INTO bureau VALUES (?, ?, ?, ?, ?, ?)",
            [(1, 11, "Active", -365, 100, 50), (1, 12, "Closed", -100, 100, 150)],
        )
        self.conn.executemany(
            "INSERT INTO previous_application VALUES (?, ?, ?, ?)",
            [(101, 1, "Approved", -100), (102, 1, "Refused", -200)],
        )
        self.conn.executemany(
            "INSERT INTO installments_payments VALUES (?, ?, ?, ?, ?, ?)",
            [(201, 1, -30, -25, 100, 80),
             (202, 1, -60, -65, 100, 100),
             (203, 1, -15, None, 100, None),
             (204, 1, -10, -12, 100, None)],
        )
        self.conn.executescript(Path("sql/feature_queries.sql").read_text(encoding="utf-8"))

    def tearDown(self):
        self.conn.close()

    def test_aggregations_preserve_applicant_grain_and_zero_history(self):
        features = pd.read_sql_query("SELECT * FROM feature_master ORDER BY SK_ID_CURR", self.conn)
        self.assertEqual(len(features), 2)
        self.assertTrue(features.SK_ID_CURR.is_unique)
        applicant = features.iloc[0]
        no_history = features.iloc[1]
        self.assertEqual(applicant.APP_CREDIT_INCOME_RATIO, 0.5)
        self.assertEqual(applicant.APP_ANNUITY_INCOME_RATIO, 0.1)
        self.assertTrue(pd.isna(no_history.APP_CREDIT_INCOME_RATIO))
        self.assertEqual(applicant.BUREAU_CREDIT_COUNT, 2)
        self.assertEqual(applicant.BUREAU_ACTIVE_COUNT, 1)
        self.assertEqual(applicant.BUREAU_TOTAL_CREDIT, 200)
        self.assertEqual(applicant.BUREAU_TOTAL_DEBT, 200)
        self.assertEqual(applicant.BUREAU_DEBT_RATIO, 1)
        self.assertEqual(applicant.PREV_APP_COUNT, 2)
        self.assertEqual(applicant.PREV_APPROVED_COUNT, 1)
        self.assertEqual(applicant.PREV_REFUSED_COUNT, 1)
        self.assertEqual(no_history.BUREAU_CREDIT_COUNT, 0)
        self.assertEqual(no_history.BUREAU_DEBT_RATIO, 0)
        self.assertEqual(no_history.PREV_APP_COUNT, 0)
        self.assertEqual(no_history.INST_TOTAL_COUNT, 0)

    def test_payment_missingness_is_separate_from_observed_lateness(self):
        row = self.conn.execute(
            "SELECT INST_TOTAL_COUNT, INST_LATE_PAYMENT_COUNT, INST_LATE_PAYMENT_RATIO, "
            "INST_UNDERPAYMENT_COUNT, INST_PAYMENT_MISSING_COUNT "
            "FROM feature_master WHERE SK_ID_CURR=1"
        ).fetchone()
        self.assertEqual(row, (4, 1, 1 / 3, 1, 2))

    def test_quality_checks_detect_future_history_dates(self):
        query = Path("sql/data_quality.sql").read_text(encoding="utf-8")
        checks = pd.read_sql_query(query, self.conn).set_index("check_name")
        self.assertTrue((checks.violation_count == 0).all())

        self.conn.execute("UPDATE previous_application SET DAYS_DECISION=1 WHERE SK_ID_PREV=101")
        self.conn.execute("UPDATE installments_payments SET DAYS_ENTRY_PAYMENT=1 WHERE SK_ID_PREV=201")
        checks = pd.read_sql_query(query, self.conn).set_index("check_name")
        self.assertEqual(checks.loc["future_previous_application", "violation_count"], 1)
        self.assertEqual(checks.loc["future_installment_payment_date", "violation_count"], 1)


@unittest.skipUnless(FEATURE_TABLE.exists(), "Build the Home Credit feature table to run integration checks")
class TestFeatureMatrixIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(FEATURE_TABLE)

    def test_row_count_matches_home_credit_training_population(self):
        self.assertEqual(len(self.df), 307511)

    def test_one_row_per_applicant(self):
        self.assertEqual(self.df.SK_ID_CURR.nunique(), len(self.df))

    def test_thin_file_population_size(self):
        self.assertEqual(int(self.df.BUREAU_CREDIT_COUNT.eq(0).sum()), 44020)

    def test_thin_file_previous_application_coverage(self):
        thin = self.df[self.df.BUREAU_CREDIT_COUNT.eq(0)]
        self.assertEqual(int(thin.PREV_APP_COUNT.gt(0).sum()), 41550)

    def test_target_is_the_only_target_column(self):
        target_columns = [column for column in self.df if "TARGET" in column.upper()]
        self.assertEqual(target_columns, ["TARGET"])

    def test_numeric_features_have_no_infinities(self):
        numeric = self.df.select_dtypes(include=[np.number])
        self.assertFalse(np.isinf(numeric.to_numpy(dtype=float)).any())

    def test_credit_income_ratio_is_nonnegative_when_defined(self):
        ratio = self.df.APP_CREDIT_INCOME_RATIO.dropna()
        self.assertTrue(ratio.ge(0).all())


if __name__ == "__main__":
    unittest.main()
