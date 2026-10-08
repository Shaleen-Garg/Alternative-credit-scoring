"""
Feature engineering module orchestrating the SQL pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from .data import create_connection, init_db, load_csv_to_sqlite

def generate_features(data_dir="data/raw", sql_dir="sql", output_dir="data/processed", db_path="data/interim/project.db"):
    """Run the end-to-end feature engineering pipeline using SQLite."""
    data_dir = Path(data_dir)
    sql_dir = Path(sql_dir)
    output_dir = Path(output_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = create_connection(db_path)
    try:
        init_db(conn, sql_dir / "schema.sql")

        load_csv_to_sqlite(
            conn, data_dir / "application_train.csv", "application_train",
            usecols=["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                     "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        )
        load_csv_to_sqlite(
            conn, data_dir / "bureau.csv", "bureau",
            usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
                     "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"],
        )
        load_csv_to_sqlite(
            conn, data_dir / "previous_application.csv", "previous_application",
            usecols=["SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION"],
        )
        load_csv_to_sqlite(
            conn, data_dir / "installments_payments.csv", "installments_payments",
            usecols=["SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                     "AMT_INSTALMENT", "AMT_PAYMENT"],
        )

        print("Executing SQL feature queries...")
        conn.executescript((sql_dir / "feature_queries.sql").read_text(encoding="utf-8"))
        conn.commit()

        checks = pd.read_sql_query(
            (sql_dir / "data_quality.sql").read_text(encoding="utf-8"), conn
        )
        failures = checks.loc[checks["violation_count"] > 0]
        if not failures.empty:
            details = ", ".join(
                f"{row.check_name}={row.violation_count}"
                for row in failures.itertuples(index=False)
            )
            raise ValueError(f"Feature pipeline data-quality checks failed: {details}")

        master_df = pd.read_sql_query("SELECT * FROM feature_master", conn)
    finally:
        conn.close()

    numeric = master_df.select_dtypes(include="number")
    if numeric.size and np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Feature table contains infinite values")
    predictors = master_df.drop(columns=["SK_ID_CURR", "TARGET"])
    non_numeric = predictors.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Predictor columns must be numeric: {non_numeric}")
    constant_features = predictors.columns[predictors.nunique(dropna=False) <= 1].tolist()
    if constant_features:
        raise ValueError(f"Feature table contains constant predictors: {constant_features}")

    print("Exporting feature_master to processed directory...")
    master_df.to_csv(output_dir / "feature_master.csv", index=False)
    print("Feature engineering complete.")
    return master_df

if __name__ == "__main__":
    generate_features()
