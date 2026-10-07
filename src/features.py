"""
Feature engineering module orchestrating the SQL pipeline.
"""
import sqlite3
import pandas as pd
import os
from .data import create_connection, init_db, load_csv_to_sqlite

def generate_features(data_dir="data/raw", sql_dir="sql", output_dir="data/processed", db_path="data/interim/project.db"):
    """Run the end-to-end feature engineering pipeline using SQLite."""
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    conn = create_connection(db_path)
    
    # 1. Init Schema
    init_db(conn, os.path.join(sql_dir, "schema.sql"))
    
    # 2. Load data
    load_csv_to_sqlite(conn, os.path.join(data_dir, "application_train.csv"), "application_train", 
                       usecols=['SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
                                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])
    load_csv_to_sqlite(conn, os.path.join(data_dir, "bureau.csv"), "bureau", 
                       usecols=['SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT'])
    load_csv_to_sqlite(conn, os.path.join(data_dir, "previous_application.csv"), "previous_application", 
                       usecols=['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    load_csv_to_sqlite(conn, os.path.join(data_dir, "installments_payments.csv"), "installments_payments", 
                       usecols=['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'])
    
    # 3. Execute Feature Queries
    print("Executing SQL feature queries...")
    with open(os.path.join(sql_dir, "feature_queries.sql"), "r") as f:
        feature_sql = f.read()
    conn.executescript(feature_sql)
    conn.commit()
    
    # 4. Extract master feature table
    print("Exporting feature_master to processed directory...")
    master_df = pd.read_sql("SELECT * FROM feature_master", conn)
    master_df.to_csv(os.path.join(output_dir, "feature_master.csv"), index=False)
    
    conn.close()
    print("Feature engineering complete.")
    return master_df

if __name__ == "__main__":
    generate_features()
