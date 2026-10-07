"""
Data loading and database management module.
"""
import pandas as pd
import sqlite3
import os
import gc

def create_connection(db_file=":memory:"):
    """Create a database connection to SQLite."""
    conn = sqlite3.connect(db_file)
    return conn

def init_db(conn, schema_path="sql/schema.sql"):
    """Initialize database schema."""
    with open(schema_path, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()

def load_csv_to_sqlite(conn, csv_path, table_name, usecols=None):
    """Load raw CSV data into the SQLite database."""
    print(f"Loading {table_name} into DB...")
    df = pd.read_csv(csv_path, usecols=usecols)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    del df
    gc.collect()
