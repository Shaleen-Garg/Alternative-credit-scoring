"""CSV loading and SQLite setup helpers."""

import pandas as pd
import sqlite3

def create_connection(db_file=":memory:"):
    return sqlite3.connect(db_file)

def init_db(conn, schema_path="sql/schema.sql"):
    with open(schema_path, encoding="utf-8") as schema_file:
        schema = schema_file.read()
    conn.executescript(schema)
    conn.commit()

def load_csv_to_sqlite(conn, csv_path, table_name, usecols=None,
                       read_chunk_size=250_000, insert_chunk_size=10_000):
    if read_chunk_size < 1 or insert_chunk_size < 1:
        raise ValueError("CSV chunk sizes must be positive")
    print(f"Loading {table_name} into DB...")
    row_count = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=read_chunk_size):
        chunk.to_sql(table_name, conn, if_exists="append", index=False,
                     chunksize=insert_chunk_size)
        row_count += len(chunk)
    if row_count == 0:
        raise ValueError(f"Source file contains no data rows: {csv_path}")
