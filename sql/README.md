# SQL feature pipeline

SQLite provides the local relational layer. The feature builder initializes schema.sql, imports the selected Home Credit CSVs in chunks, adds applicant-key indexes, runs the aggregations in feature_queries.sql, and checks target values, historical dates, and output grain with data_quality.sql before export.

Historical tables are aggregated before joining to the application table to preserve one row per applicant. Run the rebuild from the repository root with python -m src.features. See ../data/README.md for source-file setup and ../docs/reproducibility.md for the complete workflow.
