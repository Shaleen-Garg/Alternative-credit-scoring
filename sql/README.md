# SQL feature pipeline

SQLite provides the local relational layer. The feature builder initializes the schema from schema.sql, loads the selected Home Credit CSV files, runs the aggregations in feature_queries.sql, checks data quality with data_quality.sql, and exports the borrower-level feature table.

Historical tables are aggregated before joining to the application table to preserve one row per applicant. Run the rebuild from the repository root with python -m src.features. See ../data/README.md for source-file setup and ../docs/reproducibility.md for the complete workflow.
