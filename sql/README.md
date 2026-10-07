# SQL feature pipeline

The project uses SQLite as a local relational layer. `src.features` initializes the schema from `schema.sql`, loads selected Home Credit CSVs, applies the joins and aggregations in `feature_queries.sql`, runs data checks from `data_quality.sql`, and exports the borrower-level feature table used by the model modules.

Run the full rebuild from the repository root with `python -m src.features`. See [local data setup](../data/README.md) for the required files and [reproducibility](../docs/reproducibility.md) for the complete analysis sequence.
