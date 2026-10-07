# Local data setup

Place the Home Credit source CSVs in **`data/raw/`**. To rebuild the feature table from scratch, the feature pipeline requires these four files:

- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`

The remaining competition files may also be stored there for reference, but the current feature builder does not consume them. The data is available from the Home Credit Default Risk competition on Kaggle; comply with the dataset's access and use terms.

Run `python -m src.features` from the repository root. This rebuilds the local SQLite database in `data/interim/project.db` and writes `data/processed/feature_master.csv`. These generated files and all source data are intentionally excluded from version control. Do not put private or borrower-level data in Git.

The train, validation, and test ID CSVs in `data/processed/splits/` are the committed, frozen split definition. Keep those files unchanged when reproducing the analysis.
