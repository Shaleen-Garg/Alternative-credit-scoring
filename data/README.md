# Local data setup

Place the Home Credit source CSVs in data/raw/. The competition dataset includes these source and reference files:

- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- installments_payments.csv
- HomeCredit_columns_description.csv

The current feature builder requires application_train.csv, bureau.csv, previous_application.csv, and installments_payments.csv. The remaining files provide additional dataset context but are not consumed by this implementation. The raw dataset is not included in this repository and must be obtained separately.

Run python -m src.features from the repository root. This rebuilds the local SQLite database in data/interim/project.db and writes data/processed/feature_master.csv. These generated files and all source data are intentionally excluded from version control.

The train, validation, and test ID CSVs in data/processed/splits/ are the committed, frozen split definition. Keep those files unchanged when reproducing the analysis.
