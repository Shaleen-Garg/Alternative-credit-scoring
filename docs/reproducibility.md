# Reproducibility

## Requirements and data

Use Python 3.10 and install dependencies from requirements.txt:

~~~bash
python -m pip install -r requirements.txt
~~~

The Home Credit source dataset is not included. Place the supplied CSVs in data/raw/ as listed in ../data/README.md. The current feature build requires application_train.csv, bureau.csv, previous_application.csv, and installments_payments.csv; it does not consume every table in the competition dataset.

The train, validation, and test borrower IDs in data/processed/splits/ are committed and define the fixed split. Do not regenerate or edit them.

## Run the analysis

From the repository root, execute the commands in order:

~~~bash
python -m src.features
python -m src.ablation
python -m src.nonlinear
python -m src.calibration_analysis
python -m src.decision_analysis
python -m src.explainability
python -m src.stability_analysis
python -m src.release
python -m pytest -q
~~~

The feature builder creates the local SQLite database and applicant-level table. The analysis modules run the controlled feature comparison, nonlinear comparison, calibration, decision, explanation, and stability analyses. The release module assembles the curated tables and figures in reports/results/. The full test suite last passed with 26 tests.

## Generated and committed files

- data/interim/project.db and data/processed/feature_master.csv are generated locally and ignored by Git.
- Detailed run tables and intermediate figures are generated under reports/tables/ and reports/figures/ and ignored by Git.
- Curated summary tables and figures are stored under reports/results/.
- Frozen split IDs are committed so the borrower assignments remain consistent.

Runtime and numerical details may vary across machines and library builds; bit-for-bit output is not guaranteed.
