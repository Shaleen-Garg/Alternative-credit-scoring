# Reproduce the analysis

## Data and environment

Use Python 3.10 or later and install the pinned project dependencies:

```text
python -m pip install -r requirements.txt
```

Place the required Home Credit CSVs in `data/raw/` as described in [data setup](../data/README.md). The raw files, generated SQLite database, and feature table are local-only. The checked-in train, validation, and test ID files in `data/processed/splits/` define the frozen split. Do not regenerate or edit these IDs.

## Full reproduction sequence

Run from the repository root, in order:

```text
python -m src.features
python -m src.nonlinear
python -m src.phase10
python -m src.phase11
python -m src.phase12
python -m src.phase13
python -m src.release
python -m pytest -q
```

`src.features` rebuilds the SQLite database and feature table. Phase 9 selects the small HGB candidate set using validation PR-AUC and writes the selected parameters. Phase 10 selects a predictive model and calibration mapping using training and validation only. Phases 11–13 reconstruct the selected model; Phase 11 chooses cost and approval-volume thresholds on validation, Phase 12 explains the same Model C, and Phase 13 computes descriptive cohort and stability diagnostics. None of these stages changes the frozen split IDs. `src.release` assembles the compact outputs under `reports/final/` from the phase tables.

On 8 October 2026, the final requested test run completed with **26 passed**. The Phase 10 reconciliation audit records matching Phase 9/10 uncalibrated test metrics, ordered split fingerprints, and that test labels were not used in selection or fitting. Runtime and hardware may affect elapsed time, but the seeded analysis and outputs are reproducible for the same source data and dependencies.

## Generated outputs

- `data/interim/project.db`: rebuilt local SQLite database.
- `data/processed/feature_master.csv`: borrower-level modeling table.
- `reports/tables/phase*.csv` and `.json`: phase-level evaluation and audit records.
- `reports/final/`: curated portfolio tables and figures.

Raw data, database files, and processed features are ignored by Git. Never commit them. The repository does commit frozen split IDs so the evaluated borrowers remain consistent.
