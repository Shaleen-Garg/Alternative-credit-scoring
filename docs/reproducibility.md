# Reproducing the final analysis

The committed split-ID artifacts in `data/processed/splits/` are the only supported split definition. Keep the corrected `data/processed/feature_master.csv` generated from `sql/feature_queries.sql` and the Home Credit source data available locally; raw data and the feature table are intentionally not committed.

From the repository root, with dependencies from `requirements.txt` installed:

```text
python -m src.nonlinear
python -m src.phase10
python -m src.phase11
python -m src.phase12
python -m src.phase13
python -m pytest -q
```

Phase 9 selects its small HGB candidate set using validation PR-AUC and writes the frozen parameters. Phase 10 selects/calibrates using training and validation only and records the input/split reconciliation in `reports/tables/phase10_reconciliation_audit.json`. Phases 11–13 reconstruct the selected model and validation-fitted sigmoid mapping; Phase 11 selects cost thresholds on validation and evaluates those fixed thresholds on test. Phase 12 explains the same Model C; Phase 13 is descriptive and does not alter selection. Re-running modules overwrites the corresponding generated reports but does not change frozen split IDs.
