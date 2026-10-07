# Alternative Credit Scoring for Thin-File Borrowers

## Overview
This project investigates whether alternative behavioral and financial history can improve probability-of-default prediction for borrowers with limited traditional credit history.

## Research Question
Can alternative behavioral and financial data improve credit-risk prediction for borrowers with limited traditional credit history?

## Modeling Approach
```text
Relational financial data
        ↓
PostgreSQL / SQL feature engineering
        ↓
Thin-file identification
        ↓
Statistical analysis
        ↓
Credit scorecard / Logistic Regression
        ↓
Controlled Logistic Regression ablation (A/B/C)
        ↓
Nonlinear robustness and probability calibration
```

## Controlled Feature-Group Experiment
The completed primary experiment compares:
1. Application-only features
2. Application + traditional credit information
3. Application + traditional credit + alternative behavioral information

using the same stratified 70/15/15 train/validation/test split, preprocessing, and class-weighted Logistic Regression configuration. Results are evaluated for:
- Overall population
- Thin-file population

The purpose is to estimate whether alternative information adds predictive value beyond application and bureau information, particularly for thin-file borrowers. Measured results and limitations are in [the Phase 8 report](docs/phase8_ablation.md).

## Evaluation
Planned metrics:
- ROC-AUC
- PR-AUC
- KS statistic
- Precision
- Recall
- Brier score
- Calibration
- Confusion matrix

## Technology
- Python
- PostgreSQL
- SQL
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- SHAP
- Jupyter

## Dataset
The project uses the **Home Credit Default Risk** dataset.

## Status
**Phases 7A–10 complete:** the frozen-split logistic ablation (Phase 8), HistGradientBoosting robustness comparison (Phase 9), and validation-selected probability calibration (Phase 10). On the Phase 8 test set, adding non-bureau history to application plus bureau features increased ROC-AUC by 0.0090 and PR-AUC by 0.0095 overall; in thin-file borrowers the changes were +0.0190 and +0.0269. The nonlinear Model C also improved over its Model B on the test set. Phase 8 test results were inspected during the earlier benchmark, so this is a reused historical holdout, not an untouched final test. Findings are predictive associations, not causal or production claims. See [Phase 8](docs/phase8_ablation.md), [Phase 9](docs/phase9_nonlinear.md), and [Phase 10](docs/phase10_calibration.md).

## Disclaimer
This is an educational/research portfolio project and is not a production lending model.
