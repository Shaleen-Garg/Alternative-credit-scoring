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
Later phases: nonlinear models, calibration, decisions,
explainability, and stability analysis
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
**Phases 7A, 7B, and 8 complete:** controlled Application, Application + Bureau, and Application + Bureau + Alternative baselines. Alternative history improves ranking performance in the current holdout experiment; this is an observational predictive result, not a causal or production claim.

## Disclaimer
This is an educational/research portfolio project and is not a production lending model.
