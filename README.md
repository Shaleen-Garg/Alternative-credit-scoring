# Alternative Credit Scoring for Thin-File Borrowers

## Overview
This project investigates whether alternative behavioral and financial history can improve probability-of-default prediction for borrowers with limited traditional credit history.

## Research Question
Can alternative behavioral and financial data improve credit-risk prediction for borrowers with limited traditional credit history?

## Planned Approach
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
Gradient Boosting
        ↓
Probability calibration
        ↓
Cost-aware decision threshold
        ↓
Explainability
        ↓
Stability / responsible AI analysis
```

## Planned Experiments
The primary experiment will compare:
1. Application-only features
2. Application + traditional credit information
3. Application + traditional credit + alternative behavioral information

The comparison must be performed for:
- Overall population
- Thin-file population

The purpose is to determine whether alternative information provides incremental predictive value, particularly for thin-file borrowers.

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
**Repository setup / dataset preparation**

## Disclaimer
This is an educational/research portfolio project and is not a production lending model.
