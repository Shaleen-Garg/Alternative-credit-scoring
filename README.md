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
Nonlinear robustness
        ↓
Probability calibration
        ↓
Illustrative decision analysis
        ↓
Model explanations and stability checks
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
Reported diagnostics include:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- Brier score
- Log loss and calibration diagnostics
- Confusion counts and illustrative expected cost
- Validation permutation importance and split/cohort stability

## Technology
- Python
- SQL feature engineering
- pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Jupyter

## Dataset
The project uses the **Home Credit Default Risk** dataset.

## Reproducibility
The project freezes its application-level split IDs and keeps raw data local. See [the reproducibility instructions](docs/reproducibility.md) for the analysis sequence and data requirements.

## Status
**Phases 7A–13 complete:** application variables establish predictive signal; bureau history adds a modest increment; non-bureau behavioural history adds further signal. On the frozen historical holdout, logistic Model C improves over Model B by 0.0090 ROC-AUC/0.0095 PR-AUC overall and 0.0190/0.0269 among thin-file borrowers. The gain remains under HGB: Model C reaches **0.7579 ROC-AUC / 0.2509 PR-AUC**, with thin-file gains of 0.0197 ROC-AUC/0.0371 PR-AUC over Model B. Validation-selected sigmoid calibration preserves ranking and gives small overall calibration improvements; thin-file Brier/log loss improve slightly while ECE worsens slightly. Illustrative cost analysis finds modest overall decision differences and clearer thin-file gains at comparable approval volumes. Validation permutation importance shows signal from each feature group, led by application variables. Selected feature distributions and score distributions were close across the random splits, while performance varies across age and missing-score cohorts. These are observations from one historical dataset/holdout, not out-of-time performance, fairness proof, causality, real Ujjivan economics, or production banking utility. See [Phase 8](docs/phase8_ablation.md), [Phase 9](docs/phase9_nonlinear.md), [Phase 10](docs/phase10_calibration.md), [Phase 11](docs/phase11_decision_analysis.md), [Phase 12](docs/phase12_explainability.md), and [Phase 13](docs/phase13_stability.md).

## Disclaimer
This is an educational/research portfolio project and is not a production lending model.
