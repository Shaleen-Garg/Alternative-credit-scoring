# Baseline Credit Model (Phase 7A)

## 1. Objective
Build an interpretable baseline credit-risk model using ONLY the Application feature group. This establishes a clean performance baseline against which traditional credit features and alternative behavioral features will later be compared via an ablation study.

## 2. Features Used
Only the following 10 raw/derived application variables were permitted:
- `APP_INCOME_TOTAL`
- `APP_CREDIT_AMOUNT`
- `APP_ANNUITY`
- `APP_DAYS_BIRTH`
- `APP_DAYS_EMPLOYED`
- `APP_EXT_SOURCE_1`
- `APP_EXT_SOURCE_2`
- `APP_EXT_SOURCE_3`
- `APP_CREDIT_INCOME_RATIO`
- `APP_ANNUITY_INCOME_RATIO`

*Note: `SK_ID_CURR` was strictly removed prior to fitting, and all temporal / `TARGET` leakage rules were enforced.*

## 3. Split Methodology
Due to the absence of a genuine temporal application date, the dataset was split using a reproducible stratified random split based on `TARGET`.
- **Training Set**: 70% (215,257 rows)
- **Validation Set**: 15% (46,127 rows)
- **Test Set**: 15% (46,127 rows)
- **Class Balance**: Strictly maintained at ~8.07% across all splits.

## 4. Preprocessing & Imbalance Treatment
All preprocessing parameters were learned strictly on the training set:
- **Anomalies**: `APP_DAYS_EMPLOYED` == 365243 was converted to `NaN`.
- **Imputation**: Missing external sources (`APP_EXT_SOURCE_X`) were imputed using the mean, with explicit missingness indicator columns generated. Other numeric columns used median imputation.
- **Scaling**: Standard scaling (`mu=0, sigma=1`) was applied.
- **Imbalance**: A regularized Logistic Regression model (`C=1.0`) was trained using `class_weight='balanced'` to shift the model's focus toward probability/risk prediction rather than accuracy maximization.

## 5. Train/Validation/Test Metrics

| Metric | Training | Validation | Test |
|---|---|---|---|
| **ROC-AUC** | 0.7268 | 0.7315 | **0.7294** |
| **PR-AUC** | 0.2024 | 0.2147 | **0.2085** |

*Observation: Ranking metrics are close across these splits. The test set has since been inspected in Phase 8 and is not an untouched final holdout.*

## 6. Comparison Against Constant-Probability Baseline
To ensure the logistic regression learned meaningful signals rather than simply defaulting to the base rate, it was compared to a dummy baseline that strictly predicts $P(default) = 0.0807$.

**Test Set Results**:
| Metric | Baseline Logistic Model | Constant Probability (Base Rate) |
|---|---|---|
| **ROC-AUC** | 0.7294 | 0.5000 |
| **PR-AUC** | 0.2085 | 0.0807 |
| **Brier Score** | 0.2104 | **0.0742** |
| **Log Loss** | 0.6094 | **0.2805** |

*Note on Calibration: Because the logistic model was trained with `class_weight='balanced'`, its output probabilities are structurally shifted away from the raw base rate (~8%). Consequently, the raw Brier Score and Log Loss look superficially worse than the constant base rate. The ROC-AUC and PR-AUC, which measure ranking ability, unequivocally prove the model provides massive predictive lift. Formal probability calibration will be performed in Phase 10.*

## 7. Model Interpretation (Coefficients)
Sorted by absolute magnitude (standardized coefficients):

| Feature | Coefficient | Odds Ratio |
|---|---|---|
| `APP_EXT_SOURCE_3` | -0.501 | 0.605 |
| `APP_EXT_SOURCE_2` | -0.428 | 0.651 |
| `APP_EXT_SOURCE_1` | -0.252 | 0.777 |
| `APP_DAYS_EMPLOYED` | +0.145 | 1.156 |
| `APP_EXT_SOURCE_1_missing_indicator` | +0.141 | 1.152 |
| `APP_ANNUITY_INCOME_RATIO` | +0.110 | 1.116 |

*Key Takeaways:*
1. The External Sources completely dominate the model.
2. Missingness in `EXT_SOURCE_1` has its own independent positive coefficient (+0.141), confirming the Phase 6 EDA finding that missingness is associated with higher risk.
3. High payment burdens (`APP_ANNUITY_INCOME_RATIO`) increase risk.

## 8. Limitations & Next Steps
- This is merely the Application-only baseline.
- It has not been formally calibrated to real-world default probabilities.
- Subsequent work is documented in the Phase 8 controlled ablation, Phase 9 nonlinear comparison, and Phase 10 calibration reports.
