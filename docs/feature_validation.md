# Feature Validation and EDA

This document details the quality and distributional findings from the engineered feature dataset (`feature_master.csv`).

## 1. Dataset Integrity
- **Rows/Granularity**: 307,511 rows. Exactly 1 row per `SK_ID_CURR`.
- **Target Integrity**: `TARGET` is the designated outcome variable. Feature construction tests confirmed that `TARGET` is not present in the predictor feature set. Overall default rate is 8.07%.
- **Missingness & Coverage**: 
  - `APP_EXT_SOURCE_1` is missing 56.3% of values.
  - `APP_EXT_SOURCE_3` is missing 19.8% of values.
  - `APP_ANNUITY` has negligible missingness (12 rows).
  - All engineered features default to 0 gracefully for borrowers lacking history in the respective tables.

## 2. Feature-Level Anomalies & Findings
- **Constant features**: None.
- **Infinite values**: Checked programmatically, 0 found.
- **Anomalies**: `APP_DAYS_EMPLOYED` contains a known anomaly where Pensioners are encoded with the value `365243` (exactly 1000 years). **Decision**: This must be explicitly handled (e.g., converted to `NaN` or 0 with a boolean flag) during the modelling phase.
- **Correlations**: `INST_LATE_PAYMENT_COUNT` and `INST_UNDERPAYMENT_COUNT` are highly correlated ($r \approx 0.90$). **Decision**: We will investigate whether this correlation is merely behavioral or partly structural in relationship with `INST_TOTAL_COUNT`. Redundancy will be evaluated separately for the linear scorecard/logistic model and the tree-based model. We will not automatically remove features solely because of correlation at this stage.

## 3. Installment Lateness Logic Verification
- Evaluated logic:
  `payment_delay = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`
  Late payment iff:
  `payment_delay > 0`
  which is equivalent to:
  `DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT`
- An automated unit test validates implementation of this rule against the negative-day convention.

## 4. Thin-File Specific Findings
- **Size**: 44,020 borrowers (`BUREAU_CREDIT_COUNT = 0`).
- **Default Rate**: 10.12% (Thin-file) vs 7.73% (Non-thin-file). Thin-file borrowers represent elevated credit risk.
- **Alternative Data Coverage**: Of the 44,020 thin-file borrowers, 41,550 (94.4%) have usable alternative behavioral history, providing sufficient coverage to test whether alternative behavioral features improve risk prediction within the thin-file population. (Note: Coverage does not establish predictive improvement; the upcoming ablation study will test that).

## 5. Feature Groups Summary
| Group | Number of features | Missingness | Main purpose |
|---|---:|---:|---|
| Application | 10 | Med/High (EXT_SOURCES) | Current application information |
| Traditional Credit | 6 | 0 | Bureau history |
| Alternative Behavioural | 7 | 0 | Internal behavioural/payment history |

## 6. History Absence vs Zero Behavior
The zero-imputation strategy creates a structural ambiguity where `0` can mean either "No historical records exist" or "Historical records exist, but the adverse event count is zero (clean history)".
- **Evidence**: `INST_LATE_PAYMENT_COUNT` has 152,512 zeros. However, 136,644 of these occur where `INST_TOTAL_COUNT > 0`. Thus, 136,644 represent a perfectly clean payment history, while 15,868 represent a complete lack of installment history.
- **Decision**: During the modeling phase, we must decide whether to introduce explicit history-presence indicators (e.g., `HAS_INSTALLMENT_HISTORY`, `HAS_PREV_APP_HISTORY`) to allow models to differentiate "no history" from "clean history", especially for linear scorecards.

## 7. Missingness Analysis
For the external scores, missingness is not MCAR (Missing Completely At Random). Missingness itself is associated with higher default risk:
- `APP_EXT_SOURCE_1`: Target rate when missing is **8.52%**, vs **7.50%** when present.
- `APP_EXT_SOURCE_3`: Target rate when missing is **9.31%**, vs **7.77%** when present.
Missingness must be treated carefully; imputing the mean might mask this valuable risk signal.

## 8. Univariate Target Analysis (Restrained)
Selected univariate findings (T0 = Good, T1 = Default):
- `APP_EXT_SOURCE_1`: Mean is significantly lower for defaulters (0.38) vs good borrowers (0.51).
- `APP_EXT_SOURCE_3`: Mean is significantly lower for defaulters (0.39) vs good borrowers (0.52).
- `INST_LATE_PAYMENT_RATIO`: Higher for defaulters (9.9%) compared to good borrowers (6.9%).
- Within the **Thin-File Population** (0 bureau records): The alternative behavioral signals remain highly separated. For example, thin-file defaulters average a late payment ratio of 10.3%, compared to 6.7% for thin-file good borrowers.

## Visualizations Generated
