# Feature Validation and EDA

This document details the quality and distributional findings from the engineered feature dataset (`feature_master.csv`).

## 1. Dataset Integrity
- **Rows/Granularity**: 307,511 rows. Exactly 1 row per `SK_ID_CURR`.
- **Target Integrity**: `TARGET` is perfectly separated. No other variables contain future default outcomes. Overall default rate is 8.07%.
- **Missingness & Coverage**: 
  - `APP_EXT_SOURCE_1` is missing 56.3% of values.
  - `APP_EXT_SOURCE_3` is missing 19.8% of values.
  - `APP_ANNUITY` has negligible missingness (12 rows).
  - All engineered features default to 0 gracefully for borrowers lacking history in the respective tables.

## 2. Feature-Level Anomalies & Findings
- **Constant features**: None.
- **Infinite values**: Checked programmatically, 0 found.
- **Anomalies**: `APP_DAYS_EMPLOYED` contains a known anomaly where Pensioners are encoded with the value `365243` (exactly 1000 years). **Decision**: This must be explicitly handled (e.g., converted to `NaN` or 0 with a boolean flag) during the modelling phase.
- **Correlations**: `INST_LATE_PAYMENT_COUNT` and `INST_UNDERPAYMENT_COUNT` are highly correlated ($r = 0.90$). **Decision**: Retain both for now; tree-based models handle this well, and linear models will require regularization.

## 3. Installment Lateness Logic Verification
- Evaluated logic: `DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT`. 
- **Proof**: Since days are measured relative to application (negative), a scheduled day of `-30` and an entry day of `-25` (5 days late) evaluates as `-25 > -30` which is `True`. 
- An automated unit test confirms this logic strictly bounds "lateness".

## 4. Thin-File Specific Findings
- **Size**: 44,020 borrowers (`BUREAU_CREDIT_COUNT = 0`).
- **Default Rate**: 10.12% (Thin-file) vs 7.73% (Non-thin-file). Thin-file borrowers represent elevated credit risk.
- **Alternative Data Coverage**: Out of 44,020 true thin-file borrowers, **41,550** (>94%) possess Alternative Behavioural data (e.g., prior loan interactions). This is exceptionally good news for testing our core hypothesis.

## 5. Feature Groups Summary
| Group | Number of features | Missingness | Main purpose |
|---|---:|---:|---|
| Application | 10 | Med/High (EXT_SOURCES) | Current application information |
| Traditional Credit | 6 | 0 | Bureau history |
| Alternative Behavioural | 7 | 0 | Internal behavioural/payment history |

## Visualizations Generated
Several exploratory charts were generated in `reports/figures/`:
1. `target_distribution.png`: Extreme class imbalance (~8% default).
2. `correlation_heatmap.png`: Highlighting the correlation between partial and late payments.
3. `thin_file_default_rate.png`: Clearly showing the elevated risk of thin-file applicants.
