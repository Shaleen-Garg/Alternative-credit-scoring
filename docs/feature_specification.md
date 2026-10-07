# Feature Specification

This document details the selected features for the credit risk model, categorized into three primary groups. The total feature count is intentionally focused (23 features) to prioritize interpretability and robustness.

## Group A — Application Features
Features directly available on the current application.

| Feature name | Source table | Source columns | Transformation | Temporal rule | Interpretation | Missing-value treatment | Expected direction (vs Default Risk/TARGET) | Leakage risk | Final status |
|---|---|---|---|---|---|---|---|---|---|
| `APP_INCOME_TOTAL` | application | `AMT_INCOME_TOTAL` | Identity | App time | Applicant's total income | Impute median | Negative | Low | Include |
| `APP_CREDIT_AMOUNT` | application | `AMT_CREDIT` | Identity | App time | Requested loan amount | Impute median | Positive | Low | Include |
| `APP_ANNUITY` | application | `AMT_ANNUITY` | Identity | App time | Expected loan annuity | Impute median | Positive | Low | Include |
| `APP_DAYS_BIRTH` | application | `DAYS_BIRTH` | Identity | App time | Age in days (negative) | None | Negative (older=safer, lower risk) | Low | Include |
| `APP_DAYS_EMPLOYED` | application | `DAYS_EMPLOYED` | Identity | App time | Tenure in days (negative) | Handle anomalies (365243) | Negative (longer=safer, lower risk) | Low | Include |
| `APP_EXT_SOURCE_1` | application | `EXT_SOURCE_1` | Identity | App time | External normalized score | Impute mean or leave for XGB | Negative | Medium | Include |
| `APP_EXT_SOURCE_2` | application | `EXT_SOURCE_2` | Identity | App time | External normalized score | Impute mean or leave for XGB | Negative | Medium | Include |
| `APP_EXT_SOURCE_3` | application | `EXT_SOURCE_3` | Identity | App time | External normalized score | Impute mean or leave for XGB | Negative | Medium | Include |
| `APP_CREDIT_INCOME_RATIO` | application | `AMT_CREDIT`, `AMT_INCOME_TOTAL` | `AMT_CREDIT / AMT_INCOME_TOTAL` | App time | Credit burden relative to income | Impute median | Positive | Low | Include |
| `APP_ANNUITY_INCOME_RATIO`| application | `AMT_ANNUITY`, `AMT_INCOME_TOTAL`| `AMT_ANNUITY / AMT_INCOME_TOTAL`| App time | Payment burden relative to income| Impute median | Positive | Low | Include |

## Group B — Traditional Credit Features
Features derived from external traditional credit history.

| Feature name | Source table | Source columns | Transformation | Temporal rule | Interpretation | Missing-value treatment | Expected direction (vs Default Risk/TARGET) | Leakage risk | Final status |
|---|---|---|---|---|---|---|---|---|---|
| `BUREAU_CREDIT_COUNT` | bureau | `SK_ID_BUREAU` | `COUNT(*)` per `SK_ID_CURR` | Past history | Number of past traditional loans | 0 (Thin-file) | Negative (history helps) | Low | Include |
| `BUREAU_ACTIVE_COUNT` | bureau | `CREDIT_ACTIVE` | `SUM(1)` where 'Active' | Point in time | Current active credit lines | 0 | Uncertain | Low | Include |
| `BUREAU_TOTAL_CREDIT` | bureau | `AMT_CREDIT_SUM` | `SUM(AMT_CREDIT_SUM)` | Point in time | Total historical credit extended | 0 | Uncertain | Low | Include |
| `BUREAU_TOTAL_DEBT` | bureau | `AMT_CREDIT_SUM_DEBT`| `SUM(AMT_CREDIT_SUM_DEBT)` | Point in time | Total current traditional debt | 0 | Positive | Low | Include |
| `BUREAU_DEBT_RATIO` | bureau | `AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM` | `TOTAL_DEBT / TOTAL_CREDIT` | Point in time | Utilization of traditional credit | 0 | Positive | Low | Include |
| `BUREAU_AVG_DAYS_CREDIT`| bureau | `DAYS_CREDIT` | `AVG(DAYS_CREDIT)` | Past history | Average recency of prior credit accounts, encoded as negative days relative to application | Impute max (e.g. 0) | Uncertain | Low | Include |

## Group C — Alternative Behavioural Features
Features derived from internal interactions and historical payment behaviours.

| Feature name | Source table | Source columns | Transformation | Temporal rule | Interpretation | Missing-value treatment | Expected direction (vs Default Risk/TARGET) | Leakage risk | Final status |
|---|---|---|---|---|---|---|---|---|---|
| `PREV_APP_COUNT` | prev_app | `SK_ID_PREV` | `COUNT(*)` per `SK_ID_CURR` | Past history | Number of past internal apps | 0 | Uncertain | Low | Include |
| `PREV_APPROVED_COUNT` | prev_app | `NAME_CONTRACT_STATUS` | `SUM(1)` where 'Approved' | Past history | Successful past internal apps | 0 | Negative (helps risk) | Low | Include |
| `PREV_REFUSED_COUNT` | prev_app | `NAME_CONTRACT_STATUS` | `SUM(1)` where 'Refused' | Past history | Rejected past internal apps | 0 | Positive | Low | Include |
| `INST_TOTAL_COUNT` | installments | `SK_ID_PREV` | `COUNT(*)` per `SK_ID_CURR` | Past history | Total number of installments paid | 0 | Negative (track record) | Low | Include |
| `INST_LATE_PAYMENT_COUNT`| installments | `DAYS_ENTRY_PAYMENT`, `DAYS_INSTALMENT` | `SUM(ENTRY > INSTALMENT)` | Past history | Count of delayed payments | 0 | Positive | Low | Include |
| `INST_LATE_PAYMENT_RATIO`| installments | (Derived above) | `LATE_COUNT / TOTAL_COUNT` | Past history | Frequency of late payments | 0 | Positive | Low | Include |
| `INST_UNDERPAYMENT_COUNT`| installments | `AMT_PAYMENT`, `AMT_INSTALMENT` | `SUM(AMT_PAYMENT < AMT_INSTALMENT)` | Past history | Count of partial payments | 0 | Positive | Low | Include |
