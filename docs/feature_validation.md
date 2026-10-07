# Feature validation

The exported modeling table contains 307,511 rows at one row per SK_ID_CURR. TARGET is retained as the outcome column and excluded from predictors. Automated checks found no infinite values or constant predictors.

## Missingness and source coverage

- APP_EXT_SOURCE_1 is missing for about 56.3% of applicants; APP_EXT_SOURCE_3 for about 19.8%.
- APP_ANNUITY is missing for 12 applicants.
- External-score missingness is associated with different default rates in this dataset, so missingness is retained in preprocessing rather than treated as random.
- Source-history counts distinguish no historical record from a zero adverse-event count among applicants with history.

Among 44,020 thin-file applicants, 41,550 (94.39%) have previous-application history, 41,640 (94.59%) have installment history, and 41,780 (94.91%) have at least one of those histories. Coverage alone does not establish predictive value; the controlled model comparison evaluates incremental ranking performance.

## Anomalies and relationships

APP_DAYS_EMPLOYED contains the sentinel value 365243. It is converted to missing for HistGradientBoosting. Installment late-payment count and underpayment count are highly correlated in the thin-file population (Pearson r approximately 0.91); the feature groups were retained as predefined, and their conditional contribution was interpreted with that correlation in mind.

Installment lateness uses the sign convention DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 0. Tests verify this rule. Missing payment dates or amounts are not inferred to mean paid or unpaid; see feature_engineering.md for the treatment.

The detailed target-rate and distributional summaries are available in results.md.
