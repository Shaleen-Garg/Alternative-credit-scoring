# Feature engineering

The feature pipeline uses SQL and SQLite to aggregate historical records before joining them to the current application table. This avoids row multiplication from one-to-many histories and produces one modeling row per applicant.

## Feature groups

| Group | Features |
|---|---|
| Application (10) | Income, credit amount, annuity, age, employment duration, three external scores, credit-to-income ratio, annuity-to-income ratio |
| Traditional bureau (6) | Bureau credit count, active count, total credit, total debt, debt ratio, average credit recency |
| Alternative behaviour (8) | Previous-application count, approved count, refused count, installment count, late-payment count, late-payment ratio, underpayment count, missing-payment-information count |

The model comparison adds the groups in order: application; application plus bureau; then application plus bureau and alternative behavioural history. The full modeled feature names are defined in src/models.py.

## Historical aggregations

- Bureau records are grouped by SK_ID_CURR to derive counts, amounts, debt ratio, and average credit recency.
- Previous applications are grouped by SK_ID_CURR to derive total, approved, and refused application counts.
- Installment observations are grouped by SK_ID_CURR to derive installment volume and observed late- and underpayment counts.
- These summaries are joined to application_train at applicant grain.
- The source's relative-day fields are checked before export. Positive bureau, prior-decision, installment-due, or payment dates stop the build because they occur after the current application. DAYS_DECISION is loaded for this check only; it is not a model feature.

## Installment observation rules

Payment delay is DAYS_ENTRY_PAYMENT minus DAYS_INSTALMENT. An observed payment is late when this difference is positive. Underpayment is counted only when both the paid amount and scheduled installment amount are observed.

In 2,905 of 13,605,401 raw installment rows, both DAYS_ENTRY_PAYMENT and AMT_PAYMENT are missing. The source does not specify what these missing values mean. Those rows are excluded from observed late- and underpayment counts and represented by INST_PAYMENT_MISSING_COUNT; they are not presumed to be unpaid or on time. That feature counts any installment row missing either DAYS_ENTRY_PAYMENT or AMT_PAYMENT.

INST_LATE_PAYMENT_RATIO is calculated over rows with observed payment and scheduled dates. When installment history exists but no payment date is observed, the ratio stays missing and is imputed from training data by the model preprocessing. Counts of history distinguish no records from history with no observed adverse event.

## Missing values and special codes

- Counts for applicants with no source history are zero.
- BUREAU_DEBT_RATIO is zero when no bureau record exists; an undefined ratio for an applicant with bureau records remains missing for train-fitted imputation.
- APP_DAYS_EMPLOYED value 365243 is treated as missing for the tree model.
- External score missingness is retained for preprocessing and is not replaced using validation or test data.

The implementation is in sql/schema.sql, sql/feature_queries.sql, and src/features.py.
