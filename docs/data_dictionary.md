# Data dictionary and relational structure

The project uses the historical Home Credit Default Risk dataset. The target-bearing modeling population contains 307,511 current applications. Source table sizes are approximately those reported in the competition files.

| Table | Approx. rows | Grain | Identifier and relationship |
|---|---:|---|---|
| application_train | 307,511 | One current application | SK_ID_CURR; includes TARGET |
| bureau | 1,716,428 | One external bureau record | SK_ID_BUREAU; linked to applicant by SK_ID_CURR |
| bureau_balance | 27,299,925 | Monthly status for a bureau record | Linked through SK_ID_BUREAU |
| previous_application | 1,670,214 | One prior application | SK_ID_PREV; linked to applicant by SK_ID_CURR |
| POS_CASH_balance | 10,001,358 | Historical POS/CASH account record | Linked logically through SK_ID_PREV |
| credit_card_balance | 3,840,312 | Historical credit-card account record | Linked logically through SK_ID_PREV |
| installments_payments | 13,605,401 | Installment/payment observation | Linked logically through SK_ID_PREV |

POS_CASH_balance, credit_card_balance, and installments_payments also carry SK_ID_CURR in the source files. Their logical relationship is to the prior account/application identified by SK_ID_PREV. bureau_balance belongs to a specific bureau record, not directly to the current application.

TARGET is the supplied payment-difficulty outcome in application_train. TARGET equals 1 for applicants meeting the competition's payment-difficulty definition and 0 otherwise. It is used as the label, never as an input feature.

The feature pipeline currently consumes application_train, bureau, previous_application, and installments_payments. The other listed files provide context about the wider competition dataset and are not required by the current feature builder.
