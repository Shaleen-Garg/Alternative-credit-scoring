# Data Dictionary Notes

## 1. Dataset Dimensions
- **application_train.csv**: 307,511 rows, 122 columns
- **bureau.csv**: 1,716,428 rows, 17 columns
- **bureau_balance.csv**: 27,299,925 rows, 3 columns
- **previous_application.csv**: 1,670,214 rows, 37 columns
- **POS_CASH_balance.csv**: 10,001,358 rows, 8 columns
- **credit_card_balance.csv**: 3,840,312 rows, 23 columns
- **installments_payments.csv**: 13,605,401 rows, 8 columns

## 2. Table Relationships Map
```text
application (SK_ID_CURR)
  ├── bureau (SK_ID_CURR)
  │    └── bureau_balance (SK_ID_BUREAU)
  │
  └── previous_application (SK_ID_CURR, SK_ID_PREV)
       ├── POS_CASH_balance (SK_ID_PREV)
       ├── credit_card_balance (SK_ID_PREV)
       └── installments_payments (SK_ID_PREV)
```
*Note: POS, credit card, and installment tables also contain `SK_ID_CURR` for direct joining, but logically they belong to specific previous internal loans (`SK_ID_PREV`).*

## 3. Target Definition
- **TARGET**: Found in `application_train.csv`. 
- **1** = Client with payment difficulties (had late payment more than X days on at least one of the first Y installments).
- **0** = All other cases (good loans).

## 4. Available Temporal Information
Most temporal features are measured in days or months *relative to the application date* (denoted as 0).
- **Application**: `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`.
- **Bureau**: `DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `DAYS_CREDIT_UPDATE`.
- **Balances**: `MONTHS_BALANCE` (bureau, POS, credit card).
- **Previous Applications**: `DAYS_DECISION`, `DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE`, `DAYS_TERMINATION`.
- **Installments**: `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`.

## 5. Potential Behavioural / Financial Information
- **Traditional Credit Info**: `bureau` and `bureau_balance` contain credit history from other financial institutions.
- **Alternative / Internal Behavioural Info**: 
  - `previous_application`: behavior on past applications (approved/rejected).
  - `installments_payments`: actual payment behaviour vs expected (delays, underpayments).
  - `POS_CASH_balance` & `credit_card_balance`: monthly usage, drawings, DPD (Days Past Due) on previous internal accounts.
- **External Scores**: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` in application.

## 6. Potential Leakage Risks
- All `DAYS_` variables must be strictly negative (occurred before application). Positive days (e.g. `DAYS_CREDIT_ENDDATE` > 0) are valid if they represent a future expected date at the time of application.
- Ensure no post-application data is joined. (The dataset documentation implies it is a point-in-time snapshot, but this must be verified during EDA).
- External sources (`EXT_SOURCE_X`) might contain future target information if not strictly point-in-time.

## 7. Thin-File Definition
Following a rigorous audit of the training population (307,511 borrowers), we have formalized our primary thin-file definition based strictly on the absence of traditional credit bureau records (`bureau.csv`).

- **Selected Definition**: No traditional credit history (`bureau_count == 0`)
  - **Population Size**: 44,020 borrowers (14.3% of training population)
  - **Default Rate**: 10.12% (vs 7.73% for non-thin-file)
  - **Justification**: This definition explicitly captures the core research question. These borrowers have ZERO traditional credit history available. Importantly, over 94% of this group (41,550 borrowers) *do* possess alternative internal history (previous applications, POS usage, etc.), making them the perfect cohort to evaluate the incremental predictive power of alternative behavioral data.

## 8. Suitability for Project
The dataset is **highly suitable**. It explicitly separates traditional credit bureau data from alternative behavioral data (internal past loan performance, granular payment histories, etc.). This allows us to cleanly split the population into traditional vs. thin-file borrowers and measure the incremental predictive power of behavioral features.
