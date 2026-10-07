# Leakage Audit

## Point of Prediction
The prediction point is exactly the time of the current loan application (`application_train`). All temporal information is measured relative to this event (Day 0). 

## Available Information Validation
A thorough audit of temporal indicators confirms the point-in-time snapshot integrity of the dataset. All historical events legitimately occurred *before* the current application:
- `bureau`: `DAYS_CREDIT` (all <= 0) and `DAYS_ENDDATE_FACT` (all <= 0) confirm no bureau records occurring after the application date are included. Positive values in `DAYS_CREDIT_ENDDATE` represent expected future end dates at the time of application, which is legitimate information.
- `previous_application`: `DAYS_DECISION` (all <= 0) confirms all past internal applications were processed before the current application.
- `installments_payments`: `DAYS_INSTALMENT` and `DAYS_ENTRY_PAYMENT` (all <= 0) confirm no future payments were leaked.
- `POS_CASH_balance` & `credit_card_balance`: `MONTHS_BALANCE` (all <= 0).

## Feature Eligibility Framework

| Data source | Feature concept | Available before application? | Potential leakage? | Decision |
|---|---|---:|---:|---|
| bureau | historical credit count | Yes | Low | Include |
| installments | historical payment delay | Yes | Low | Include |
| previous_application | historical approval/refusal | Yes | Low | Include |
| POS_CASH | historical DPD | Yes | Low | Include |
| credit_card | historical utilization | Yes | Low | Include |
| application | EXT_SOURCE_1/2/3 | Yes (Assumed) | Medium (if scores update post-app) | Include |

*Note on EXT_SOURCE: External scores are generally point-in-time scores retrieved at application. However, we must remain vigilant that they do not perfectly predict `TARGET`, which would imply post-application behavioral leakage.*
