# Phase 8: Controlled Linear Feature-Group Ablation

## Audit and experiment controls

Feature SQL aggregates bureau, previous-application, and installment rows by borrower before joining the application table, avoiding join fan-out. Predictors exclude `TARGET` and `SK_ID_CURR`. The original two-stage stratified split is frozen in `data/processed/splits/{train,validation,test}_ids.csv`; every later phase must load these IDs rather than generate a new split. The artifacts contain identifiers only, have zero overlap, and their union is all 307,511 modelling applicants.

The audit found 2,905 of 13,605,401 raw installment rows have both `DAYS_ENTRY_PAYMENT` and `AMT_PAYMENT` missing. The source dictionary describes those fields as when/how much was actually paid, but does not explain nulls. Their scheduled due dates precede the current application; this alone does not establish whether they were unpaid or absent for another reason. The corrected SQL excludes these events from observed late/underpayment measures and counts them separately as `INST_PAYMENT_MISSING_COUNT`. `INST_LATE_PAYMENT_RATIO` is late events divided by rows with observed payment and scheduled dates. If a borrower has installment rows but none with observed payment dates, the ratio remains missing and is imputed using training data. This adds one justified non-bureau behavioural feature to the original 23-feature group (24 predictors total). Superseded benchmark plots were removed during final packaging; the corrected result is authoritative.

The feature table was regenerated. Model A and B results remain unchanged; all three models were rerun. Preprocessing is fitted on train only. Application external-source features use training means and missingness indicators; other linear-model inputs use training medians and standardization. `APP_DAYS_EMPLOYED == 365243` is treated as missing. Bureau ratio missingness uses training median imputation and a training-fitted indicator. No calibration, threshold selection, or boosting was used in this phase.

## Feature groups and split sizes

| Model | Predictors |
|---|---|
| A — Application | `APP_INCOME_TOTAL`, `APP_CREDIT_AMOUNT`, `APP_ANNUITY`, `APP_DAYS_BIRTH`, `APP_DAYS_EMPLOYED`, `APP_EXT_SOURCE_1`, `APP_EXT_SOURCE_2`, `APP_EXT_SOURCE_3`, `APP_CREDIT_INCOME_RATIO`, `APP_ANNUITY_INCOME_RATIO` |
| B — Application + Bureau | Model A plus `BUREAU_CREDIT_COUNT`, `BUREAU_ACTIVE_COUNT`, `BUREAU_TOTAL_CREDIT`, `BUREAU_TOTAL_DEBT`, `BUREAU_DEBT_RATIO`, `BUREAU_AVG_DAYS_CREDIT` |
| C — Application + Bureau + Alternative / non-bureau behaviour | Model B plus `PREV_APP_COUNT`, `PREV_APPROVED_COUNT`, `PREV_REFUSED_COUNT`, `INST_TOTAL_COUNT`, `INST_LATE_PAYMENT_COUNT`, `INST_LATE_PAYMENT_RATIO`, `INST_UNDERPAYMENT_COUNT`, `INST_PAYMENT_MISSING_COUNT` |

Train: **215,257**; validation: **46,127**; test: **46,127**. The split files retain the Phase 7A borrower assignment and order.

## Overall performance

Every model uses `LogisticRegression(C=1.0, class_weight='balanced')` with the same preprocessing. ROC-AUC and PR-AUC (average precision) measure ranking. Brier and log loss are reported, but the class-weighted raw probabilities are not calibrated default probabilities.

| Split | Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| Train | A — Application | 0.726838 | 0.202485 | 0.211059 | 0.610679 |
| Train | B — + Bureau | 0.729488 | 0.206388 | 0.210186 | 0.608573 |
| Train | C — + Non-bureau behaviour | 0.738823 | 0.214728 | 0.206752 | 0.601658 |
| Validation | A — Application | 0.731575 | 0.214745 | 0.210390 | 0.609052 |
| Validation | B — + Bureau | 0.734062 | 0.218716 | 0.209609 | 0.607154 |
| Validation | C — + Non-bureau behaviour | 0.742421 | 0.222855 | 0.206402 | 0.600722 |
| Test | A — Application | **0.729418** | **0.208596** | 0.210476 | 0.609418 |
| Test | B — + Bureau | **0.732424** | **0.212172** | 0.209441 | 0.607029 |
| Test | C — + Non-bureau behaviour | **0.741447** | **0.221676** | 0.205648 | 0.599153 |

| Test comparison | ROC-AUC change | Relative change | PR-AUC change | Relative change |
|---|---:|---:|---:|---:|
| Bureau added to A | +0.003006 | +0.41% | +0.003577 | +1.71% |
| Non-bureau behaviour added to B | **+0.009023** | **+1.23%** | **+0.009504** | **+4.48%** |

The refit Phase 7A model gives 0.729418 ROC-AUC and 0.208596 PR-AUC, matching its reported 0.7294 and 0.2085 to the stated precision. Compared with the frozen pre-missingness benchmark (Model C 0.741084 / 0.219708 overall; 0.708819 / 0.232135 thin-file), corrected Model C is +0.000363 ROC-AUC and +0.001968 PR-AUC overall, and +0.001181 / +0.005817 in thin-file. Models A and B are unchanged. The conclusion is unchanged and slightly stronger after correcting the treatment of unknown installment events.

The Phase 8 test set has already been inspected in the previous benchmark, including subgroup, coefficient, redundancy, and bootstrap analyses. It is not an untouched test set. Results are frozen now; future model and calibration choices use train/validation only, with test reserved for predefined final comparisons.

## Thin-file and non-thin-file performance

Thin-file remains `BUREAU_CREDIT_COUNT == 0`, fixed before modelling. The full population has 44,020 thin-file applicants.

| Test cohort | n | Model A ROC / PR | Model B ROC / PR | Model C ROC / PR |
|---|---:|---:|---:|---:|
| Overall | 46,127 | 0.729418 / 0.208596 | 0.732424 / 0.212172 | 0.741447 / 0.221676 |
| Thin-file | 6,534 | 0.691025 / 0.211024 | 0.691050 / 0.211087 | **0.710000 / 0.237952** |
| Non-thin-file | 39,593 | 0.734491 / 0.209345 | 0.738019 / 0.214031 | 0.745645 / 0.219875 |

Bureau features barely change thin-file performance (ROC-AUC +0.000024, PR-AUC +0.000063). Adding non-bureau behavioural history to Model B improves thin-file ROC-AUC by **0.018951** and PR-AUC by **0.026865**. The gain is directionally positive on validation as well: Model C versus B thin-file ROC-AUC is 0.717217 versus 0.696906, and PR-AUC is 0.222163 versus 0.213772.

## Bootstrap uncertainty

Paired bootstrap intervals based on **2,000** test resamples for Model C minus Model B:

| Cohort | Metric | 95% percentile interval |
|---|---|---:|
| Overall | ROC-AUC | [0.00657, 0.01155] |
| Overall | PR-AUC | [0.00408, 0.01458] |
| Thin-file | ROC-AUC | [0.01074, 0.02750] |
| Thin-file | PR-AUC | [0.01002, 0.04379] |

The intervals describe resampling uncertainty within this historical holdout, not dataset or time-shift uncertainty.

## Non-bureau history coverage and zero interpretation

Of 44,020 thin-file applicants, 41,550 (**94.39%**) have previous-application history; 41,640 (**94.59%**) have installment history; and 41,780 (**94.91%**) have either. The previously quoted 94.4% refers to previous applications alone. Among thin-file applicants, 252 (**0.57%**) have installment history with at least one row missing actual payment information. The feature `INST_PAYMENT_MISSING_COUNT` marks these unknowns; it does not assert that they were unpaid.

`INST_TOTAL_COUNT` distinguishes no installment records from history with zero observed late events. Previous-application counts distinguish no application history from histories with zero approvals/refusals. Among thin-file applicants, 21,079 have installment history and zero recorded late events; 24,996 have history and zero recorded underpayments.

## Redundancy and coefficients

In the thin-file test group, `INST_LATE_PAYMENT_COUNT` and `INST_UNDERPAYMENT_COUNT` remain highly correlated (r = **0.911**); late count and late ratio correlate at 0.667. Leave-one-feature-out diagnostic refits found removing late count changed test ROC-AUC by -0.000056 / PR-AUC -0.000170, and removing underpayment count changed them by +0.000018 / +0.000019. Removing late ratio reduced ROC-AUC by 0.002327 and PR-AUC by 0.003833. The correlated counts add little measurable conditional value in this linear model, but are retained in the predefined feature set.

Largest Model C standardized associations:

| Feature | Coefficient | Odds ratio per standardized unit | Direction |
|---|---:|---:|---|
| `APP_EXT_SOURCE_2` | -0.426 | 0.653 | Higher score associated with lower predicted risk |
| `APP_EXT_SOURCE_3` | -0.392 | 0.676 | Higher score associated with lower predicted risk |
| `APP_EXT_SOURCE_1` | -0.250 | 0.779 | Higher score associated with lower predicted risk |
| `INST_LATE_PAYMENT_RATIO` | +0.211 | 1.235 | Higher observed late share associated with higher predicted risk |
| `PREV_APPROVED_COUNT` | -0.171 | 0.843 | More approvals associated with lower predicted risk |
| `BUREAU_DEBT_RATIO` | +0.134 | 1.144 | Higher debt ratio associated with higher predicted risk |

These are conditional associations, not causal effects. Correlated count features have small negative conditional coefficients despite adverse univariate meanings; this is a collinearity/suppression warning, not evidence of protection.

## Remaining technical risks and artifacts

The explicit split depends on the original Home Credit modeling population IDs; the loader fails if IDs are missing, duplicated, overlapping, or incomplete. Null payment reasons remain unspecified in source documentation, so the corrected representation records uncertainty rather than inferring nonpayment. Results from this single historical benchmark do not establish out-of-time stability, calibration, fairness, or business utility.

Tables are under `reports/tables/phase8_*.csv`; the selected final plots are in `reports/final/`. See [the final evaluation](final_results.md) for the consolidated results. This phase is the controlled linear benchmark for the nonlinear comparison.
