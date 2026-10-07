# Phase 8: Controlled Feature-Group Ablation

## Audit and experiment controls

The repository already had a point-in-time feature plan, a one-row-per-application master table, explicit exclusion of target and application ID from predictors, and a deterministic two-stage stratified split. Feature SQL aggregates bureau, previous-application, and installment rows by borrower before joining them to the application table, avoiding join fan-out. The 7A pipeline fits imputation and scaling within the training pipeline. No split index file had been saved, so this work reproduces the original 70/15/15 `train_test_split` calls with seed 42; the refit application model reproduces the Phase 7A test ROC-AUC to the published precision.

The Phase 7B notebook was still a TODO and modeling code only exposed the application-only list. I centralized the 10 application, 6 bureau, and 7 alternative feature lists, retained the original preprocessing and Logistic Regression setup, then reused that pipeline for all three models. No target/ID columns are accepted as predictors. `DAYS_EMPLOYED == 365243` is changed to missing before training-only median imputation. External-source means and missingness indicators are learned within each model's training fit. Other numerical predictors use training-only medians and standard scaling. No winsorization, calibration, thresholds, or boosting were introduced.

## Feature groups and split sizes

| Model | Predictors |
|---|---|
| A — Application | `APP_INCOME_TOTAL`, `APP_CREDIT_AMOUNT`, `APP_ANNUITY`, `APP_DAYS_BIRTH`, `APP_DAYS_EMPLOYED`, `APP_EXT_SOURCE_1`, `APP_EXT_SOURCE_2`, `APP_EXT_SOURCE_3`, `APP_CREDIT_INCOME_RATIO`, `APP_ANNUITY_INCOME_RATIO` |
| B — Application + Bureau | Model A plus `BUREAU_CREDIT_COUNT`, `BUREAU_ACTIVE_COUNT`, `BUREAU_TOTAL_CREDIT`, `BUREAU_TOTAL_DEBT`, `BUREAU_DEBT_RATIO`, `BUREAU_AVG_DAYS_CREDIT` |
| C — Application + Bureau + Alternative | Model B plus `PREV_APP_COUNT`, `PREV_APPROVED_COUNT`, `PREV_REFUSED_COUNT`, `INST_TOTAL_COUNT`, `INST_LATE_PAYMENT_COUNT`, `INST_LATE_PAYMENT_RATIO`, `INST_UNDERPAYMENT_COUNT` |

The exact 70/15/15 row counts are train **215,257**, validation **46,127**, and test **46,127**. The split was regenerated from the same dataset row order, target, two-stage stratification, and seed used by Phase 7A.

## Overall performance

All three models use `LogisticRegression(C=1.0, class_weight='balanced')` with the same preprocessing. ROC-AUC and average precision (PR-AUC) are ranking metrics; Brier and log loss are included as requested, but raw probabilities from this class-weighted model are not calibrated default probabilities.

| Split | Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| Train | A — Application | 0.726838 | 0.202485 | 0.211059 | 0.610679 |
| Train | B — + Bureau | 0.729488 | 0.206388 | 0.210186 | 0.608573 |
| Train | C — + Alternative | 0.738578 | 0.214124 | 0.206888 | 0.601847 |
| Validation | A — Application | 0.731575 | 0.214745 | 0.210390 | 0.609052 |
| Validation | B — + Bureau | 0.734062 | 0.218716 | 0.209609 | 0.607154 |
| Validation | C — + Alternative | 0.741816 | 0.222412 | 0.206594 | 0.601123 |
| Test | A — Application | **0.729418** | **0.208596** | 0.210476 | 0.609418 |
| Test | B — + Bureau | **0.732424** | **0.212172** | 0.209441 | 0.607029 |
| Test | C — + Alternative | **0.741084** | **0.219708** | 0.205795 | 0.599472 |

| Test comparison | ROC-AUC change | Relative change | PR-AUC change | Relative change |
|---|---:|---:|---:|---:|
| Bureau added to A | +0.003006 | +0.41% | +0.003577 | +1.71% |
| Alternative added to B | **+0.008660** | **+1.18%** | **+0.007536** | **+3.55%** |

The original Phase 7A report gave 0.7294 ROC-AUC and 0.2085 PR-AUC; this refit gives 0.729418 and 0.208596, consistent at the stated ROC precision and differing by 0.0001 in PR-AUC at four decimal places. Bureau information yields a modest population-wide improvement. Alternative features add a larger, consistent ranking gain on validation and test, but the gain is still an incremental result on one random holdout, not evidence of causal impact or production value.

## Thin-file and non-thin-file results

Thin-file remains the prespecified definition `BUREAU_CREDIT_COUNT == 0`; it was not changed after looking at outcomes. The full population contains 44,020 thin-file records. Holdout subgroup metrics are:

| Test cohort | n | Model A ROC / PR | Model B ROC / PR | Model C ROC / PR |
|---|---:|---:|---:|---:|
| Overall | 46,127 | 0.729418 / 0.208596 | 0.732424 / 0.212172 | 0.741084 / 0.219708 |
| Thin-file | 6,534 | 0.691025 / 0.211024 | 0.691050 / 0.211087 | **0.708819 / 0.232135** |
| Non-thin-file | 39,593 | 0.734491 / 0.209345 | 0.738019 / 0.214031 | 0.745402 / 0.218604 |

On thin-file test borrowers, bureau features barely change the score (ROC-AUC +0.000024; PR-AUC +0.000063), as expected when bureau history is absent. Adding alternative history to B improves thin-file ROC-AUC by **0.017769** and PR-AUC by **0.021048**. The direction is also positive on validation: Model C versus B changes thin-file ROC-AUC from 0.696906 to 0.715807 and PR-AUC from 0.213772 to 0.220168. The result supports incremental ranking signal for this cohort in this dataset; it does not prove generalization to other lenders or borrowers.

Paired bootstrap intervals (300 resamples, test set) for Model C minus Model B were:

| Cohort | Metric | 95% percentile interval |
|---|---|---:|
| Overall | ROC-AUC | [0.00623, 0.01114] |
| Overall | PR-AUC | [0.00166, 0.01258] |
| Thin-file | ROC-AUC | [0.00996, 0.02524] |
| Thin-file | PR-AUC | [0.00328, 0.03791] |

These intervals describe resampling uncertainty on this held-out sample, not model or dataset shift uncertainty.

## Alternative-history coverage and zero interpretation

Recomputing on the current feature master gives 41,550 of 44,020 thin-file borrowers (**94.39%**) with previous-application history. Installment history exists for 41,640 (**94.59%**). Either previous-application or installment history exists for 41,780 (**94.91%**). Thus 94.4% corresponds to previous applications alone; combining the two included alternative sources yields 94.9% coverage. “Not thin-file” is not used as a proxy for alternative-history coverage.

Among the 44,020 thin-file borrowers, 21,079 have installment history and zero recorded late-payment events, while 2,380 have no installment rows. For underpayment, 24,996 have installment history and zero recorded underpayment events; 2,380 have no installment rows. `INST_TOTAL_COUNT` differentiates these cases in the model's fixed feature set. Previous-application counts similarly distinguish absent history from histories with zero approvals or refusals. No extra history-presence features were added because the core experiment's 23-feature set is locked.

## Redundancy and coefficients

Within the thin-file test group, Pearson correlation is **0.911** between `INST_LATE_PAYMENT_COUNT` and `INST_UNDERPAYMENT_COUNT`; late count and late ratio correlate at 0.667. A diagnostic leave-one-feature-out fit (not used to select the core Model C) found removing late-payment count changed test ROC-AUC by -0.000030 and PR-AUC by -0.000183; removing underpayment count changed them by +0.000018 and +0.000030. Removing late-payment ratio reduced ROC-AUC by 0.002238 and PR-AUC by 0.003698. The two highly correlated counts add little measurable conditional ranking value in this linear setup; both remain in the mandated core feature set, with no post-hoc cherry-picking.

Largest standardized Model C associations on the fitted model:

| Feature | Coefficient | Odds ratio per standardized unit | Direction |
|---|---:|---:|---|
| `APP_EXT_SOURCE_2` | -0.426 | 0.653 | Higher score associated with lower predicted risk |
| `APP_EXT_SOURCE_3` | -0.392 | 0.676 | Higher score associated with lower predicted risk |
| `APP_EXT_SOURCE_1` | -0.250 | 0.778 | Higher score associated with lower predicted risk |
| `INST_LATE_PAYMENT_RATIO` | +0.209 | 1.233 | Higher late-payment share associated with higher predicted risk |
| `PREV_APPROVED_COUNT` | -0.171 | 0.842 | More approvals associated with lower predicted risk |
| `BUREAU_DEBT_RATIO` | +0.134 | 1.143 | Higher debt ratio associated with higher predicted risk |
| `BUREAU_ACTIVE_COUNT` | +0.131 | 1.140 | More active accounts associated with higher predicted risk |

These are conditional associations, not causal effects. Counts and rates are correlated; notably `INST_LATE_PAYMENT_COUNT` (-0.026) and `INST_UNDERPAYMENT_COUNT` (-0.012) have small negative conditional coefficients despite adverse univariate meanings. This is a suppression/collinearity warning, not evidence that adverse payment events are protective. The external scores remain the largest coefficients.

## Audit findings and remaining risks

Solid foundations: borrower-level aggregation before joins; one-row-per-application feature master; locked thin-file definition; deterministic stratification; application-only model with documented results; target and ID exclusion; and train-fitted imputation/scaling.

Changes in this phase: centralized the 23 predictors and model-building pipeline; implemented Phase 7B and A/B/C evaluation, subgroup and coverage tables, bootstrap deltas, coefficient and redundancy diagnostics; replaced the Phase 8 notebook TODO with the reproducible runner; updated README and added Phase 7B/8 reports; and added synthetic tests for feature membership, leakage exclusion, exact split reproduction, train-only statistics, finite transforms, and probabilities.

The audit also found that bureau SQL was coercing an undefined debt ratio to zero for people who did have bureau records. The SQL now preserves that value as missing, keeps zero for borrowers with no bureau rows, and the bureau preprocessing fits a median and missingness indicator on training data. There are 8,337 undefined bureau ratios, including 1,083 with a zero/NULL total-credit denominator; the corrected feature table was regenerated and split borrower IDs were verified identical to the pre-fix split.

Remaining technical risks: no split index artifact was saved, so exactness depends on preserving the original row order and scikit-learn split method (verified here). The SQL installment logic treats a missing `DAYS_ENTRY_PAYMENT` as not late and missing `AMT_PAYMENT` as not underpaid; the database has 2,905 such rows among 13,605,401 installment rows. This is rare (0.021%) but its behavioral meaning should be explicitly decided before using these features in a later production-oriented study. The dataset is a single historical benchmark; validation/test results do not establish out-of-time stability, probability calibration, fairness, or business utility.

## Artifacts and next phase

Machine-readable tables are in `reports/tables/phase8_*.csv`; plots are in `reports/figures/phase8_*.png`. The report, notebook, and these generated outputs are committed. The next step should be to review these results, resolve the installment-missing-payment interpretation, and only then design a controlled nonlinear robustness comparison. Do not treat this ablation as a reason to skip temporal validation or calibration.
