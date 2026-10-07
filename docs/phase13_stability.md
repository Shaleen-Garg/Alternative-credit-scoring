# Phase 13: Stability and responsible-modelling checks

## Holdout cohort performance

Phase 13 keeps the frozen Phase 10 HGB parameters and validation-fitted sigmoid mapping fixed. Model B is refit with the same HGB settings and separately sigmoid-calibrated on validation only as a diagnostic comparator. No stability result changes model selection.

| Test cohort | n | Model B ROC / PR | Model C ROC / PR | Model B Brier / log loss | Model C Brier / log loss |
|---|---:|---:|---:|---:|---:|
| Overall | 46,127 | 0.7490 / 0.2385 | **0.7579 / 0.2509** | 0.06813 / 0.24821 | **0.06754 / 0.24554** |
| Thin-file | 6,534 | 0.7179 / 0.2416 | **0.7376 / 0.2787** | 0.08455 / 0.29806 | **0.08266 / 0.29119** |
| Non-thin-file | 39,593 | 0.7528 / 0.2383 | **0.7602 / 0.2457** | 0.06542 / 0.23999 | **0.06504 / 0.23800** |

The alternative group improves these test ranking and scoring metrics over B in the three main cohorts. Segment results are not uniformly identical: for borrowers with recorded employment under one year (n=4,143), C ROC-AUC is 0.7389 versus B 0.7410 while C PR-AUC is 0.2820 versus B 0.2763. That is a subgroup difference to monitor, not a basis for further test-driven model changes.

Paired, outcome-stratified percentile bootstrap intervals for fixed test predictions (500 resamples; no refitting) quantify the Model C minus B ranking differences:

| Cohort | ROC-AUC delta (95% interval) | PR-AUC delta (95% interval) |
|---|---:|---:|
| Overall | +0.00890 [0.00623, 0.01175] | +0.01243 [0.00749, 0.01741] |
| Thin-file | +0.01970 [0.00973, 0.02933] | +0.03710 [0.01682, 0.05501] |
| Non-thin-file | +0.00736 [0.00470, 0.01039] | +0.00731 [0.00257, 0.01150] |

## Feature and score distribution stability

For five selected model features, train-decile PSI values on validation/test were all below **0.0003**. Medians and quantiles were also close between splits. Examples: `APP_EXT_SOURCE_3` missing rate was 19.91% train, 19.65% validation, and 19.60% test; `BUREAU_DEBT_RATIO` PSI was 0.00021 validation and 0.00020 test. PSI is a descriptive binned diagnostic, not a guarantee against all distribution shift.

The calibrated Model C test score distribution has mean 0.0806 and p99 0.4040 overall; thin-file mean is 0.0999 and p99 0.4113. Test maximum is 0.8145, 0.36% of scores exceed 0.5, and none are below 0.001 or above 0.99. Train/validation/test score means were 0.0806/0.0807/0.0806 overall. Train scores are in-sample and therefore not directly comparable with held-out distributions.

## Missingness and practical segments

`APP_EXT_SOURCE_1` missingness was 56.35%/56.18%/56.74% across train/validation/test; `APP_EXT_SOURCE_2` was 0.21%/0.22%/0.22%; `APP_EXT_SOURCE_3` was 19.91%/19.65%/19.60%. For test Model C, ROC-AUC was 0.7565 among rows missing `APP_EXT_SOURCE_1` (n=26,172) and 0.7583 among rows where present (n=19,955). For `APP_EXT_SOURCE_3`, ROC-AUC was 0.7336 when missing (n=9,040) and 0.7630 when present (n=37,087); PR-AUC was 0.2603 and 0.2490 respectively, reflecting different default rates. `APP_EXT_SOURCE_2` missingness has only 100 test cases; its metrics are too uncertain for interpretation.

Age bands and employment-duration bands were reviewed descriptively. Ranking was lower at age <25 (Model C ROC-AUC 0.6954, n=1,752) and age 55+ (0.7153, n=10,422) than in ages 25–54 (about 0.755–0.764). The sentinel-coded employment cohort had 8,492 test cases and ROC-AUC 0.7258. These sensitive-adjacent descriptive slices are not fairness tests; no protected-group variables or causal claims are used.

## Responsible-modelling observations and limitations

No extreme probability pile-up or large train/validation/test drift appeared in the selected features. Ranking still varies across cohorts, particularly age and some missing-score segments, and should be monitored if this were developed further. `INST_PAYMENT_MISSING_COUNT` has low global permutation importance (AP decrease 0.0005), but replacing it with the training median changed one selected high-risk thin-file example's PD by 0.141. This is a large local sensitivity for a rare feature; it may reflect a nonlinear local split or an unrealistic one-feature replacement and warrants targeted review before any real use. It is not evidence that missing payments cause default. Tiny slices (notably missing `APP_EXT_SOURCE_2`, n=100) have unstable estimates. This analysis does not establish fairness, out-of-time stability, representativeness, or production suitability. The dataset lacks a clean temporal evaluation design, and its historical holdout has already been inspected.

Tables, including the paired intervals, are `reports/tables/phase13_*.csv`; plots are `reports/figures/phase13_*.png`.
