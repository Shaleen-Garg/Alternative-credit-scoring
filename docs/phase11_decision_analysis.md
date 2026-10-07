# Phase 11: Illustrative decision and cost analysis

## Method and assumptions

This analysis uses the selected Phase 10 HGB Model C and its validation-fitted sigmoid calibrator. The base estimator is fit on train only; sigmoid is fit on validation only. Threshold candidates from 0.01 to 0.50 in 0.01 increments were evaluated on validation. The threshold minimizing validation normalized cost was then applied unchanged to test. The primary threshold is selected overall; validation-selected cohort-specific thresholds are shown as subgroup diagnostics, not a proposed policy.

Costs are illustrative normalized units, not Ujjivan economics: approving a borrower who defaults costs `C_FN`; rejecting a borrower who would repay costs `C_FP`. Four relative assumptions were considered: FN:FP = 1:1, 2:1, 5:1, and 10:1. Per applicant cost is `(FN × C_FN + FP × C_FP) / n`. No actual bank policy or monetary loss is inferred. A default is the positive class: TP is a default rejected; FN is a default approved; FP is a good borrower rejected; TN is a good borrower approved.

## Validation-selected thresholds

| FN:FP cost ratio | Overall threshold | Thin-file diagnostic | Non-thin diagnostic |
|---|---:|---:|---:|
| 1:1 | 0.49 | 0.49 | 0.49 |
| 2:1 | 0.34 | 0.33 | 0.34 |
| 5:1 | 0.15 | 0.19 | 0.15 |
| 10:1 | 0.08 | 0.10 | 0.08 |

Under the illustrative 5:1 ratio, for example, validation selected threshold 0.15 overall; its validation normalized cost was 0.3388 per applicant. This states the result under an assumption, not a “correct” lending threshold.

## Frozen-test evaluation

At the overall validation-selected threshold for each cost ratio, Model B and C use the same cutoff. At 5:1, threshold 0.15 gives Model C 86.03% test approval, 41.78% default recall, 5.46% defaults among approved, and 0.3409 normalized cost per applicant. Model B gives 86.26%, 41.27%, 5.50%, and 0.3411. This is a small overall decision-value difference.

At a threshold selected separately for each cohort from validation, the 5:1 thin-file diagnostic (threshold 0.19) gives Model C 87.47% approval, 34.90% recall, 7.51% defaults among approved, and 0.4184 cost units per applicant. Model B at the same cutoff gives 88.31%, 31.71%, 7.80%, and 0.4293. The non-thin-file result is not uniformly better: at its 5:1 diagnostic threshold 0.15, Model C's test normalized cost is 0.3279 versus 0.3264 for Model B. Thus the thin-file improvement is clearer than the overall or non-thin cost difference.

### Comparison at similar approval volume

For this separate diagnostic, each model's cutoff was selected on validation to target 50%, 70%, or 90% approval; the resulting cutoffs were evaluated on test. This compares operating points at approximately the same volume without using test data to select thresholds.

At a 70% target, Model C versus B on test:

| Cohort | Realized approval B / C | Default recall B / C | Default rate among approved B / C |
|---|---:|---:|---:|
| Overall | 70.01% / 70.07% | 63.80% / 65.12% | 4.17% / 4.02% |
| Thin-file | 69.39% / 69.71% | 59.03% / 61.61% | 5.96% / 5.55% |
| Non-thin-file | 69.81% / 70.12% | 64.63% / 65.35% | 3.92% / 3.83% |

The thin-file test cohort has 6,534 people; differences are descriptive for this holdout. Figures show the threshold trade-offs and approval-volume/risk curve. Full confusion counts and every tested threshold are in `reports/tables/phase11_*.csv`.

## Limits

These are controlled illustrations under arbitrary relative cost assumptions, not expected profits, underwriting policy, or lending recommendations. Costs omit revenue, capital, exposure, recoveries, fairness constraints, and applicant utility. Test results are from a previously inspected historical holdout and do not establish time stability or production value.
