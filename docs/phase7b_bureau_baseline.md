# Phase 7B: Application + Traditional Credit

## Experiment design

Phase 7B adds only six bureau aggregates to the ten Phase 7A application predictors: `BUREAU_CREDIT_COUNT`, `BUREAU_ACTIVE_COUNT`, `BUREAU_TOTAL_CREDIT`, `BUREAU_TOTAL_DEBT`, `BUREAU_DEBT_RATIO`, and `BUREAU_AVG_DAYS_CREDIT`. The model reuses the exact two-stage stratified 70/15/15 split from Phase 7A (seed 42), train-fitted preprocessing, and `LogisticRegression(C=1.0, class_weight='balanced')`.

Rows: train 215,257; validation 46,127; test 46,127. Application-only was refit through the same shared pipeline as a reproducibility check and returned test ROC-AUC 0.729418 and PR-AUC 0.208596, consistent with the published Phase 7A values of 0.7294 and 0.2085 (small PR-AUC rounding difference).

## Results

| Split | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Train | 0.729488 | 0.206388 | 0.210186 | 0.608573 |
| Validation | 0.734062 | 0.218716 | 0.209609 | 0.607154 |
| Test | 0.732424 | 0.212172 | 0.209441 | 0.607029 |

Compared with Application-only on test, adding bureau features changed ROC-AUC by +0.003006 (+0.41% relative) and PR-AUC by +0.003577 (+1.71% relative). This is a modest improvement overall. Brier score and log loss are reported descriptively; class weighting shifts raw probabilities, so these are not calibrated default probabilities.

| Test population | n | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Overall | 46,127 | 0.732424 | 0.212172 |
| Thin-file (`BUREAU_CREDIT_COUNT == 0`) | 6,534 | 0.691050 | 0.211087 |
| Non-thin-file | 39,593 | 0.738019 | 0.214031 |

For thin-file borrowers, bureau history is structurally absent; the bureau model is effectively unchanged versus application-only (ROC-AUC 0.691025 to 0.691055; PR-AUC 0.211024 to 0.211091). This phase does not test alternative data.

See `reports/tables/phase8_split_metrics.csv` and `reports/tables/phase8_cohort_metrics.csv` for the full A/B/C comparisons.
