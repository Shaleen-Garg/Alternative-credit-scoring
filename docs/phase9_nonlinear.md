# Phase 9: Nonlinear robustness comparison

## Design

HistGradientBoosting was compared with the Phase 8 logistic models on the frozen train, validation, and test IDs. The three feature groups match Phase 8. A small candidate grid was ranked by validation PR-AUC on Model C; the selected parameters were then held fixed across all three groups. Missing values are supported natively by the estimator, and the employment sentinel is converted to missing. No test results selected parameters. Permutation importance and two interaction diagnostics use validation data only.

Selected parameters: `learning_rate=0.08`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `l2_regularization=2.0` (`early_stopping=False`, `random_state=42`).

## Test performance

| Feature group | Logistic ROC-AUC / PR-AUC | HGB ROC-AUC / PR-AUC |
|---|---:|---:|
| Application | 0.729418 / 0.208596 | 0.744624 / 0.230231 |
| Application + Bureau | 0.732424 / 0.212172 | 0.749005 / 0.238513 |
| Application + Bureau + Alternative | 0.741447 / 0.221676 | **0.757904 / 0.250944** |

For HGB, adding alternative history over Model B improves test ROC-AUC by **0.008899** and PR-AUC by **0.012431**. On the thin-file test cohort (n=6,534), Model C reaches 0.737647 ROC-AUC and 0.278670 PR-AUC, changes of +0.019703 and +0.037095 over Model B. This supports the direction of the linear ablation under a nonlinear learner; it remains a single historical holdout, not independent temporal validation.

Validation permutation importance is led by application external scores, followed by age/employment and previous approvals; installment late-payment ratio and bureau debt ratio also contribute. These rankings describe this fitted model and are not causal effects. Validation interaction plots are diagnostic, not evidence of causal interaction.

## Artifacts

Candidate scores, runtime, test/validation metrics, cohort metrics, permutation importance, and linear-vs-nonlinear comparisons are in `reports/tables/phase9_*.csv` and `.json`; plots are in `reports/figures/phase9_*.png`.
