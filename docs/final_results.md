# Final evaluation

## Scope and frozen evaluation

This report summarizes the final Phase 8–13 evaluation on the project's fixed random 70/15/15 borrower split. There are 215,257 training, 46,127 validation, and 46,127 test borrowers. The thin-file definition is `BUREAU_CREDIT_COUNT == 0`; the frozen test cohort contains 6,534 borrowers. The test split had already been inspected during the earlier benchmark, so results are treated as a final historical holdout evaluation, not an untouched confirmatory test.

Model A uses application variables, Model B adds traditional bureau information, and Model C adds internal alternative behavioural history. Logistic Regression provides the controlled feature-group ablation. HistGradientBoosting (HGB) is the nonlinear robustness comparison. The full machine-readable table is [`model_comparison.csv`](../reports/final/model_comparison.csv); thin-file metrics are in [`thin_file_model_comparison.csv`](../reports/final/thin_file_model_comparison.csv).

| Feature group | Logistic overall ROC-AUC / PR-AUC | HGB overall ROC-AUC / PR-AUC | Logistic thin-file ROC-AUC / PR-AUC | HGB thin-file ROC-AUC / PR-AUC |
|---|---:|---:|---:|---:|
| Application | 0.7294 / 0.2086 | 0.7446 / 0.2302 | 0.6910 / 0.2110 | 0.7202 / 0.2425 |
| Application + Bureau | 0.7324 / 0.2122 | 0.7490 / 0.2385 | 0.6910 / 0.2111 | 0.7179 / 0.2416 |
| Application + Bureau + Alternative | **0.7414 / 0.2217** | **0.7579 / 0.2509** | **0.7100 / 0.2380** | **0.7376 / 0.2787** |

HGB Model C improves over Model B by 0.0089 ROC-AUC and 0.0124 PR-AUC overall, and by 0.0197 and 0.0371 on the thin-file cohort. The paired, outcome-stratified bootstrap (500 resamples on fixed predictions) gives 95% intervals of [0.0062, 0.0118] and [0.0075, 0.0174] overall, and [0.0097, 0.0293] and [0.0168, 0.0550] for thin-file ROC-AUC and PR-AUC gains. See [`paired_bootstrap_intervals.csv`](../reports/final/paired_bootstrap_intervals.csv).

![Model comparison](../reports/final/model_comparison.png)

![Thin-file comparison](../reports/final/thin_file_comparison.png)

## Calibration and reconciliation

The selected HGB Model C uses `learning_rate=0.08`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=100`, and `l2_regularization=2.0`; early stopping is disabled and the random state is 42. The model is fit on train only. A sigmoid mapping is selected and fit using validation predictions and outcomes. Test outcomes are used only for final reporting.

The Phase 9/10 audit verifies that the regenerated feature table and ordered split IDs are recorded, all three frozen ID sets have their expected row counts, and the uncalibrated Phase 10 predictions exactly reproduce Phase 9: test ROC-AUC 0.757904 and PR-AUC 0.250944. HGB's `APP_DAYS_EMPLOYED == 365243` sentinel is converted to missing for both phases. The audit records `test_metrics_match_phase9=true` and `test_labels_used_for_any_selection_or_fit=false`.

| Cohort | Mapping | ROC-AUC | PR-AUC | Brier | Log loss | 15-bin ECE |
|---|---|---:|---:|---:|---:|---:|
| Overall | Raw | 0.757904 | 0.250944 | 0.067541 | 0.245562 | 0.002111 |
| Overall | Sigmoid | 0.757904 | 0.250944 | **0.067537** | **0.245538** | **0.001263** |
| Thin-file | Raw | 0.737647 | 0.278670 | 0.082692 | 0.291252 | 0.005440 |
| Thin-file | Sigmoid | 0.737647 | 0.278670 | **0.082662** | **0.291187** | 0.006581 |

Sigmoid preserves ranking, slightly improves overall calibration scores, and slightly improves thin-file Brier/log loss while thin-file ECE rises. Calibration is therefore mixed and modest, not perfect. Reliability bins and the full metrics are in [`calibration_summary.csv`](../reports/final/calibration_summary.csv).

![Reliability diagrams](../reports/final/calibration_reliability.png)

## Decision analysis

The decision exercise assigns normalized costs to false negatives and false positives and chooses cutoffs on validation. At an illustrative 5:1 false-negative:false-positive ratio, the overall validation cutoff is 0.15. On test, HGB Model C approves 86.03% with 41.78% default recall, 5.46% defaults among approved, and normalized cost 0.3409 per applicant; Model B at the same cutoff gives 86.26%, 41.27%, 5.50%, and 0.3411. The overall difference is small.

At cutoffs independently chosen on validation to target 70% approval, realized thin-file test approval is 69.39% for B and 69.71% for C. Default recall is 59.03% and 61.61%, respectively; defaults among approved are 5.96% and 5.55%. These are illustrative comparisons under arbitrary cost assumptions. They are not bank economics, underwriting policy, or lending recommendations. Full rows are in [`decision_at_70pct_approval.csv`](../reports/final/decision_at_70pct_approval.csv).

![Decision operating points](../reports/final/decision_tradeoff.png)

## Explanations and stability

Validation permutation importance is led by the three application external scores. Previous approvals, installment late-payment ratio, and bureau debt ratio also contribute to this fitted model. Group permutation shows additional reliance on alternative behavioural variables, but the group effects are not additive because the feature groups can be correlated. Importance is model reliance, not causality.

Local explanation examples use one-feature-at-a-time replacement with the training median; they are not SHAP values, can create unrealistic feature combinations, and are not automatically suitable regulatory adverse-action reasons. A notable sensitivity remains: `INST_PAYMENT_MISSING_COUNT` has low global permutation importance but a large effect in one selected thin-file case. This may reflect local tree structure or the replacement method and warrants review before any real use.

Across selected features, train-decile PSI was below 0.0003 on validation and test. Paired bootstrap intervals for Model C minus B were positive in the overall, thin-file, and non-thin-file cohorts. Metrics vary by age and missing-score cohorts; these are descriptive slices, not fairness tests. The random split cannot establish out-of-time performance. Train score distributions are in-sample and should not be compared directly with held-out scores.

![Validation feature importance](../reports/final/feature_importance.png)

![Prediction stability](../reports/final/prediction_stability.png)

## Limitations and conclusion

- The data comes from one historical Home Credit competition dataset and has limited transferability to another lender, market, or time period.
- The random holdout was previously inspected; there is no independent out-of-time evaluation.
- No fairness, legal sufficiency, causal effect, applicant utility, or production operations assessment is established.
- The normalized decision costs omit revenue, exposure, capital, recoveries, and real policy constraints.
- Model performance varies across some practical and sensitive-adjacent cohorts; small cohorts have uncertain estimates.

Within those limits, the controlled ablation and nonlinear robustness results support incremental predictive signal from alternative behavioural history, with the clearest ranking gain among thin-file borrowers. This is a research result, not evidence that the model should be deployed.
