# Phase 12: Practical model explanations

## Global and group importance

The explanations use the frozen HGB Model C selected in Phase 10. Its individual permutation importance is calculated on validation data using average precision (three repeats; full table in `phase12_validation_feature_importance.csv`). Top features are `APP_EXT_SOURCE_2` (0.0637 AP decrease), `APP_EXT_SOURCE_3` (0.0541), `APP_EXT_SOURCE_1` (0.0278), `APP_DAYS_BIRTH` (0.0132), `APP_DAYS_EMPLOYED` (0.0114), `PREV_APPROVED_COUNT` (0.0095), and `INST_LATE_PAYMENT_RATIO` (0.0089).

A separate joint group permutation repeats the same row permutation across all columns in a group, retaining within-group combinations while disrupting that group's alignment with the target. Mean validation AP decreases were Application **0.1506**, Alternative Behaviour **0.0228**, and Traditional Bureau **0.0121**; corresponding ROC-AUC decreases were 0.1967, 0.0192, and 0.0122. This indicates greater reliance on application information in this model and additional reliance on alternative history. Group effects are not an additive percentage decomposition: correlated groups and interactions prevent that interpretation.

Alternative features with the largest individual AP decreases were `PREV_APPROVED_COUNT` (0.0095), `INST_LATE_PAYMENT_RATIO` (0.0089), and `PREV_REFUSED_COUNT` (0.0053). `INST_TOTAL_COUNT` had a smaller decrease (0.0026). `INST_PAYMENT_MISSING_COUNT` was small (0.0005); `PREV_APP_COUNT` was near zero (0.0002); late count and underpayment count were slightly negative at this permutation resolution, meaning no measurable added ranking value in this run. These are fitted-model reliance diagnostics, not causal effects.

## Representative test borrowers and reason codes

Selection rule was fixed before examining explanations: within thin-file and non-thin-file groups, take the borrower nearest the median predicted probability in the top decile, and the borrower nearest the median in the bottom decile. The rule uses scores and cohort only, not outcomes or feature contributions. IDs are retained to make the examples reproducible.

| Cohort | Risk band | Test ID | Sigmoid PD |
|---|---|---:|---:|
| Thin-file | High | 262556 | 0.2738 |
| Thin-file | Low | 116198 | 0.0200 |
| Non-thin-file | High | 449485 | 0.2455 |
| Non-thin-file | Low | 399990 | 0.0139 |

Local reasons use one-feature-at-a-time replacement with the training median, then measure the change in this borrower's fitted-model PD. For thin-file high-risk ID 262556, replacing `INST_PAYMENT_MISSING_COUNT` with its training median lowered PD by 0.1415; replacing `APP_EXT_SOURCE_3` lowered it by 0.1288; replacing `INST_LATE_PAYMENT_RATIO` lowered it by 0.0969. In the same case, median replacement of `APP_DAYS_EMPLOYED` raised PD by 0.1991 and replacement of `PREV_APPROVED_COUNT` raised PD by 0.0234. For non-thin high-risk ID 449485, the largest risk-increasing local comparisons were `APP_EXT_SOURCE_3` (+0.1261), `APP_EXT_SOURCE_1` (+0.0875), and `BUREAU_DEBT_RATIO` (+0.0455) when each was replaced by its training median. Low-risk examples had lower PD contributions from the external scores relative to the median reference.

The generated human-readable reason codes state that a feature raises or lowers this fitted model's score relative to a median replacement. They are local perturbation comparisons, not additive SHAP values; they can create unrealistic combinations and do not establish why a borrower defaults. The same explanation procedure and all feature changes are in `reports/tables/phase12_*.csv`.

## Limits

Importance is specific to this fitted model and validation sample. Correlated inputs can split or mask importance. Neither permutation importance nor local replacement effects establish causality. These experimental reason codes are not automatically regulatory adverse-action reasons and have not been assessed for legal sufficiency.
