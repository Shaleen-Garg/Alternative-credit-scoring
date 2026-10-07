# Phase 10: Probability calibration

## Selection and fitting

The base model was selected from the Phase 8 and Phase 9 train/validation summaries by rank-aggregating validation overall and thin-file ROC-AUC/PR-AUC, accounting for the train-validation ROC gap, and using model complexity as a tie-break. This selected HistGradientBoosting Model C (24 features). Sigmoid and isotonic mappings were fit using half of validation predictions and labels; the method was selected on the other stratified half by log loss, then Brier score. The selected sigmoid mapping was refit on the full validation set. Test labels were not used to fit the model, fit a calibrator, or choose a calibration method.

### Phase 9/10 reconciliation

The first Phase 10 run differed from Phase 9 because the reconstructed Phase 10 HGB model passed `APP_DAYS_EMPLOYED == 365243` through unchanged. Phase 9 converts that sentinel to missing before both fitting and prediction, allowing HGB's native missing-value handling. Train/validation/test ID files, feature table, 24-feature list and order, target extraction and split row order, HGB parameters (`learning_rate=0.08`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `l2_regularization=2.0`, `early_stopping=False`, `random_state=42`), and train-only fit procedure otherwise matched. Phase 10 intentionally reconstructs and refits the deterministic model on the same training IDs rather than loading a serialized Phase 9 estimator; calibration requires validation predictions and no model artifact had been persisted. Aligning the sentinel transformation resolved the discrepancy without changing Phase 9. The corrected uncalibrated Phase 10 test metrics exactly equal Phase 9: **0.757904 ROC-AUC and 0.250944 PR-AUC**.

Calibration leakage audit: the base estimator is fit only on train. Half of validation fits candidate calibrators and the other half selects the mapping. The selected sigmoid is refit on all validation predictions/labels. Test labels enter only the final metrics and reliability-bin calculations. No test labels are used for model selection, calibrator fitting, calibration-method selection, or threshold selection; Phase 10 performs no threshold selection.

## Final test calibration metrics

| Cohort | Method | ROC-AUC | PR-AUC | Brier | Log loss | Equal-width ECE |
|---|---|---:|---:|---:|---:|---:|
| Overall | Uncalibrated | 0.757904 | 0.250944 | 0.067541 | 0.245562 | 0.002111 |
| Overall | Sigmoid (selected) | 0.757904 | 0.250944 | **0.067537** | **0.245538** | **0.001263** |
| Overall | Isotonic | 0.757427 | 0.239422 | 0.067587 | 0.245672 | 0.002514 |
| Thin-file | Uncalibrated | 0.737647 | 0.278670 | 0.082692 | 0.291252 | 0.005440 |
| Thin-file | Sigmoid (selected) | 0.737647 | 0.278670 | **0.082662** | **0.291187** | 0.006581 |
| Thin-file | Isotonic | 0.737155 | 0.265092 | 0.082743 | 0.291623 | 0.006688 |

Sigmoid calibration preserves ranking metrics exactly, as expected for this increasing monotone mapping. Overall Brier, log loss, and equal-width expected calibration error improve slightly. Thin-file Brier and log loss improve slightly; ECE rises from 0.005440 to 0.006581, a small deterioration. This is a mixed but modest calibration change, not a claim of perfect calibration. The reported ECE uses one fixed 15-bin scheme and does not establish calibration across time or populations. The overall test default rate is 8.07%; thin-file is 10.09%.

### Calibration intercept and slope

| Cohort | Probabilities | Intercept | Slope |
|---|---|---:|---:|
| Overall | Raw | +0.061013 | 1.029170 |
| Overall | Sigmoid | +0.011543 | 1.004769 |
| Thin-file | Raw | +0.124146 | 1.057035 |
| Thin-file | Sigmoid | +0.073336 | 1.031972 |

Both diagnostics move closer to intercept 0 and slope 1 after sigmoid calibration; the remaining thin-file intercept is still positive. These are descriptive recalibration diagnostics on the historical holdout, not proof of exact calibration.

The exact Phase 9/10 feature-table fingerprint, ordered split-ID fingerprints, feature order, and final metric equality are recorded in `reports/tables/phase10_reconciliation_audit.json`. Reliability-bin data and validation method-selection metrics are in `reports/tables/phase10_*.csv`, model-selection details in `phase10_selected_model.json`, and plots in `reports/figures/phase10_*.png`. See `reproducibility.md` for the module sequence.
