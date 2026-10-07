# Phase 10: Probability calibration

## Selection and fitting

The base model was selected from the Phase 8 and Phase 9 train/validation summaries by rank-aggregating validation overall and thin-file ROC-AUC/PR-AUC, accounting for the train-validation ROC gap, and using model complexity as a tie-break. This selected HistGradientBoosting Model C (24 features). Sigmoid and isotonic mappings were fit using half of validation predictions and labels; the method was selected on the other stratified half by log loss, then Brier score. The selected sigmoid mapping was refit on the full validation set. Test labels were not used to fit the model, fit a calibrator, or choose a calibration method.

## Final test calibration metrics

| Cohort | Method | ROC-AUC | PR-AUC | Brier | Log loss | Equal-width ECE |
|---|---|---:|---:|---:|---:|---:|
| Overall | Uncalibrated | 0.757473 | 0.248892 | 0.067620 | 0.245805 | 0.002345 |
| Overall | Sigmoid (selected) | 0.757473 | 0.248892 | **0.067620** | **0.245791** | **0.001969** |
| Overall | Isotonic | 0.756989 | 0.239278 | 0.067706 | 0.246116 | 0.002119 |
| Thin-file | Uncalibrated | 0.737095 | 0.276602 | 0.082734 | 0.291482 | 0.004784 |
| Thin-file | Sigmoid (selected) | 0.737095 | 0.276602 | **0.082711** | **0.291439** | **0.004209** |
| Thin-file | Isotonic | 0.735879 | 0.263387 | 0.082749 | 0.291954 | 0.004743 |

Sigmoid calibration preserves ranking metrics, as expected for a monotone mapping, and gives small improvements in Brier/log loss and equal-width expected calibration error. The improvement is modest. The reported calibration diagnostics use one fixed 15-bin scheme; they should not be treated as proof of calibration across time or populations. The overall test default rate is 8.07%; thin-file is 10.09%.

Reliability-bin data and validation method-selection metrics are in `reports/tables/phase10_*.csv`, model-selection details in `phase10_selected_model.json`, and plots in `reports/figures/phase10_*.png`.
