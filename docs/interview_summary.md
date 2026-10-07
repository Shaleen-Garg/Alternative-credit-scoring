# Interview summary

## One-minute project story

I evaluated whether internal repayment behaviour can improve default-risk ranking for borrowers with thin bureau files. Using the Home Credit Default Risk data, I built borrower-level aggregates from application, bureau, previous-loan, and installment tables in SQLite. I defined thin-file borrowers as applicants with no bureau records and froze one stratified train/validation/test split. A controlled logistic ablation compared application-only, application-plus-bureau, and application-plus-bureau-plus-alternative models. I then checked whether the result held under HistGradientBoosting, selected calibration using validation only, and completed illustrative decision, model-reliance, and stability analyses. Adding alternative history improved HGB ROC-AUC/PR-AUC by 0.0089/0.0124 overall and 0.0197/0.0371 for thin-file borrowers. The finding is encouraging but limited to a historical random holdout; it is not an out-of-time or fairness result and should not be presented as deployment evidence.

## Design choices worth explaining

- **Thin-file definition:** exactly zero bureau records, matching the absence of traditional credit history.
- **Controlled comparison:** same borrowers, split IDs, and core modeling setup for A/B/C; only feature groups change.
- **Leakage controls:** preprocessing is train-fitted; model selection and calibration use validation; frozen test results are reported after choices are fixed.
- **Missing installments:** rows with unobserved payment data are counted separately rather than assumed to be on-time or unpaid.
- **Nonlinear robustness:** a validation-selected HGB configuration preserves the direction of the feature-group gain.
- **Calibration and decisions:** sigmoid mapping and thresholds are validation-selected; assumed costs are normalized illustrations.
- **Explanations:** permutation and one-feature median-replacement diagnostics are model-behaviour measures, not causal explanations.

## Follow-up work before real-world use

Obtain a time-based, institution-specific dataset; validate point-in-time feature availability; test out-of-time and external performance; conduct legally reviewed fairness and adverse-action analyses; estimate real loss and applicant-utility trade-offs; monitor drift and calibration; and establish governance, security, and human review. Any future comparison should be pre-registered to avoid repeated tuning on the inspected holdout.
