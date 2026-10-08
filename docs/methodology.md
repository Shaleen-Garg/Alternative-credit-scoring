# Evaluation methodology

## Prediction point and population

The prediction point is the current application. The modelling population is the 307,511 applicants in application_train.csv, with TARGET as the supplied outcome. Thin-file applicants are defined as those with BUREAU_CREDIT_COUNT equal to zero (44,020 applicants). This definition was fixed before the main model comparison.

## Data and feature preparation

Historical bureau, previous-application, and installment records are summarized at applicant level before joining to the current application table. The model table has one row per SK_ID_CURR. TARGET and SK_ID_CURR are excluded from predictor columns. The feature build checks bureau, prior-decision, installment-due, and payment dates and stops if any positive relative-day values occur after the current application.

The feature groups are nested:

| Model | Predictors |
|---|---|
| A | 10 application features |
| B | A plus 6 traditional bureau features |
| C | B plus 8 alternative behavioural features |

The same frozen borrower IDs are used for all comparisons. Preprocessing parameters are fit on training data. Logistic Regression uses the same class-weighted configuration across the three feature groups.

## Data split

| Split | Borrowers |
|---|---:|
| Train | 215,257 |
| Validation | 46,127 |
| Test | 46,127 |

The split is stratified 70/15/15 at borrower level. Ordered IDs are stored in data/processed/splits/ and checked for duplicates, overlap, and full population coverage.

## Model selection and evaluation

HistGradientBoosting is a nonlinear robustness comparison. Its small candidate set is evaluated using validation performance, and one configuration is applied consistently to feature groups A, B, and C. The selected configuration is learning_rate 0.08, max_iter 200, max_leaf_nodes 15, min_samples_leaf 100, and l2_regularization 2.0; early stopping is disabled and random_state is 42. The employment sentinel 365243 is converted to missing before fitting and prediction.

The calibration analysis selects a model from training and validation metrics, selects a calibration method on a held-out half of validation, then refits the selected mapping on all validation predictions and outcomes. Decision thresholds are selected on those calibrated validation predictions under explicit illustrative cost ratios or approval-volume targets. Cost minimization checks each distinct validation score boundary and an all-approve cutoff. Validation calibration diagnostics and threshold-cost curves are descriptive for a calibrator fit on the same split. Test outcomes are not used to fit the model or calibrator or to select thresholds.

## Interpretation of the holdout

The test split is a final historical holdout, but it was inspected during earlier benchmarking. It is not an untouched confirmatory test. Bootstrap intervals quantify resampling uncertainty within this holdout only. No out-of-time or external evaluation is available.
