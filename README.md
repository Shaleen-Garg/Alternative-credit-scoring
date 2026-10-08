# Alternative Credit Scoring for Thin-File Borrowers

> Evaluating whether alternative behavioural history improves default-risk prediction when traditional bureau history is limited.

This project builds an applicant-level feature table from Home Credit application and historical account data. It tests whether previous application and installment-payment behaviour adds predictive signal beyond application and bureau variables, with particular focus on applicants with no bureau records.

The target is the dataset's payment-difficulty outcome (TARGET = 1 for payment difficulty). The pipeline uses Python, SQLite and SQL, pandas, NumPy, scikit-learn, Matplotlib, and pytest.

**Adding alternative behavioural history improved HGB ROC-AUC from 0.7490 to 0.7579 overall and from 0.7179 to 0.7376 among thin-file applicants.**

![Overall model comparison](reports/results/model_comparison.png)

## Problem and hypothesis

Traditional bureau history can be sparse or absent for some applicants. Those applicants may still have prior applications and observed repayment behaviour in the historical data. The project tests whether those non-bureau signals improve default-risk ranking beyond current application and bureau information.

The source is the historical Home Credit Default Risk competition dataset. The target-bearing application_train file contains 307,511 applicants. This is not current banking data and does not represent a specific lender's customer population.

Thin-file is defined as BUREAU_CREDIT_COUNT = 0: no bureau records at the current application point. This definition was set before the main model comparison and identifies 44,020 applicants. Among them, 41,550 have previous-application history, 41,640 have installment history, and 41,780 have either.

## Data architecture

The source dataset is relational. The diagram shows logical relationships across its historical tables. The current feature builder uses application_train, bureau, previous_application, and installments_payments. The other competition tables are included here to clarify the full data structure, but are not consumed by this implementation.

Source CSVs are imported in chunks and historical applicant keys are indexed before aggregation. The build checks target values, one-row-per-applicant output, numeric predictors, finite values, and relative dates; any positive event date stops export as a potential post-application leakage path.

~~~mermaid
erDiagram
    APPLICATION ||--o{ BUREAU : "SK_ID_CURR"
    BUREAU ||--o{ BUREAU_BALANCE : "SK_ID_BUREAU"
    APPLICATION ||--o{ PREVIOUS_APPLICATION : "SK_ID_CURR"
    PREVIOUS_APPLICATION ||--o{ POS_CASH_BALANCE : "SK_ID_PREV"
    PREVIOUS_APPLICATION ||--o{ CREDIT_CARD_BALANCE : "SK_ID_PREV"
    PREVIOUS_APPLICATION ||--o{ INSTALLMENTS_PAYMENTS : "SK_ID_PREV"

    APPLICATION {
        int SK_ID_CURR
        int TARGET
    }
    BUREAU {
        int SK_ID_BUREAU
        int SK_ID_CURR
    }
    BUREAU_BALANCE {
        int SK_ID_BUREAU
    }
    PREVIOUS_APPLICATION {
        int SK_ID_PREV
        int SK_ID_CURR
    }
    POS_CASH_BALANCE {
        int SK_ID_PREV
    }
    CREDIT_CARD_BALANCE {
        int SK_ID_PREV
    }
    INSTALLMENTS_PAYMENTS {
        int SK_ID_PREV
        int SK_ID_CURR
    }
~~~

POS_CASH_balance, credit_card_balance, and installments_payments also contain SK_ID_CURR in the source files. Their logical historical relationship is through SK_ID_PREV, which identifies a prior application. bureau_balance is linked to a bureau record through SK_ID_BUREAU.

| Source table | Approx. rows | Grain and role |
|---|---:|---|
| application_train | 307,511 | One current application; includes TARGET |
| bureau | 1,716,428 | One external bureau record |
| bureau_balance | 27,299,925 | Monthly status record for a bureau record |
| previous_application | 1,670,214 | One historical application |
| POS_CASH_balance | 10,001,358 | Historical POS/CASH account record |
| credit_card_balance | 3,840,312 | Historical credit-card account record |
| installments_payments | 13,605,401 | Historical installment/payment observation |

The pipeline aggregates historical rows before joining them to the current application table. Grouping by applicant or prior application first avoids one-to-many join fan-out and preserves one modeling row per applicant.

The raw files are not included in this repository and must be obtained separately. The competition source files are application_train.csv, bureau.csv, bureau_balance.csv, previous_application.csv, POS_CASH_balance.csv, credit_card_balance.csv, installments_payments.csv, and HomeCredit_columns_description.csv. Put them in data/raw/. The current feature builder requires application_train.csv, bureau.csv, previous_application.csv, and installments_payments.csv; the remaining files provide dataset context but are not consumed by the current pipeline. See [data setup](data/README.md).

## Feature groups

The controlled comparison adds feature groups in sequence:

~~~text
A  Application
   ↓
B  Application + Traditional Bureau
   ↓
C  Application + Traditional Bureau + Alternative Behaviour
~~~

The same borrowers, frozen split, and core preprocessing are used across A/B/C. The feature groups are the controlled change.

| Group | Examples | Features |
|---|---|---:|
| Application | Income, credit amount, annuity, age, employment duration, three external scores, credit/income and annuity/income ratios | 10 |
| Traditional bureau | Credit and active counts, total credit and debt, debt ratio, average credit recency | 6 |
| Alternative behaviour | Previous/approved/refused application counts; installment volume, observed late and underpayment counts, late-payment ratio, missing-payment-information count | 8 |

PREV_APPROVED_COUNT summarizes prior application outcomes. INST_LATE_PAYMENT_RATIO measures observed late installments relative to records with observed payment and scheduled dates. Both contribute meaningful non-application signal in the fitted model.

### Unspecified installment missingness

In 2,905 of 13,605,401 raw installment rows, both DAYS_ENTRY_PAYMENT and AMT_PAYMENT are missing. The source does not explain what those records mean. They are excluded from observed late- and underpayment counts and represented separately by INST_PAYMENT_MISSING_COUNT. The feature design does not infer that the missing events were unpaid or on time.

## Modelling and evaluation

The target is the supplied Home Credit payment-difficulty outcome. TARGET and SK_ID_CURR are not predictors. The borrower-level modeling table is split into fixed, stratified train/validation/test sets:

| Split | Applicants |
|---|---:|
| Train | 215,257 |
| Validation | 46,127 |
| Test | 46,127 |

The 70/15/15 split IDs are committed under data/processed/splits/ and reused across analyses. Preprocessing is fitted using training data. Logistic Regression is the controlled linear comparison; HistGradientBoosting (HGB) checks whether the feature-group result holds with a nonlinear model. The selected HGB configuration is chosen using validation data and shared across A/B/C.

Calibration uses a sigmoid mapping selected on validation predictions and outcomes. Decision thresholds are also selected on validation. Test outcomes are used for final reporting and predefined diagnostics, not model, calibration-method, or threshold selection. The historical test split had been inspected during earlier benchmarking; it is not an untouched confirmatory test.

The selected calibration mapping is refit on all validation predictions before threshold tuning. Validation calibration diagnostics and threshold costs are descriptive for a mapping fit on that same split; the test split remains excluded from model and calibrator fitting and threshold selection.

~~~mermaid
flowchart TD
    A[Home Credit source tables] --> B[SQLite relational aggregation]
    B --> C[Applicant-level feature table]
    C --> D[Thin-file cohort definition]
    D --> E[Frozen train / validation / test IDs]
    E --> F[Controlled A/B/C comparison]
    F --> G[Logistic Regression]
    F --> H[HistGradientBoosting]
    H --> I[Validation-selected calibration]
    I --> J[Illustrative decision analysis]
    H --> K[Model-reliance explanations]
    H --> L[Cohort and stability diagnostics]
    J --> M[Results]
    K --> M
    L --> M
~~~

## Results

### Overall historical holdout

| Feature group | Logistic ROC-AUC | Logistic PR-AUC | HGB ROC-AUC | HGB PR-AUC |
|---|---:|---:|---:|---:|
| Application | 0.7294 | 0.2086 | 0.7446 | 0.2302 |
| Application + Bureau | 0.7324 | 0.2122 | 0.7490 | 0.2385 |
| Application + Bureau + Alternative | **0.7414** | **0.2217** | **0.7579** | **0.2509** |

### Thin-file historical holdout

| Feature group | Logistic ROC-AUC | Logistic PR-AUC | HGB ROC-AUC | HGB PR-AUC |
|---|---:|---:|---:|---:|
| Application | 0.6910 | 0.2110 | 0.7202 | 0.2425 |
| Application + Bureau | 0.6910 | 0.2111 | 0.7179 | 0.2416 |
| Application + Bureau + Alternative | **0.7100** | **0.2380** | **0.7376** | **0.2787** |

![Thin-file model comparison](reports/results/thin_file_comparison.png)

For HGB, Model C minus Model B gains were:

| Cohort | ROC-AUC gain | PR-AUC gain |
|---|---:|---:|
| Overall | +0.0089 | +0.0124 |
| Thin-file | +0.0197 | +0.0371 |

Paired, outcome-stratified bootstrap intervals from 500 resamples of fixed predictions:

| Cohort | ROC-AUC gain (95% interval) | PR-AUC gain (95% interval) |
|---|---:|---:|
| Overall | +0.0089 [0.0062, 0.0118] | +0.0124 [0.0075, 0.0174] |
| Thin-file | +0.0197 [0.0097, 0.0293] | +0.0371 [0.0168, 0.0550] |

These intervals describe resampling uncertainty within this historical holdout. They do not establish out-of-time stability or generalization to another institution.

## Calibration and decision analysis

For the selected HGB Model C, sigmoid calibration preserves ranking (ROC-AUC 0.757904; PR-AUC 0.250944). Calibration improvement is modest and mixed rather than perfect.

| Cohort | Measure | Raw | Sigmoid |
|---|---|---:|---:|
| Overall | Brier score | 0.067541 | 0.067537 |
| Overall | Log loss | 0.245562 | 0.245538 |
| Overall | 15-bin ECE | 0.002111 | 0.001263 |
| Thin-file | Brier score | 0.082692 | 0.082662 |
| Thin-file | Log loss | 0.291252 | 0.291187 |
| Thin-file | 15-bin ECE | 0.005440 | 0.006581 |

![Calibration reliability](reports/results/calibration_reliability.png)

At an illustrative 5:1 false-negative:false-positive normalized cost ratio, validation selected an overall threshold of 0.1497 (about 0.15). On the historical holdout, Model C approval was 85.97%, default recall 41.94%, defaults among approved 5.45%, and normalized cost 0.3408 per applicant. Model B at the same cutoff had 86.21% approval, 41.41% default recall, 5.49% defaults among approved, and 0.3410 normalized cost per applicant. These values are illustrative, not lending policy, bank economics, or underwriting recommendations.

At validation-selected cutoffs targeting approximately 70% approval, thin-file results were:

| Model | Holdout approval | Default recall | Defaults among approved |
|---|---:|---:|---:|
| Application + Bureau | 69.39% | 59.03% | 5.96% |
| Application + Bureau + Alternative | 69.71% | 61.61% | 5.55% |

![Illustrative decision trade-off](reports/results/decision_tradeoff.png)

## Model interpretation and stability

Validation permutation importance is led by application external scores. Previous approval count and installment late-payment ratio contribute alternative behavioural signal; BUREAU_DEBT_RATIO is an important bureau feature.

| Feature | Validation average-precision decrease |
|---|---:|
| APP_EXT_SOURCE_2 | 0.0637 |
| APP_EXT_SOURCE_3 | 0.0541 |
| APP_EXT_SOURCE_1 | 0.0278 |
| APP_DAYS_BIRTH | 0.0132 |
| APP_DAYS_EMPLOYED | 0.0114 |
| PREV_APPROVED_COUNT | 0.0095 |
| INST_LATE_PAYMENT_RATIO | 0.0089 |

Permutation importance measures model reliance, not causality. Local explanation diagnostics use one-feature-at-a-time replacement with the training median; they are not SHAP values and are not automatically suitable as regulatory adverse-action reasons.

Train-decile PSI for selected features was below 0.0003 on validation and test. Cohort performance varies, and estimates for small slices are uncertain.

![Validation feature importance](reports/results/feature_importance.png)

![Prediction stability](reports/results/prediction_stability.png)

## Reproducibility

Use Python 3.10 and install requirements from requirements.txt. Put the competition files listed above in data/raw/. From the repository root, run:

~~~bash
python -m pip install -r requirements.txt
python -m src.features
python -m src.ablation
python -m src.nonlinear
python -m src.calibration_analysis
python -m src.decision_analysis
python -m src.explainability
python -m src.stability_analysis
python -m src.release
python -m pytest -q
~~~

The feature builder writes a local SQLite database and data/processed/feature_master.csv. Detailed intermediate outputs are generated under reports/tables/ and reports/figures/ and are not tracked. Curated tables and figures are stored under reports/results/. The test suite last passed with 30 tests. See [reproduction details](docs/reproducibility.md).

## Repository structure

~~~text
.
├── data/
│   ├── raw/                     # Local source data; only .gitkeep is committed
│   ├── interim/                 # Local database; only .gitkeep is committed
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── splits/
│   │       ├── README.md
│   │       ├── train_ids.csv
│   │       ├── validation_ids.csv
│   │       └── test_ids.csv
│   └── README.md
├── docs/
│   ├── data_dictionary.md
│   ├── feature_engineering.md
│   ├── feature_validation.md
│   ├── methodology.md
│   ├── results.md
│   └── reproducibility.md
├── reports/
│   └── results/
│       ├── calibration_reliability.png
│       ├── calibration_summary.csv
│       ├── decision_at_70pct_approval.csv
│       ├── decision_tradeoff.png
│       ├── feature_importance.png
│       ├── model_comparison.csv
│       ├── model_comparison.png
│       ├── paired_bootstrap_intervals.csv
│       ├── prediction_stability.png
│       ├── test_cohort_comparison.csv
│       ├── thin_file_comparison.png
│       └── thin_file_model_comparison.csv
├── sql/
│   ├── README.md
│   ├── schema.sql
│   ├── feature_queries.sql
│   └── data_quality.sql
├── src/
│   ├── ablation.py
│   ├── calibration.py
│   ├── calibration_analysis.py
│   ├── data.py
│   ├── decision_analysis.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── features.py
│   ├── frozen_model.py
│   ├── models.py
│   ├── nonlinear.py
│   ├── release.py
│   ├── stability_analysis.py
├── tests/
│   ├── test_calibration.py
│   ├── test_decisions.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_validation.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── environment.yml
~~~

## Limitations

- The data is historical and comes from one competition setting.
- The evaluation is a random holdout, not an out-of-time study; the holdout was inspected during earlier benchmarking.
- No independent external validation, fairness certification, causal finding, or production monitoring system is provided.
- Decision costs are illustrative and do not represent real lending economics or applicant utility.
- The meaning of some missing installment observations is unspecified by the source.
- The results do not establish legal sufficiency or production readiness.

## Conclusion

The controlled comparison provides evidence that alternative behavioural history adds predictive signal beyond application and bureau information in this dataset, with the largest HGB ranking gains among applicants with no bureau records. These results do not establish performance over time, at another institution, or in production.

**Technical documentation:** [Results](docs/results.md) · [Methodology](docs/methodology.md) · [Feature engineering](docs/feature_engineering.md) · [Feature validation](docs/feature_validation.md) · [Data dictionary](docs/data_dictionary.md) · [Reproducibility](docs/reproducibility.md)
