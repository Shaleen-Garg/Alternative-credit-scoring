# Alternative Credit Scoring for Thin-File Borrowers

This project tests whether prior bureau history and internal payment behaviour add useful default-risk signal beyond current application data, with a specific focus on borrowers who have no bureau record. It uses the Home Credit Default Risk dataset and a frozen, stratified borrower split.

## Main result

On the historical test split, adding alternative behavioural features improved ranking for both the logistic ablation and the nonlinear model. The largest measured gain is among thin-file borrowers.

| Feature group | Logistic overall ROC-AUC / PR-AUC | HGB overall ROC-AUC / PR-AUC | Logistic thin-file ROC-AUC / PR-AUC | HGB thin-file ROC-AUC / PR-AUC |
|---|---:|---:|---:|---:|
| Application | 0.7294 / 0.2086 | 0.7446 / 0.2302 | 0.6910 / 0.2110 | 0.7202 / 0.2425 |
| Application + Bureau | 0.7324 / 0.2122 | 0.7490 / 0.2385 | 0.6910 / 0.2111 | 0.7179 / 0.2416 |
| Application + Bureau + Alternative | **0.7414 / 0.2217** | **0.7579 / 0.2509** | **0.7100 / 0.2380** | **0.7376 / 0.2787** |

For HGB, Model C versus Model B gains **0.0089 ROC-AUC / 0.0124 PR-AUC overall** and **0.0197 / 0.0371 among thin-file borrowers**. These are results on one historical holdout that has already been inspected; they are not out-of-time validation or proof of production value.

## Reproduce

Use Python with the packages in `requirements.txt`. Keep the required local source CSVs in `data/raw/`; see [data setup](data/README.md). From the repository root, run:

```text
python -m src.features
python -m src.nonlinear
python -m src.phase10
python -m src.phase11
python -m src.phase12
python -m src.phase13
python -m src.release
python -m pytest -q
```

The split-ID CSVs under `data/processed/splits/` define the only supported split. Generated features, databases, and raw data are not committed. Full setup and interpretation are in [reproducibility](docs/reproducibility.md).

## Results and project map

- [Final evaluation](docs/final_results.md): model comparison, calibration, decision analysis, explanations, stability checks, and limitations.
- [Phase 8 controlled ablation](docs/phase8_ablation.md): experiment design and the primary linear comparison.
- [Curated tables and figures](reports/final/): release-ready model/cohort comparisons, calibration, decision points, importance, and prediction stability.
- `src/`: feature engineering and reproducible analysis modules.
- `sql/`: SQLite schema, analytical feature queries, and data-quality checks.
- `data/processed/splits/`: frozen train, validation, and test borrower IDs.

## Interpretation and intended use

Decision costs are illustrative normalized units, not bank economics or a lending policy. Feature importance and local reason codes describe model behaviour and are not causal or automatically suitable for adverse-action notices. The analysis does not establish fairness, temporal stability, representativeness, or production readiness. This is an educational portfolio study, not a lending system.
