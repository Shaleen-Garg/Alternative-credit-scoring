"""Run the locked A/B/C logistic-regression feature-group comparison."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import (
    evaluate_model, extract_coefficients, plot_coefficients, plot_metric_bars,
    plot_model_comparison, plot_prob_distribution,
)
from src.models import FEATURE_GROUPS, load_and_split_data, build_pipeline, thin_file_mask


MODEL_KEYS = list(FEATURE_GROUPS)


def _bootstrap_delta(y, first, second, seed=42, repetitions=2000):
    """Percentile interval for paired second-minus-first metric difference."""
    y = np.asarray(y)
    first, second = np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = {"roc_auc": [], "pr_auc": []}
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        deltas["roc_auc"].append(roc_auc_score(y[idx], second[idx]) - roc_auc_score(y[idx], first[idx]))
        deltas["pr_auc"].append(average_precision_score(y[idx], second[idx]) - average_precision_score(y[idx], first[idx]))
    return {metric: (float(np.quantile(values, .025)), float(np.quantile(values, .975)))
            for metric, values in deltas.items()}


def run_ablation(filepath="data/processed/feature_master.csv", output_dir="reports",
                 split_dir="data/processed/splits"):
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath, split_dir)
    splits = {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
    predictions = {split: {} for split in splits}
    fitted = {}
    for model_name, features in FEATURE_GROUPS.items():
        model = build_pipeline(features)
        model.fit(X_train[features], y_train)
        fitted[model_name] = model
        for split, (X, _) in splits.items():
            predictions[split][model_name] = model.predict_proba(X[features])[:, 1]

    out = Path(output_dir)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for split, (_, y) in splits.items():
        for model_name in MODEL_KEYS:
            metric_rows.append({"split": split, "model": model_name,
                                **evaluate_model(y, predictions[split][model_name])})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables / "ablation_split_metrics.csv", index=False)

    thin = thin_file_mask(X_test).to_numpy()
    cohorts = {"Overall": np.ones(len(X_test), dtype=bool), "Thin-file": thin, "Non-thin-file": ~thin}
    subset_rows = []
    for split in ("validation", "test"):
        X_split, y_split = splits[split]
        thin_split = thin_file_mask(X_split).to_numpy()
        split_cohorts = {"Overall": np.ones(len(X_split), dtype=bool),
                         "Thin-file": thin_split, "Non-thin-file": ~thin_split}
        for cohort, mask in split_cohorts.items():
            for model_name in MODEL_KEYS:
                subset_rows.append({"split": split, "cohort": cohort, "model": model_name, "n": int(mask.sum()),
                                    **evaluate_model(y_split.to_numpy()[mask], predictions[split][model_name][mask])})
    subset_metrics = pd.DataFrame(subset_rows)
    subset_metrics.to_csv(tables / "ablation_cohort_metrics.csv", index=False)

    test = metrics[metrics.split == "test"].set_index("model")
    thin_metrics = subset_metrics[(subset_metrics.split == "test") & (subset_metrics.cohort == "Thin-file")].set_index("model")
    summary = pd.DataFrame([
        {"model": m, "feature_groups": ("Application" if i == 0 else "Application + Bureau" if i == 1 else "Application + Bureau + Alternative"),
         "test_roc_auc": test.loc[m, "roc_auc"], "test_pr_auc": test.loc[m, "pr_auc"],
         "thin_file_roc_auc": thin_metrics.loc[m, "roc_auc"], "thin_file_pr_auc": thin_metrics.loc[m, "pr_auc"]}
        for i, m in enumerate(MODEL_KEYS)
    ])
    summary.to_csv(tables / "ablation_summary.csv", index=False)

    increments = []
    for label, first, second in [("Bureau added to Application", MODEL_KEYS[0], MODEL_KEYS[1]),
                                 ("Alternative added to Application + Bureau", MODEL_KEYS[1], MODEL_KEYS[2])]:
        increments.append({"comparison": label,
                           "incremental_roc_auc": test.loc[second, "roc_auc"] - test.loc[first, "roc_auc"],
                           "incremental_pr_auc": test.loc[second, "pr_auc"] - test.loc[first, "pr_auc"],
                           "relative_roc_auc_pct": 100 * (test.loc[second, "roc_auc"] - test.loc[first, "roc_auc"]) / test.loc[first, "roc_auc"],
                           "relative_pr_auc_pct": 100 * (test.loc[second, "pr_auc"] - test.loc[first, "pr_auc"]) / test.loc[first, "pr_auc"]})
    increments = pd.DataFrame(increments)
    increments.to_csv(tables / "ablation_incremental_value.csv", index=False)

    predictions_test = predictions["test"]
    y_test_np = y_test.to_numpy()
    plot_model_comparison(y_test_np, predictions_test, figures / "ablation_roc_curves.png", "roc_auc")
    plot_model_comparison(y_test_np, predictions_test, figures / "ablation_pr_curves.png", "pr_auc")
    plot_metric_bars(summary.rename(columns={"test_roc_auc": "roc_auc"}), "roc_auc", figures / "ablation_test_roc_auc.png")
    plot_metric_bars(summary.rename(columns={"test_pr_auc": "pr_auc"}), "pr_auc", figures / "ablation_test_pr_auc.png")
    thin_frame = subset_metrics[(subset_metrics.split == "test") & (subset_metrics.cohort == "Thin-file")].rename(columns={"roc_auc": "thin_file_roc_auc", "pr_auc": "thin_file_pr_auc"})
    plot_metric_bars(thin_frame.rename(columns={"thin_file_roc_auc": "roc_auc"}), "roc_auc", figures / "thin_file_roc_auc.png", "Thin-file test ROC-AUC")
    plot_metric_bars(thin_frame.rename(columns={"thin_file_pr_auc": "pr_auc"}), "pr_auc", figures / "thin_file_pr_auc.png", "Thin-file test PR-AUC")
    plot_prob_distribution(y_test_np, predictions_test[MODEL_KEYS[-1]], figures / "test_probability_distribution.png")

    coeff = extract_coefficients(fitted[MODEL_KEYS[-1]], {
        "Application": FEATURE_GROUPS[MODEL_KEYS[0]],
        "Traditional Credit": FEATURE_GROUPS[MODEL_KEYS[1]][len(FEATURE_GROUPS[MODEL_KEYS[0]]):],
        "Alternative Behaviour": FEATURE_GROUPS[MODEL_KEYS[2]][len(FEATURE_GROUPS[MODEL_KEYS[1]]):],
    })
    coeff.to_csv(tables / "logistic_model_c_coefficients.csv", index=False)
    plot_coefficients(coeff, figures / "logistic_coefficients.png")

    all_features = pd.concat([X_train, X_val, X_test], ignore_index=True)
    coverage_rows = []
    for population, cohort_mask in [("Full thin-file population", thin_file_mask(all_features)),
                                    ("Thin-file test", thin_file_mask(X_test))]:
        frame = all_features.loc[cohort_mask] if population.startswith("Full") else X_test.loc[cohort_mask]
        size = len(frame)
        measures = [
            ("Any previous-application or installment history", (frame.PREV_APP_COUNT > 0) | (frame.INST_TOTAL_COUNT > 0)),
            ("Previous-application history", frame.PREV_APP_COUNT > 0),
            ("No previous applications", frame.PREV_APP_COUNT == 0),
            ("Previous history, zero approvals", (frame.PREV_APP_COUNT > 0) & (frame.PREV_APPROVED_COUNT == 0)),
            ("Previous history, positive approvals", frame.PREV_APPROVED_COUNT > 0),
            ("Previous history, zero refusals", (frame.PREV_APP_COUNT > 0) & (frame.PREV_REFUSED_COUNT == 0)),
            ("Previous history, positive refusals", frame.PREV_REFUSED_COUNT > 0),
            ("Installment history", frame.INST_TOTAL_COUNT > 0),
            ("No installment history", frame.INST_TOTAL_COUNT == 0),
            ("Installment history with missing payment information", frame.INST_PAYMENT_MISSING_COUNT > 0),
            ("Installment history, zero recorded late payments", (frame.INST_TOTAL_COUNT > 0) & (frame.INST_LATE_PAYMENT_COUNT == 0)),
            ("Installment history, positive recorded late payments", frame.INST_LATE_PAYMENT_COUNT > 0),
            ("Installment history, zero underpayments", (frame.INST_TOTAL_COUNT > 0) & (frame.INST_UNDERPAYMENT_COUNT == 0)),
            ("Installment history, positive underpayments", frame.INST_UNDERPAYMENT_COUNT > 0),
        ]
        coverage_rows.append({"population": population, "measure": "Population size", "n": size, "share": 1.0})
        for measure, condition in measures:
            count = int(condition.sum())
            coverage_rows.append({"population": population, "measure": measure, "n": count,
                                 "share": count / size if size else np.nan})
    coverage_table = pd.DataFrame(coverage_rows)
    coverage_table.to_csv(tables / "alternative_history_coverage.csv", index=False)

    alt = X_test.loc[thin, ["INST_TOTAL_COUNT", "INST_LATE_PAYMENT_COUNT", "INST_LATE_PAYMENT_RATIO", "INST_UNDERPAYMENT_COUNT"]]
    alt.corr(method="pearson").to_csv(tables / "installment_correlations_thin_file.csv")
    redundant_features = ["INST_TOTAL_COUNT", "INST_LATE_PAYMENT_COUNT",
                          "INST_LATE_PAYMENT_RATIO", "INST_UNDERPAYMENT_COUNT"]
    sensitivity_rows = []
    full_test = test.loc[MODEL_KEYS[-1]]
    full_thin = thin_metrics.loc[MODEL_KEYS[-1]]
    for omitted in redundant_features:
        reduced = [feature for feature in FEATURE_GROUPS[MODEL_KEYS[-1]] if feature != omitted]
        diagnostic = build_pipeline(reduced).fit(X_train[reduced], y_train)
        for split in ("validation", "test"):
            X, y = splits[split]
            scores = diagnostic.predict_proba(X[reduced])[:, 1]
            metric_values = evaluate_model(y, scores)
            row = {"omitted_feature": omitted, "split": split, **metric_values}
            if split == "test":
                for key in ("roc_auc", "pr_auc", "brier", "log_loss"):
                    row[f"delta_{key}_vs_full_model_c"] = metric_values[key] - full_test[key]
                thin_scores = scores[X.BUREAU_CREDIT_COUNT.eq(0).to_numpy()]
                thin_y = y.to_numpy()[X.BUREAU_CREDIT_COUNT.eq(0).to_numpy()]
                thin_values = evaluate_model(thin_y, thin_scores)
                row["thin_file_roc_auc"] = thin_values["roc_auc"]
                row["delta_thin_file_roc_auc_vs_full"] = thin_values["roc_auc"] - full_thin.roc_auc
                row["thin_file_pr_auc"] = thin_values["pr_auc"]
                row["delta_thin_file_pr_auc_vs_full"] = thin_values["pr_auc"] - full_thin.pr_auc
            sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(tables / "ablation_redundancy_sensitivity.csv", index=False)
    bootstrap_repetitions = 2000
    ci_overall = _bootstrap_delta(y_test_np, predictions_test[MODEL_KEYS[1]], predictions_test[MODEL_KEYS[2]], repetitions=bootstrap_repetitions)
    ci_thin = _bootstrap_delta(y_test_np[thin], predictions_test[MODEL_KEYS[1]][thin], predictions_test[MODEL_KEYS[2]][thin], repetitions=bootstrap_repetitions)
    pd.DataFrame([
        {"cohort": cohort, "metric": metric, "resamples": bootstrap_repetitions,
         "ci_2_5_pct": interval[0], "ci_97_5_pct": interval[1]}
        for cohort, intervals in [("Overall test", ci_overall), ("Thin-file test", ci_thin)]
        for metric, interval in intervals.items()
    ]).to_csv(tables / "ablation_bootstrap_intervals.csv", index=False)

    return {"splits": (X_train, X_val, X_test), "metrics": metrics, "subsets": subset_metrics,
            "summary": summary, "increments": increments, "coefficients": coeff,
            "coverage": coverage_table, "correlations": alt.corr(),
            "ci_overall": ci_overall, "ci_thin": ci_thin}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/feature_master.csv")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--splits", default="data/processed/splits")
    args = parser.parse_args()
    result = run_ablation(args.data, args.output, args.splits)
    print("Split sizes:", [len(split) for split in result["splits"]])
    print(result["summary"].to_string(index=False))
    print(result["increments"].to_string(index=False))
