"""Cohort, feature-distribution, missingness, and probability stability checks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import evaluate_model
from src.frozen_model import fit_hgb_with_validation_sigmoid
from src.models import load_and_split_data, thin_file_mask


MODEL_B = "Application + Bureau"
MODEL_C = "Application + Bureau + Alternative"
STABILITY_FEATURES = ["INST_LATE_PAYMENT_RATIO", "BUREAU_DEBT_RATIO", "PREV_APPROVED_COUNT",
                      "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3"]
MISSINGNESS_FEATURES = ["APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3"]


def _safe_metrics(y, p):
    values = evaluate_model(y, p)
    values["n"] = len(y)
    values["default_rate"] = float(np.mean(y)) if len(y) else np.nan
    return values


def _cohorts(frame):
    result = {"Overall": np.ones(len(frame), dtype=bool)}
    thin = thin_file_mask(frame).to_numpy()
    result["Thin-file"] = thin
    result["Non-thin-file"] = ~thin
    for feature in MISSINGNESS_FEATURES:
        missing = frame[feature].isna().to_numpy()
        result[f"{feature}: missing"] = missing
        result[f"{feature}: present"] = ~missing
    age = frame.APP_DAYS_BIRTH.abs() / 365.25
    age_groups = pd.cut(age, bins=[0, 25, 35, 45, 55, np.inf], right=False,
                        labels=["<25", "25-34", "35-44", "45-54", "55+"])
    for label in age_groups.cat.categories:
        result[f"Age {label}"] = age_groups.eq(label).to_numpy()
    employed = frame.APP_DAYS_EMPLOYED
    years = employed.abs() / 365.25
    employment_group = pd.Series(np.select(
        [employed.eq(365243), years < 1, years < 5],
        ["Sentinel-coded", "<1 year", "1-4 years"], default="5+ years"), index=frame.index)
    for label in ("Sentinel-coded", "<1 year", "1-4 years", "5+ years"):
        result[f"Employment {label}"] = employment_group.eq(label).to_numpy()
    return result


def _psi(train, current, bins=10):
    train = pd.Series(train, dtype=float)
    current = pd.Series(current, dtype=float)
    valid = train.dropna()
    if valid.empty:
        return np.nan
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([valid.iloc[0] - 1e-12, valid.iloc[0] + 1e-12])
    edges[0], edges[-1] = -np.inf, np.inf
    bins_train = pd.cut(train, edges, include_lowest=True).astype("object").fillna("Missing")
    bins_current = pd.cut(current, edges, include_lowest=True).astype("object").fillna("Missing")
    categories = list(pd.unique(pd.concat([bins_train, bins_current], ignore_index=True)))
    a = bins_train.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    b = bins_current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    epsilon = 1e-6
    a, b = np.maximum(a, epsilon), np.maximum(b, epsilon)
    return float(np.sum((b - a) * np.log(b / a)))


def _paired_bootstrap_deltas(y, p_b, p_c, n_bootstraps=500, seed=42):
    """Stratified paired bootstrap intervals for fixed predictions, no refitting."""
    y = np.asarray(y, dtype=int)
    p_b, p_c = np.asarray(p_b), np.asarray(p_c)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstraps):
        sample = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        differences.append((roc_auc_score(y[sample], p_c[sample]) - roc_auc_score(y[sample], p_b[sample]),
                            average_precision_score(y[sample], p_c[sample]) -
                            average_precision_score(y[sample], p_b[sample])))
    delta = np.asarray(differences)
    return {"roc_auc_delta": float(roc_auc_score(y, p_c) - roc_auc_score(y, p_b)),
            "roc_auc_ci_low": float(np.quantile(delta[:, 0], .025)),
            "roc_auc_ci_high": float(np.quantile(delta[:, 0], .975)),
            "pr_auc_delta": float(average_precision_score(y, p_c) - average_precision_score(y, p_b)),
            "pr_auc_ci_low": float(np.quantile(delta[:, 1], .025)),
            "pr_auc_ci_high": float(np.quantile(delta[:, 1], .975)),
            "bootstrap_resamples": n_bootstraps,
            "interval_method": "paired percentile bootstrap, stratified by outcome; predictions fixed"}


def run_phase13(filepath="data/processed/feature_master.csv", split_dir="data/processed/splits",
                output_dir="reports"):
    out = Path(output_dir)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath, split_dir)
    frames = {"train": X_train, "validation": X_val, "test": X_test}
    targets = {"train": y_train.to_numpy(), "validation": y_val.to_numpy(), "test": y_test.to_numpy()}
    models, predictions = {}, {split: {} for split in frames}
    for group in (MODEL_B, MODEL_C):
        models[group] = fit_hgb_with_validation_sigmoid(group, X_train, y_train, X_val, y_val, output_dir)
        for split, frame in frames.items():
            predictions[split][group] = models[group]["predict"](frame)

    performance_rows = []
    for split, frame in frames.items():
        for cohort, mask in _cohorts(frame).items():
            if not mask.any():
                continue
            for group in (MODEL_B, MODEL_C):
                performance_rows.append({"split": split, "cohort": cohort,
                                         "model": group, "n": int(mask.sum()),
                                         **_safe_metrics(targets[split][mask], predictions[split][group][mask])})
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(tables / "phase13_cohort_performance.csv", index=False)

    bootstrap_rows = []
    test_thin = thin_file_mask(X_test).to_numpy()
    for cohort, mask in {"Overall": np.ones(len(X_test), dtype=bool), "Thin-file": test_thin,
                         "Non-thin-file": ~test_thin}.items():
        bootstrap_rows.append({"cohort": cohort, "n": int(mask.sum()),
                               **_paired_bootstrap_deltas(
                                   y_test.to_numpy()[mask],
                                   predictions["test"][MODEL_B][mask],
                                   predictions["test"][MODEL_C][mask])})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(tables / "phase13_model_c_minus_b_bootstrap_ci.csv", index=False)

    feature_rows = []
    for feature in STABILITY_FEATURES:
        train_values = X_train[feature]
        for split in ("train", "validation", "test"):
            values = frames[split][feature]
            nonmissing = values.dropna()
            feature_rows.append({"feature": feature, "split": split, "n": len(values),
                                 "missing_rate": float(values.isna().mean()),
                                 "mean": float(nonmissing.mean()) if len(nonmissing) else np.nan,
                                 "median": float(nonmissing.median()) if len(nonmissing) else np.nan,
                                 "p10": float(nonmissing.quantile(.1)) if len(nonmissing) else np.nan,
                                 "p90": float(nonmissing.quantile(.9)) if len(nonmissing) else np.nan,
                                 "psi_vs_train": 0.0 if split == "train" else _psi(train_values, values)})
    feature_stability = pd.DataFrame(feature_rows)
    feature_stability.to_csv(tables / "phase13_feature_distribution_stability.csv", index=False)

    missing_rows = []
    for split, frame in frames.items():
        for feature in MISSINGNESS_FEATURES:
            miss = frame[feature].isna().to_numpy()
            missing_rows.append({"split": split, "feature": feature, "missing_n": int(miss.sum()),
                                 "missing_rate": float(miss.mean()), "present_n": int((~miss).sum())})
    missing_rates = pd.DataFrame(missing_rows)
    missing_rates.to_csv(tables / "phase13_missingness_rates.csv", index=False)

    prediction_rows = []
    for split, frame in frames.items():
        for cohort, mask in {"Overall": np.ones(len(frame), bool),
                             "Thin-file": thin_file_mask(frame).to_numpy(),
                             "Non-thin-file": ~thin_file_mask(frame).to_numpy()}.items():
            p = predictions[split][MODEL_C][mask]
            prediction_rows.append({"split": split, "cohort": cohort, "n": len(p),
                                    "mean": float(np.mean(p)), "median": float(np.median(p)),
                                    "p90": float(np.quantile(p, .9)), "p99": float(np.quantile(p, .99)),
                                    "maximum": float(np.max(p)),
                                    "share_below_0_01": float(np.mean(p < .01)),
                                    "share_above_0_5": float(np.mean(p > .5)),
                                    "share_below_0_001": float(np.mean(p < .001)),
                                    "share_above_0_99": float(np.mean(p > .99)),
                                    "note": "train predictions are in-sample; validation/test are held-out"})
    prediction_stability = pd.DataFrame(prediction_rows)
    prediction_stability.to_csv(tables / "phase13_prediction_stability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    for split, frame in frames.items():
        axes[0].hist(predictions[split][MODEL_C], bins=np.linspace(0, 1, 51), density=True,
                     histtype="step", linewidth=1.6, color=colors[split], label=split)
        mask = thin_file_mask(frame).to_numpy()
        axes[1].hist(predictions[split][MODEL_C][mask], bins=np.linspace(0, 1, 51), density=True,
                     histtype="step", linewidth=1.6, color=colors[split], label=f"{split} thin")
    axes[0].set(title="Model C calibrated PD — overall", xlabel="Predicted PD", ylabel="Density")
    axes[1].set(title="Model C calibrated PD — thin-file", xlabel="Predicted PD", ylabel="Density")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "phase13_prediction_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(STABILITY_FEATURES), figsize=(16, 4))
    for ax, feature in zip(axes, STABILITY_FEATURES):
        for split, frame in frames.items():
            vals = frame[feature].dropna()
            if len(vals):
                ax.hist(vals, bins=25, density=True, histtype="step", color=colors[split], label=split)
        ax.set_title(feature.replace("_", "\n"), fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Selected feature distributions by split (non-missing values)")
    fig.tight_layout()
    fig.savefig(figures / "phase13_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"performance": performance, "feature_stability": feature_stability,
            "missingness_rates": missing_rates, "prediction_stability": prediction_stability,
            "bootstrap_ci": bootstrap}


if __name__ == "__main__":
    result = run_phase13()
    print(result["performance"].query("split == 'test' and cohort in ['Overall', 'Thin-file', 'Non-thin-file']").to_string(index=False))
    print(result["feature_stability"].to_string(index=False))
    print(result["prediction_stability"].to_string(index=False))
