"""Controlled nonlinear robustness study using HistGradientBoostingClassifier."""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.evaluation import evaluate_model, plot_model_comparison
from src.models import FEATURE_GROUPS, load_and_split_data, thin_file_mask, build_pipeline


HGB_CANDIDATES = [
    {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15,
     "min_samples_leaf": 100, "l2_regularization": 5.0},
    {"learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 15,
     "min_samples_leaf": 100, "l2_regularization": 2.0},
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31,
     "min_samples_leaf": 100, "l2_regularization": 5.0},
]


def _tree_features(frame, columns):
    result = frame[columns].copy()
    if "APP_DAYS_EMPLOYED" in result:
        result.loc[result.APP_DAYS_EMPLOYED == 365243, "APP_DAYS_EMPLOYED"] = np.nan
    values = result.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite values found in tree model inputs")
    return result.astype(float)


def _new_model(params):
    return HistGradientBoostingClassifier(
        **params, early_stopping=False, random_state=42,
    )


def run_phase9(filepath="data/processed/feature_master.csv",
               split_dir="data/processed/splits", output_dir="reports"):
    run_started = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath, split_dir)
    split_x = {"train": X_train, "validation": X_val, "test": X_test}
    split_y = {"train": y_train, "validation": y_val, "test": y_test}

    # Select one shared configuration on Model C validation PR-AUC only.
    model_c_features = FEATURE_GROUPS["Application + Bureau + Alternative"]
    train_c = _tree_features(X_train, model_c_features)
    val_c = _tree_features(X_val, model_c_features)
    candidate_rows = []
    candidate_times = []
    for candidate_id, params in enumerate(HGB_CANDIDATES, start=1):
        started = time.perf_counter()
        candidate = _new_model(params).fit(train_c, y_train)
        seconds = time.perf_counter() - started
        pred = candidate.predict_proba(val_c)[:, 1]
        scores = evaluate_model(y_val, pred)
        candidate_rows.append({"candidate": candidate_id, **params, "feature_count": len(model_c_features),
                               "fit_seconds": seconds, **scores})
        candidate_times.append(seconds)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["pr_auc", "roc_auc", "fit_seconds"], ascending=[False, False, True]
    ).reset_index(drop=True)
    selected_row = candidates.iloc[0]
    selected_params = HGB_CANDIDATES[int(selected_row["candidate"]) - 1].copy()

    out = Path(output_dir)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(tables / "phase9_validation_candidates.csv", index=False)
    (tables / "phase9_selected_parameters.json").write_text(
        json.dumps({"model": "HistGradientBoostingClassifier", "parameters": selected_params,
                    "selected_on": "Model C validation PR-AUC, ROC-AUC tie-break",
                    "random_state": 42, "early_stopping": False}, indent=2), encoding="utf-8"
    )

    fitted = {}
    predictions = {split: {} for split in split_x}
    runtime_rows = []
    for model_name, features in FEATURE_GROUPS.items():
        started = time.perf_counter()
        model = _new_model(selected_params).fit(_tree_features(X_train, features), y_train)
        fit_seconds = time.perf_counter() - started
        fitted[model_name] = model
        runtime_rows.append({"model": model_name, "model_type": "HistGradientBoostingClassifier",
                             "fit_seconds": fit_seconds, "feature_count": len(features),
                             **selected_params})
        for split, frame in split_x.items():
            predictions[split][model_name] = model.predict_proba(_tree_features(frame, features))[:, 1]

    linear_predictions = {split: {} for split in split_x}
    for model_name, features in FEATURE_GROUPS.items():
        linear_model = build_pipeline(features).fit(X_train[features], y_train)
        for split, frame in split_x.items():
            linear_predictions[split][model_name] = linear_model.predict_proba(frame[features])[:, 1]

    metric_rows = []
    for split in ("train", "validation", "test"):
        for model_name in FEATURE_GROUPS:
            metric_rows.append({"split": split, "model": model_name,
                                **evaluate_model(split_y[split], predictions[split][model_name])})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables / "phase9_nonlinear_metrics.csv", index=False)

    cohort_rows = []
    for split in ("validation", "test"):
        frame, target = split_x[split], split_y[split]
        thin = thin_file_mask(frame).to_numpy()
        for cohort, mask in {"Overall": np.ones(len(frame), dtype=bool),
                             "Thin-file": thin, "Non-thin-file": ~thin}.items():
            for model_name in FEATURE_GROUPS:
                cohort_rows.append({"split": split, "cohort": cohort, "model": model_name,
                                    "n": int(mask.sum()), **evaluate_model(
                                        target.to_numpy()[mask], predictions[split][model_name][mask])})
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(tables / "phase9_nonlinear_cohort_metrics.csv", index=False)

    # Freeze the linear benchmark into the joint comparison; do not refit it here.
    linear = pd.read_csv(tables / "phase8_split_metrics.csv") if (tables / "phase8_split_metrics.csv").exists() else None
    if linear is not None:
        linear_test = linear[linear.split.isin(["validation", "test"])].copy()
        linear_test["model_type"] = "Logistic Regression"
        nonlinear_vt = metrics[metrics.split.isin(["validation", "test"])].copy()
        nonlinear_vt["model_type"] = "HistGradientBoostingClassifier"
        combined = pd.concat([linear_test, nonlinear_vt], ignore_index=True)
        combined.to_csv(tables / "phase9_linear_nonlinear_metrics.csv", index=False)
        validation_test = combined.pivot(index=["model", "model_type"], columns="split", values=["roc_auc", "pr_auc"])
        comparison = pd.DataFrame({
            "model": [index[0] for index in validation_test.index],
            "model_type": [index[1] for index in validation_test.index],
            "validation_roc_auc": validation_test[("roc_auc", "validation")].to_numpy(),
            "test_roc_auc": validation_test[("roc_auc", "test")].to_numpy(),
            "validation_pr_auc": validation_test[("pr_auc", "validation")].to_numpy(),
            "test_pr_auc": validation_test[("pr_auc", "test")].to_numpy(),
        })
        comparison.to_csv(tables / "phase9_model_comparison.csv", index=False)
        delta_rows = []
        for group, first, second in [("Bureau addition", "Application", "Application + Bureau"),
                                     ("Alternative addition", "Application + Bureau", "Application + Bureau + Alternative")]:
            row = {"feature_group_addition": group}
            for prefix, model_type in (("logistic", "Logistic Regression"),
                                       ("nonlinear", "HistGradientBoostingClassifier")):
                for metric in ("roc_auc", "pr_auc"):
                    val_a = combined[(combined.model == first) & (combined.model_type == model_type) & (combined.split == "test")][metric].iloc[0]
                    val_b = combined[(combined.model == second) & (combined.model_type == model_type) & (combined.split == "test")][metric].iloc[0]
                    row[f"{prefix}_delta_{metric}"] = val_b - val_a
            delta_rows.append(row)
        pd.DataFrame(delta_rows).to_csv(tables / "phase9_incremental_comparison.csv", index=False)

    runtime_rows.append({"model": "Full Phase 9 run", "model_type": "including validation search, fits, metrics, importance, and plots",
                         "fit_seconds": time.perf_counter() - run_started, "feature_count": len(model_c_features),
                         **selected_params})
    pd.DataFrame(runtime_rows).to_csv(tables / "phase9_runtime.csv", index=False)
    comparison_predictions = {}
    for model_name in FEATURE_GROUPS:
        comparison_predictions[f"Logistic {model_name}"] = linear_predictions["test"][model_name]
        comparison_predictions[f"Nonlinear {model_name}"] = predictions["test"][model_name]
    plot_model_comparison(y_test.to_numpy(), comparison_predictions,
                          figures / "phase9_logistic_vs_nonlinear_roc.png", "roc_auc")
    plot_model_comparison(y_test.to_numpy(), comparison_predictions,
                          figures / "phase9_logistic_vs_nonlinear_pr.png", "pr_auc")

    model_c = fitted["Application + Bureau + Alternative"]
    perm = permutation_importance(
        model_c, val_c, y_val, scoring="average_precision", n_repeats=3,
        random_state=42, n_jobs=1,
    )
    importance = pd.DataFrame({"feature": model_c_features,
                               "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std})
    importance["feature_group"] = importance.feature.map({
        feature: "Application" for feature in FEATURE_GROUPS["Application"]
    } | {
        feature: "Traditional Credit" for feature in FEATURE_GROUPS["Application + Bureau"][len(FEATURE_GROUPS["Application"]):]
    } | {
        feature: "Alternative / non-bureau behaviour" for feature in FEATURE_GROUPS["Application + Bureau + Alternative"][len(FEATURE_GROUPS["Application + Bureau"]):]
    })
    importance.sort_values("importance_mean", ascending=False).to_csv(
        tables / "phase9_validation_permutation_importance.csv", index=False
    )
    top = importance.sort_values("importance_mean").tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.feature, top.importance_mean, xerr=top.importance_std, color="#4C78A8")
    ax.set(xlabel="Decrease in validation average precision after permutation",
           title="Model C validation permutation importance")
    fig.savefig(figures / "phase9_validation_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Two-way partial dependence uses validation features only and is diagnostic, not causal.
    pdp_sample = val_c.sample(n=min(750, len(val_c)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [("INST_LATE_PAYMENT_RATIO", "APP_EXT_SOURCE_2"),
             ("PREV_APPROVED_COUNT", "APP_EXT_SOURCE_2")]
    for pair, ax in zip(pairs, axes):
        PartialDependenceDisplay.from_estimator(
            model_c, pdp_sample, [pair], kind="average", grid_resolution=12,
            ax=ax, subsample=750, random_state=42,
        )
        ax.set_title(f"Validation partial dependence: {pair[0]} × {pair[1]}")
    fig.tight_layout()
    fig.savefig(figures / "phase9_validation_interaction_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"candidate_metrics": candidates, "selected_params": selected_params,
            "metrics": metrics, "cohorts": cohorts, "comparison": comparison if linear is not None else None,
            "importance": importance, "runtime": pd.DataFrame(runtime_rows), "predictions": predictions,
            "models": fitted}


if __name__ == "__main__":
    result = run_phase9()
    print("Selected parameters:", result["selected_params"])
    print(result["metrics"].to_string(index=False))
