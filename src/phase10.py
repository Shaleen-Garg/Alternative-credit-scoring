"""Select a frozen Phase 9 model on validation, calibrate, and evaluate once on test."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.calibration import (
    CALIBRATION_METHODS, apply_calibrator, calibration_metrics, fit_calibrator,
)
from src.evaluation import evaluate_model
from src.models import FEATURE_GROUPS, load_and_split_data, build_pipeline, thin_file_mask
from src.nonlinear import _new_model


def select_model_for_calibration(reports_dir="reports"):
    """Select from train/validation only using overall/thin ranking, gap, and complexity."""
    tables = Path(reports_dir) / "tables"
    linear = pd.read_csv(tables / "phase8_split_metrics.csv")
    nonlinear = pd.read_csv(tables / "phase9_nonlinear_metrics.csv")
    cohort_linear = pd.read_csv(tables / "phase8_cohort_metrics.csv")
    cohort_nonlinear = pd.read_csv(tables / "phase9_nonlinear_cohort_metrics.csv")
    rows = []
    for model_type, metrics, cohorts in [
        ("Logistic Regression", linear, cohort_linear),
        ("HistGradientBoostingClassifier", nonlinear, cohort_nonlinear),
    ]:
        for model_name in FEATURE_GROUPS:
            train = metrics[(metrics.split == "train") & (metrics.model == model_name)].iloc[0]
            val = metrics[(metrics.split == "validation") & (metrics.model == model_name)].iloc[0]
            thin = cohorts[(cohorts.split == "validation") & (cohorts.cohort == "Thin-file") &
                           (cohorts.model == model_name)].iloc[0]
            rows.append({"model_type": model_type, "model": model_name,
                         "train_roc_auc": train.roc_auc,
                         "validation_roc_auc": val.roc_auc,
                         "validation_pr_auc": val.pr_auc,
                         "thin_validation_roc_auc": thin.roc_auc,
                         "thin_validation_pr_auc": thin.pr_auc,
                         "train_validation_roc_gap": train.roc_auc - val.roc_auc,
                         "complexity_order": 0 if model_type == "Logistic Regression" else 1})
    frame = pd.DataFrame(rows)
    measures = ["validation_roc_auc", "validation_pr_auc",
                "thin_validation_roc_auc", "thin_validation_pr_auc"]
    ranks = []
    for metric in measures:
        ranks.append(frame[metric].rank(pct=True))
    gap_score = 1 - frame.train_validation_roc_gap.abs().rank(pct=True)
    frame["validation_selection_score"] = (sum(ranks) + gap_score) / (len(ranks) + 1)
    frame = frame.sort_values(["validation_selection_score", "complexity_order"],
                              ascending=[False, True]).reset_index(drop=True)
    return frame


def _draw_reliability(bins, cohort, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    for method, group in bins[bins.cohort == cohort].groupby("method"):
        ax.plot(group.mean_predicted, group.observed_rate, marker="o", label=method)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted default probability",
           ylabel="Observed default frequency", title=f"Test reliability — {cohort}")
    ax.legend()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_phase10(filepath="data/processed/feature_master.csv",
                split_dir="data/processed/splits", output_dir="reports"):
    out = Path(output_dir)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    selection = select_model_for_calibration(output_dir)
    selection.to_csv(tables / "phase10_model_selection.csv", index=False)
    chosen = selection.iloc[0]
    model_type, model_name = chosen.model_type, chosen.model
    features = FEATURE_GROUPS[model_name]
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath, split_dir)
    if model_type == "Logistic Regression":
        base_model = build_pipeline(features)
    else:
        params = json.loads((tables / "phase9_selected_parameters.json").read_text(encoding="utf-8"))["parameters"]
        base_model = _new_model(params)
    base_model.fit(X_train[features], y_train)
    p_val = base_model.predict_proba(X_val[features])[:, 1]
    p_test = base_model.predict_proba(X_test[features])[:, 1]

    # Use one half of validation to fit calibrators and the other half to select a method.
    fit_idx, select_idx = train_test_split(
        np.arange(len(y_val)), test_size=0.5, stratify=y_val, random_state=42,
    )
    selection_rows = []
    fitted = {"uncalibrated": None}
    val_select_predictions = {"uncalibrated": p_val[select_idx]}
    for method in ("sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, p_val[fit_idx], y_val.iloc[fit_idx])
        fitted[method] = calibrator
        val_select_predictions[method] = apply_calibrator(method, calibrator, p_val[select_idx])
    for method in CALIBRATION_METHODS:
        scores = evaluate_model(y_val.iloc[select_idx], val_select_predictions[method])
        selection_rows.append({"method": method, "n_calibrator_fit": len(fit_idx),
                               "n_validation_selection": len(select_idx), **scores})
    method_selection = pd.DataFrame(selection_rows).sort_values(
        ["log_loss", "brier"], ascending=True
    ).reset_index(drop=True)
    selected_method = method_selection.iloc[0].method
    method_selection["selected"] = method_selection.method.eq(selected_method)
    method_selection.to_csv(tables / "phase10_calibration_validation_selection.csv", index=False)

    # Refit only the chosen mapping on all validation predictions, never on test labels.
    if selected_method != "uncalibrated":
        final_calibrator = fit_calibrator(selected_method, p_val, y_val)
    else:
        final_calibrator = None
    test_predictions = {"uncalibrated": p_test}
    for method in ("sigmoid", "isotonic"):
        mapping = final_calibrator if method == selected_method else fit_calibrator(method, p_val, y_val)
        test_predictions[method] = apply_calibrator(method, mapping, p_test)

    thin = thin_file_mask(X_test).to_numpy()
    cohorts = {"Overall": np.ones(len(X_test), dtype=bool),
               "Thin-file": thin, "Non-thin-file": ~thin}
    test_rows, bin_frames = [], []
    for cohort, mask in cohorts.items():
        for method in CALIBRATION_METHODS:
            values, bins = calibration_metrics(y_test.to_numpy()[mask], test_predictions[method][mask], n_bins=15)
            test_rows.append({"selected_model_type": model_type, "selected_model": model_name,
                              "cohort": cohort, "method": method,
                              "method_selected_on_validation": method == selected_method, **values})
            bins["cohort"], bins["method"] = cohort, method
            bin_frames.append(bins)
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(tables / "phase10_calibration_test_metrics.csv", index=False)
    bins = pd.concat(bin_frames, ignore_index=True)
    bins.to_csv(tables / "phase10_calibration_reliability_bins.csv", index=False)
    _draw_reliability(bins, "Overall", figures / "phase10_reliability_overall.png")
    _draw_reliability(bins, "Thin-file", figures / "phase10_reliability_thin_file.png")

    record = {
        "selected_model_type": model_type,
        "selected_feature_group": model_name,
        "feature_count": len(features),
        "selection_basis": "validation-only rank aggregation of overall and thin-file ROC-AUC/PR-AUC, train-validation ROC gap, then model complexity",
        "calibration_method": selected_method,
        "calibrator_fit_data": "validation predictions and labels only",
        "calibration_method_selection": "held-out half of validation (stratified), minimizing log loss then Brier",
        "test_used_for_model_or_calibrator_fit": False,
    }
    (tables / "phase10_selected_model.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"selection": selection, "calibration_selection": method_selection,
            "test_metrics": test_metrics, "reliability_bins": bins,
            "selected_model": record}


if __name__ == "__main__":
    result = run_phase10()
    print(json.dumps(result["selected_model"], indent=2))
    print(result["calibration_selection"].to_string(index=False))
    print(result["test_metrics"].to_string(index=False))
