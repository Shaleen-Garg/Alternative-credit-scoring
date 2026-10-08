"""Validation-selected decision thresholds under transparent illustrative costs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.frozen_model import fit_hgb_with_validation_sigmoid
from src.models import load_and_split_data, thin_file_mask


COST_RATIOS = (1, 2, 5, 10)  # C_FN:C_FP, in normalized cost units
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
MODEL_B = "Application + Bureau"
MODEL_C = "Application + Bureau + Alternative"


def _decision_arrays(y, pd_hat):
    y = np.asarray(y).reshape(-1)
    pd_hat = np.asarray(pd_hat, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(pd_hat):
        raise ValueError("Targets and probabilities must have equal non-zero length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1")
    if not np.isfinite(pd_hat).all() or ((pd_hat < 0) | (pd_hat > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    return y.astype(int), pd_hat


def _validate_costs(fn_cost, fp_cost):
    costs = np.asarray([fn_cost, fp_cost], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Decision costs must be finite and non-negative")


def decision_metrics(y, pd_hat, threshold, fn_cost=1, fp_cost=1):
    y, pd_hat = _decision_arrays(y, pd_hat)
    _validate_costs(fn_cost, fp_cost)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predicted_default = pd_hat >= threshold
    tp = int(np.sum(predicted_default & (y == 1)))
    fp = int(np.sum(predicted_default & (y == 0)))
    tn = int(np.sum(~predicted_default & (y == 0)))
    fn = int(np.sum(~predicted_default & (y == 1)))
    approved = tn + fn
    rejected = tp + fp
    return {
        "threshold": float(threshold), "n": len(y), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "approval_rate": approved / len(y) if len(y) else np.nan,
        "rejection_rate": rejected / len(y) if len(y) else np.nan,
        "default_rate_among_approved": fn / approved if approved else np.nan,
        "expected_cost": fn * fn_cost + fp * fp_cost,
        "normalized_expected_cost": (fn * fn_cost + fp * fp_cost) / len(y) if len(y) else np.nan,
    }


def _select_cost_threshold(y, pd_hat, fn_cost, fp_cost=1):
    y, pd_hat = _decision_arrays(y, pd_hat)
    _validate_costs(fn_cost, fp_cost)
    order = np.argsort(pd_hat, kind="stable")
    sorted_probabilities = pd_hat[order]
    sorted_targets = y[order]
    thresholds = np.append(np.unique(sorted_probabilities),
                           np.nextafter(sorted_probabilities[-1], np.inf))
    approved_count = np.searchsorted(sorted_probabilities, thresholds, side="left")
    approved_defaults = np.r_[0, np.cumsum(sorted_targets == 1)][approved_count]
    approved_nondefaults = np.r_[0, np.cumsum(sorted_targets == 0)][approved_count]
    rejected_nondefaults = int(np.sum(y == 0)) - approved_nondefaults
    costs = (approved_defaults * fn_cost + rejected_nondefaults * fp_cost) / len(y)
    best = np.lexsort((-thresholds, costs))[0]
    return decision_metrics(y, pd_hat, thresholds[best], fn_cost, fp_cost)


def _cohort_masks(frame):
    thin = thin_file_mask(frame).to_numpy()
    return {"Overall": np.ones(len(frame), dtype=bool), "Thin-file": thin,
            "Non-thin-file": ~thin}


def run_decision_analysis(filepath="data/processed/feature_master.csv", split_dir="data/processed/splits",
                          output_dir="reports"):
    out = Path(output_dir)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(filepath, split_dir)
    models = {}
    for group in (MODEL_B, MODEL_C):
        models[group] = fit_hgb_with_validation_sigmoid(group, X_train, y_train, X_val, y_val, output_dir)
    predictions = {
        "validation": {g: models[g]["predict"](X_val) for g in models},
        "test": {g: models[g]["predict"](X_test) for g in models},
    }
    targets = {"validation": y_val.to_numpy(), "test": y_test.to_numpy()}
    frames = {"validation": X_val, "test": X_test}
    masks = {split: _cohort_masks(frame) for split, frame in frames.items()}

    curve_rows = []
    for split in ("validation", "test"):
        for group in (MODEL_B, MODEL_C):
            for cohort, mask in masks[split].items():
                for threshold in THRESHOLDS:
                    for ratio in COST_RATIOS:
                        metrics = decision_metrics(targets[split][mask], predictions[split][group][mask],
                                                   threshold, fn_cost=ratio, fp_cost=1)
                        curve_rows.append({"split": split, "model": group, "cohort": cohort,
                                           "FN_to_FP_cost_ratio": f"{ratio}:1", **metrics})
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(tables / "decision_threshold_curves.csv", index=False)

    selected_rows = []
    for ratio in COST_RATIOS:
        for cohort, mask in masks["validation"].items():
            best = _select_cost_threshold(
                y_val.to_numpy()[mask], predictions["validation"][MODEL_C][mask],
                fn_cost=ratio, fp_cost=1,
            )
            selected_rows.append({"model_used_for_threshold_selection": MODEL_C, "cohort": cohort,
                                  "FN_to_FP_cost_ratio": f"{ratio}:1", **best,
                                  "selection_data": "validation only"})
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(tables / "selected_validation_thresholds.csv", index=False)

    test_rows = []
    for row in selected.itertuples(index=False):
        mask = masks["test"][row.cohort]
        for group in (MODEL_B, MODEL_C):
            test_rows.append({"threshold_selected_on": MODEL_C, "model": group,
                              "cohort": row.cohort, "FN_to_FP_cost_ratio": row.FN_to_FP_cost_ratio,
                              **decision_metrics(y_test.to_numpy()[mask], predictions["test"][group][mask],
                                                 row.threshold, fn_cost=int(row.FN_to_FP_cost_ratio.split(":")[0]), fp_cost=1)})
    test_selected = pd.DataFrame(test_rows)
    test_selected.to_csv(tables / "selected_test_thresholds.csv", index=False)

    # Select separate Model B/C validation cutoffs to target the same validation
    # approval volume, then compare the realized frozen-test trade-offs.
    volume_rows = []
    for cohort in ("Overall", "Thin-file", "Non-thin-file"):
        val_mask = masks["validation"][cohort]
        test_mask = masks["test"][cohort]
        for target_approval in (0.50, 0.70, 0.90):
            for group in (MODEL_B, MODEL_C):
                p_val = predictions["validation"][group][val_mask]
                threshold = float(np.quantile(p_val, target_approval, method="higher"))
                val_metrics = decision_metrics(y_val.to_numpy()[val_mask], p_val, threshold)
                test_metrics = decision_metrics(y_test.to_numpy()[test_mask],
                                                predictions["test"][group][test_mask], threshold)
                volume_rows.append({"cohort": cohort, "target_validation_approval_rate": target_approval,
                                    "model": group, "threshold_selected_on": "own validation predictions",
                                    "threshold": threshold,
                                    "validation_approval_rate": val_metrics["approval_rate"],
                                    **{f"test_{key}": value for key, value in test_metrics.items() if key != "threshold"}})
    volume = pd.DataFrame(volume_rows)
    volume.to_csv(tables / "approval_rate_comparison.csv", index=False)

    # Plot 1: validation cost curves, plus validation and test volume-risk curves.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    overall_val = curves[(curves.split == "validation") & (curves.model == MODEL_C) & (curves.cohort == "Overall")]
    for ratio in COST_RATIOS:
        subset = overall_val[overall_val.FN_to_FP_cost_ratio == f"{ratio}:1"]
        axes[0, 0].plot(subset.threshold, subset.normalized_expected_cost, label=f"FN:FP {ratio}:1")
    axes[0, 0].set(title="Validation normalized cost (Model C)", xlabel="Default-probability threshold",
                   ylabel="Cost units per applicant")
    axes[0, 0].legend()
    for metric, ax, title in [("approval_rate", axes[0, 1], "Approval rate"),
                              ("recall", axes[1, 0], "Default recall"),
                              ("precision", axes[1, 1], "Default precision")]:
        subset = overall_val[overall_val.FN_to_FP_cost_ratio == "2:1"]
        ax.plot(subset.threshold, subset[metric])
        ax.set(title=f"Validation {title} (FN:FP 2:1)", xlabel="Default-probability threshold", ylabel=title)
    fig.tight_layout()
    fig.savefig(figures / "threshold_tradeoffs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cohort, marker in (("Overall", "o"), ("Thin-file", "s"), ("Non-thin-file", "^") ):
        for group, style in ((MODEL_B, "--"), (MODEL_C, "-")):
            subset = volume[(volume.cohort == cohort) & (volume.model == group)]
            ax.plot(subset.test_approval_rate, subset.test_default_rate_among_approved,
                    linestyle=style, marker=marker,
                    label=f"{cohort} — {'Model B' if group == MODEL_B else 'Model C'}")
    ax.set(title="Frozen-test risk–approval-volume trade-off", xlabel="Approval rate",
           ylabel="Default rate among approved", xlim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "risk_vs_approval.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"curves": curves, "selected_validation": selected,
            "selected_test": test_selected, "same_volume": volume}


if __name__ == "__main__":
    result = run_decision_analysis()
    print(result["selected_validation"].to_string(index=False))
    print(result["selected_test"].to_string(index=False))
