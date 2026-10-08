"""Evaluation helpers for risk-ranking experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
    roc_auc_score, roc_curve,
)


def evaluate_model(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("y_true and y_prob must have the same non-zero length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(y_prob).all() or (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if np.unique(y_true).size != 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan,
                "brier": brier_score_loss(y_true, y_prob),
                "log_loss": log_loss(y_true, y_prob, labels=[0, 1])}
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
    }


def extract_coefficients(pipeline, feature_groups=None):
    """Return standardized model coefficients with feature names and group labels."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    names = [name.split("__", 1)[-1] for name in names]
    coef = pipeline.named_steps["clf"].coef_[0]
    group_map = {}
    for group, features in (feature_groups or {}).items():
        for feature in features:
            group_map[feature] = group
    frame = pd.DataFrame({"feature": names, "coefficient": coef})
    frame["odds_ratio_per_sd"] = np.exp(np.clip(frame["coefficient"], -700, 700))
    frame["feature_group"] = frame.feature.map(group_map).fillna("Missingness indicator")
    frame["abs_coefficient"] = frame.coefficient.abs()
    return frame.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_prob_distribution(y_true, y_prob, save_path=None):
    frame = pd.DataFrame({"target": np.asarray(y_true), "probability": np.asarray(y_prob)})
    fig, ax = plt.subplots()
    for target, label in [(0, "Non-default"), (1, "Default")]:
        values = frame.loc[frame.target == target, "probability"]
        ax.hist(values, bins=40, density=True, alpha=.45, label=label)
    ax.set(xlabel="Predicted probability", ylabel="Density", title="Test probability distribution")
    ax.legend()
    _save(fig, save_path)


def plot_model_comparison(y_true, predictions, save_path, metric="roc_auc"):
    from sklearn.metrics import auc
    fig, ax = plt.subplots()
    for name, prob in predictions.items():
        if metric == "roc_auc":
            x, y, _ = roc_curve(y_true, prob)
            score = auc(x, y)
            ax.plot(x, y, label=f"{name} ({score:.3f})")
        else:
            y, x, _ = precision_recall_curve(y_true, prob)
            score = average_precision_score(y_true, prob)
            ax.plot(x, y, label=f"{name} ({score:.3f})")
    if metric == "roc_auc":
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    else:
        ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curves")
    ax.legend()
    _save(fig, save_path)


def plot_metric_bars(frame, metric, save_path, title=None):
    fig, ax = plt.subplots()
    ax.bar(frame["model"], frame[metric], color=["#4C78A8", "#F58518", "#54A24B"][:len(frame)])
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Test {metric.upper()} comparison")
    ax.tick_params(axis="x", rotation=18)
    _save(fig, save_path)


def plot_coefficients(coefficients, save_path, feature_group="Alternative Behaviour"):
    selected = coefficients[coefficients.feature_group == feature_group].sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, max(3, .35 * len(selected))))
    ax.barh(selected.feature, selected.coefficient, color="#54A24B")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Standardized coefficient", title=f"{feature_group} coefficients")
    _save(fig, save_path)
