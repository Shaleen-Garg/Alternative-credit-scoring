"""Validation-fitted probability calibration and evaluation diagnostics."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def _safe_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration probabilities must be finite and non-empty")
    return np.clip(values, 1e-7, 1 - 1e-7)


def fit_calibrator(method, probabilities, target):
    """Fit a calibrator from one calibration partition only."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be sigmoid or isotonic")
    p = _safe_probabilities(probabilities)
    y = np.asarray(target, dtype=int).reshape(-1)
    if len(p) != len(y) or np.unique(y).size != 2:
        raise ValueError("Calibration fit requires matching probabilities and both target classes")
    if method == "sigmoid":
        # Platt scaling: fit a logistic map from raw prediction log-odds to outcome.
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        calibrator.fit(logit(p).reshape(-1, 1), y)
        return calibrator
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(p, y)
    return calibrator


def apply_calibrator(method, calibrator, probabilities):
    p = _safe_probabilities(probabilities)
    if method == "uncalibrated":
        return p
    if method == "sigmoid":
        return calibrator.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.clip(calibrator.predict(p), 0, 1)
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_metrics(target, probabilities, n_bins=15):
    """Ranking, proper scoring, bin counts/ECE, and descriptive slope/intercept."""
    y = np.asarray(target, dtype=int).reshape(-1)
    p = _safe_probabilities(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("target and probabilities must have equal non-zero length")
    metrics = {
        "n": len(y), "default_rate": float(y.mean()), "calibration_bin_count": int(n_bins),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    calibration_bins = []
    ece = 0.0
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        if not selected.any():
            continue
        mean_pred = float(p[selected].mean())
        observed = float(y[selected].mean())
        count = int(selected.sum())
        ece += (count / len(y)) * abs(observed - mean_pred)
        calibration_bins.append({"bin": bin_id + 1, "lower": edges[bin_id],
                                 "upper": edges[bin_id + 1], "n": count,
                                 "mean_predicted": mean_pred, "observed_rate": observed})
    metrics["ece_equal_width"] = float(ece)

    x = logit(p)
    def objective(parameters):
        eta = np.clip(parameters[0] + parameters[1] * x, -35, 35)
        return float(np.logaddexp(0, eta).sum() - np.dot(y, eta))
    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    metrics["calibration_intercept"] = float(fit.x[0])
    metrics["calibration_slope"] = float(fit.x[1])
    return metrics, pd.DataFrame(calibration_bins)
