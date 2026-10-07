"""Reconstruct the selected model and validation-fitted calibrator."""

import json
from pathlib import Path

from src.calibration import apply_calibrator, fit_calibrator
from src.models import FEATURE_GROUPS
from src.nonlinear import _new_model, _tree_features


def fit_hgb_with_validation_sigmoid(feature_group, X_train, y_train, X_val, y_val,
                                    reports_dir="reports"):
    """Fit selected HGB parameters on train and the sigmoid mapping on validation."""
    features = FEATURE_GROUPS[feature_group]
    param_path = Path(reports_dir) / "tables" / "hgb_parameters.json"
    params = json.loads(param_path.read_text(encoding="utf-8"))["parameters"]
    model = _new_model(params)
    train = _tree_features(X_train, features)
    validation = _tree_features(X_val, features)
    model.fit(train, y_train)
    p_val = model.predict_proba(validation)[:, 1]
    calibrator = fit_calibrator("sigmoid", p_val, y_val)
    return {
        "model": model,
        "calibrator": calibrator,
        "features": features,
        "predict": lambda frame: apply_calibrator(
            "sigmoid", calibrator, model.predict_proba(_tree_features(frame, features))[:, 1]
        ),
        "predict_raw": lambda frame: model.predict_proba(_tree_features(frame, features))[:, 1],
    }
