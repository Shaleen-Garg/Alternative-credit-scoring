"""Shared feature definitions, deterministic splits, and logistic pipelines."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APPLICATION_FEATURES = [
    "APP_INCOME_TOTAL", "APP_CREDIT_AMOUNT", "APP_ANNUITY",
    "APP_DAYS_BIRTH", "APP_DAYS_EMPLOYED", "APP_EXT_SOURCE_1",
    "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_CREDIT_INCOME_RATIO",
    "APP_ANNUITY_INCOME_RATIO",
]
BUREAU_FEATURES = [
    "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_TOTAL_CREDIT",
    "BUREAU_TOTAL_DEBT", "BUREAU_DEBT_RATIO", "BUREAU_AVG_DAYS_CREDIT",
]
ALTERNATIVE_FEATURES = [
    "PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT",
    "INST_TOTAL_COUNT", "INST_LATE_PAYMENT_COUNT", "INST_LATE_PAYMENT_RATIO",
    "INST_UNDERPAYMENT_COUNT",
]
FEATURE_GROUPS = {
    "Application": APPLICATION_FEATURES,
    "Application + Bureau": APPLICATION_FEATURES + BUREAU_FEATURES,
    "Application + Bureau + Alternative": APPLICATION_FEATURES + BUREAU_FEATURES + ALTERNATIVE_FEATURES,
}
SPLIT_SEED = 42
THIN_FILE_COLUMN = "BUREAU_CREDIT_COUNT"
THIN_FILE_VALUE = 0


def thin_file_mask(frame):
    """Return the project's locked thin-file cohort mask."""
    if THIN_FILE_COLUMN not in frame:
        raise ValueError(f"Thin-file definition requires {THIN_FILE_COLUMN}")
    return frame[THIN_FILE_COLUMN].eq(THIN_FILE_VALUE)


def load_and_split_data(filepath, random_state=SPLIT_SEED):
    """Load features and recreate Phase 7A's exact two-stage stratified split."""
    df = pd.read_csv(filepath)
    required = ["SK_ID_CURR", "TARGET", *APPLICATION_FEATURES, *BUREAU_FEATURES, *ALTERNATIVE_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    X = df[["SK_ID_CURR", *APPLICATION_FEATURES, *BUREAU_FEATURES, *ALTERNATIVE_FEATURES]].copy()
    y = df["TARGET"].copy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(features):
    """Build the common train-fitted numerical preprocessing + LR model."""
    features = list(features)
    if not features or "TARGET" in features or "SK_ID_CURR" in features:
        raise ValueError("Predictors must be non-empty and exclude TARGET/SK_ID_CURR")
    if len(features) != len(set(features)):
        raise ValueError("Predictor list contains duplicate feature names")
    ext_cols = [c for c in features if c in {"APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3"}]
    bureau_cols = [c for c in features if c in BUREAU_FEATURES]
    other_cols = [c for c in features if c not in ext_cols and c not in bureau_cols]
    transformers = []
    if ext_cols:
        transformers.append(("ext", Pipeline([
            ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), ext_cols))
    if other_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), other_cols))
    if bureau_cols:
        transformers.append(("bureau", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), bureau_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("anomaly", DaysEmployedAnomalyHandler()),
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
    ])


class DaysEmployedAnomalyHandler(BaseEstimator, TransformerMixin):
    """Convert Home Credit's sentinel employment value before imputation."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame) and "APP_DAYS_EMPLOYED" in X:
            X.loc[X["APP_DAYS_EMPLOYED"] == 365243, "APP_DAYS_EMPLOYED"] = np.nan
        return X

def build_baseline_pipeline():
    """Backward-compatible Phase 7A constructor."""
    return build_pipeline(APPLICATION_FEATURES)


def get_constant_baseline_predictions(y_train, n_samples):
    return np.full(n_samples, y_train.mean())
