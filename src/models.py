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
    "INST_UNDERPAYMENT_COUNT", "INST_PAYMENT_MISSING_COUNT",
]
FEATURE_GROUPS = {
    "Application": APPLICATION_FEATURES,
    "Application + Bureau": APPLICATION_FEATURES + BUREAU_FEATURES,
    "Application + Bureau + Alternative": APPLICATION_FEATURES + BUREAU_FEATURES + ALTERNATIVE_FEATURES,
}
SPLIT_SEED = 42
SPLIT_DIR = Path("data/processed/splits")
THIN_FILE_COLUMN = "BUREAU_CREDIT_COUNT"
THIN_FILE_VALUE = 0


def thin_file_mask(frame):
    """Return the project's locked thin-file cohort mask."""
    if THIN_FILE_COLUMN not in frame:
        raise ValueError(f"Thin-file definition requires {THIN_FILE_COLUMN}")
    return frame[THIN_FILE_COLUMN].eq(THIN_FILE_VALUE)


def create_frozen_split_artifacts(filepath, output_dir=SPLIT_DIR):
    """Create borrower ID files using the project's stratified split procedure."""
    df = pd.read_csv(filepath)
    if "SK_ID_CURR" not in df or "TARGET" not in df:
        raise ValueError("Split generation requires SK_ID_CURR and TARGET")
    if df.SK_ID_CURR.duplicated().any():
        raise ValueError("SK_ID_CURR must be unique before generating split IDs")
    X = df[["SK_ID_CURR"]]
    y = df["TARGET"].copy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SPLIT_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val,
        random_state=SPLIT_SEED,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", X_train), ("validation", X_val), ("test", X_test)):
        split[["SK_ID_CURR"]].to_csv(destination / f"{name}_ids.csv", index=False)
    return {"train": X_train.SK_ID_CURR.to_numpy(), "validation": X_val.SK_ID_CURR.to_numpy(),
            "test": X_test.SK_ID_CURR.to_numpy()}


def load_split_ids(population_ids, split_dir=SPLIT_DIR):
    """Load explicit split ID files and validate them against the modeling population."""
    directory = Path(split_dir)
    population = pd.Index(population_ids)
    if population.has_duplicates:
        raise ValueError("Modeling population contains duplicate SK_ID_CURR values")
    split_ids = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}_ids.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required explicit split artifact not found: {path}")
        ids = pd.read_csv(path)
        if list(ids.columns) != ["SK_ID_CURR"] or ids.SK_ID_CURR.duplicated().any():
            raise ValueError(f"Invalid or duplicate IDs in {path}")
        split_ids[name] = ids.SK_ID_CURR.tolist()
    sets = {name: set(ids) for name, ids in split_ids.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("Train, validation, and test split IDs overlap")
    if set.union(*sets.values()) != set(population):
        raise ValueError("Split ID union does not equal the current modeling population")
    return split_ids


def load_and_split_data(filepath, split_dir=SPLIT_DIR):
    """Load model data in the persisted borrower-ID split order."""
    df = pd.read_csv(filepath)
    required = ["SK_ID_CURR", "TARGET", *APPLICATION_FEATURES, *BUREAU_FEATURES, *ALTERNATIVE_FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    if df.SK_ID_CURR.duplicated().any():
        raise ValueError("SK_ID_CURR must be unique in the feature dataset")
    split_ids = load_split_ids(df.SK_ID_CURR, split_dir)
    indexed = df.set_index("SK_ID_CURR", drop=False)
    X_parts, y_parts = [], []
    for name in ("train", "validation", "test"):
        part = indexed.loc[split_ids[name]]
        X_parts.append(part[["SK_ID_CURR", *APPLICATION_FEATURES, *BUREAU_FEATURES, *ALTERNATIVE_FEATURES]].copy())
        y_parts.append(part["TARGET"].copy())
    if len(df) == 307511 and [len(x) for x in X_parts] != [215257, 46127, 46127]:
        raise ValueError("Persisted split sizes do not match the verified modeling population")
    return (*X_parts, *y_parts)


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
    """Build the established logistic baseline pipeline."""
    return build_pipeline(APPLICATION_FEATURES)


def get_constant_baseline_predictions(y_train, n_samples):
    return np.full(n_samples, y_train.mean())
