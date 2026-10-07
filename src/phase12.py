"""Practical global and local explanations for the frozen HGB Model C."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.frozen_model import fit_hgb_with_validation_sigmoid
from src.models import FEATURE_GROUPS, load_and_split_data
from src.nonlinear import _tree_features


MODEL_C = "Application + Bureau + Alternative"
GROUPS = {
    "Application": FEATURE_GROUPS["Application"],
    "Traditional Bureau": FEATURE_GROUPS["Application + Bureau"][len(FEATURE_GROUPS["Application"]):],
    "Alternative Behaviour": FEATURE_GROUPS[MODEL_C][len(FEATURE_GROUPS["Application + Bureau"]):],
}
FEATURE_LABELS = {
    "APP_INCOME_TOTAL": "Current total income", "APP_CREDIT_AMOUNT": "Current credit amount",
    "APP_ANNUITY": "Current scheduled annuity", "APP_DAYS_BIRTH": "Age at application",
    "APP_DAYS_EMPLOYED": "Recorded employment duration (sentinel treated as missing)",
    "APP_EXT_SOURCE_1": "External risk score 1", "APP_EXT_SOURCE_2": "External risk score 2",
    "APP_EXT_SOURCE_3": "External risk score 3", "APP_CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "APP_ANNUITY_INCOME_RATIO": "Annuity-to-income ratio", "BUREAU_CREDIT_COUNT": "Bureau credit count",
    "BUREAU_ACTIVE_COUNT": "Active bureau credit count", "BUREAU_TOTAL_CREDIT": "Total bureau credit",
    "BUREAU_TOTAL_DEBT": "Total bureau debt", "BUREAU_DEBT_RATIO": "Bureau debt ratio",
    "BUREAU_AVG_DAYS_CREDIT": "Average bureau credit age",
    "PREV_APP_COUNT": "Previous application count", "PREV_APPROVED_COUNT": "Previous approvals",
    "PREV_REFUSED_COUNT": "Previous refusals", "INST_TOTAL_COUNT": "Installment record count",
    "INST_LATE_PAYMENT_COUNT": "Observed late-payment count",
    "INST_LATE_PAYMENT_RATIO": "Observed late-payment ratio",
    "INST_UNDERPAYMENT_COUNT": "Observed underpayment count",
    "INST_PAYMENT_MISSING_COUNT": "Installment records with missing actual-payment details",
}


def _feature_group(feature):
    for group, features in GROUPS.items():
        if feature in features:
            return group
    return "Other"


def _reason(feature, sign):
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    direction = "raises" if sign > 0 else "lowers"
    return f"{label} {direction} this borrower's model PD relative to replacing it with the training median."


def run_phase12(filepath="data/processed/feature_master.csv", split_dir="data/processed/splits",
                output_dir="reports"):
    out = Path(output_dir)
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, _ = load_and_split_data(filepath, split_dir)
    bundle = fit_hgb_with_validation_sigmoid(MODEL_C, X_train, y_train, X_val, y_val, output_dir)
    features = bundle["features"]
    model = bundle["model"]
    val_features = _tree_features(X_val, features)
    base_raw = model.predict_proba(val_features)[:, 1]
    base_ap = average_precision_score(y_val, base_raw)
    base_auc = roc_auc_score(y_val, base_raw)

    # Jointly permute each conceptual feature group using the same row order,
    # preserving within-group patterns while disrupting its target association.
    rng = np.random.RandomState(42)
    group_rows = []
    for group, columns in GROUPS.items():
        drops_ap, drops_auc = [], []
        for _ in range(3):
            permutation = rng.permutation(len(val_features))
            perturbed = val_features.copy()
            perturbed.loc[:, columns] = val_features.iloc[permutation][columns].to_numpy()
            p = model.predict_proba(perturbed)[:, 1]
            drops_ap.append(base_ap - average_precision_score(y_val, p))
            drops_auc.append(base_auc - roc_auc_score(y_val, p))
        group_rows.append({"feature_group": group, "validation_AP_drop_mean": np.mean(drops_ap),
                           "validation_AP_drop_std": np.std(drops_ap, ddof=1),
                           "validation_ROC_AUC_drop_mean": np.mean(drops_auc),
                           "validation_ROC_AUC_drop_std": np.std(drops_auc, ddof=1),
                           "permutations": 3})
    group_importance = pd.DataFrame(group_rows).sort_values("validation_AP_drop_mean", ascending=False)
    group_importance.to_csv(tables / "phase12_group_permutation_importance.csv", index=False)

    individual = pd.read_csv(tables / "phase9_validation_permutation_importance.csv")
    individual["feature_group"] = individual.feature.map(_feature_group)
    individual.to_csv(tables / "phase12_validation_feature_importance.csv", index=False)
    alternative = individual[individual.feature_group == "Alternative Behaviour"].sort_values(
        "importance_mean", ascending=False)
    alternative.to_csv(tables / "phase12_alternative_feature_importance.csv", index=False)

    # Select the median-risk case inside each cohort's top/bottom decile. The
    # rule uses predicted risk and cohort only; it does not inspect outcomes.
    test_features = _tree_features(X_test, features)
    p_test = bundle["predict"](X_test)
    thin = X_test.BUREAU_CREDIT_COUNT.eq(0).to_numpy()
    example_rows = []
    for cohort, cohort_mask in (("Thin-file", thin), ("Non-thin-file", ~thin)):
        indices = np.flatnonzero(cohort_mask)
        cohort_risk = p_test[indices]
        for risk_level, quantile, choose in (("High risk", 0.90, "median_top_decile"),
                                             ("Low risk", 0.10, "median_bottom_decile")):
            cutoff = np.quantile(cohort_risk, quantile)
            eligible = indices[cohort_risk >= cutoff] if quantile == 0.90 else indices[cohort_risk <= cutoff]
            median_risk = np.median(p_test[eligible])
            selected = min(eligible, key=lambda i: (abs(p_test[i] - median_risk), int(X_test.iloc[i].SK_ID_CURR)))
            example_rows.append({"cohort": cohort, "risk_band": risk_level, "selection_rule": choose,
                                 "row_index": int(selected), "SK_ID_CURR": int(X_test.iloc[selected].SK_ID_CURR),
                                 "predicted_PD_sigmoid": float(p_test[selected])})
    examples = pd.DataFrame(example_rows)
    examples.drop(columns="row_index").to_csv(
        tables / "phase12_representative_borrowers.csv", index=False)

    train_transformed = _tree_features(X_train, features)
    reference = train_transformed.median(axis=0, skipna=True)
    local_rows = []
    for ex in examples.itertuples(index=False):
        row = test_features.iloc[[ex.row_index]].copy()
        original_pd = p_test[ex.row_index]
        for feature in features:
            changed = row.copy()
            changed.loc[:, feature] = reference[feature]
            replacement_pd = bundle["predict"](changed)
            delta = float(original_pd - replacement_pd[0])
            local_rows.append({"cohort": ex.cohort, "risk_band": ex.risk_band,
                               "SK_ID_CURR": ex.SK_ID_CURR, "predicted_PD": original_pd,
                               "feature": feature, "feature_group": _feature_group(feature),
                               "observed_value": row.iloc[0][feature],
                               "training_median_replacement": reference[feature],
                               "PD_change_vs_median_replacement": delta,
                               "direction": "raises risk vs median" if delta > 0 else "lowers risk vs median",
                               "reason_code": _reason(feature, delta)})
    local = pd.DataFrame(local_rows)
    local.to_csv(tables / "phase12_local_feature_replacements.csv", index=False)
    top_local = pd.concat([
        local.sort_values("PD_change_vs_median_replacement").groupby("SK_ID_CURR").head(5),
        local.sort_values("PD_change_vs_median_replacement", ascending=False).groupby("SK_ID_CURR").head(5),
    ]).drop_duplicates(["SK_ID_CURR", "feature"])
    top_local.to_csv(tables / "phase12_reason_codes.csv", index=False)

    top = individual.sort_values("importance_mean", ascending=False).head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.feature, top.importance_mean, xerr=top.importance_std, color="#4C78A8")
    ax.set(title="Model C validation permutation importance", xlabel="Average-precision decrease")
    fig.tight_layout()
    fig.savefig(figures / "phase12_global_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(group_importance.feature_group, group_importance.validation_AP_drop_mean,
           yerr=group_importance.validation_AP_drop_std, color="#72A0C1")
    ax.set(title="Joint feature-group permutation on validation", ylabel="Average-precision decrease")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "phase12_group_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"group_importance": group_importance, "individual_importance": individual,
            "alternative_importance": alternative, "examples": examples, "local_effects": local}


if __name__ == "__main__":
    result = run_phase12()
    print(result["group_importance"].to_string(index=False))
    print(result["examples"].to_string(index=False))
