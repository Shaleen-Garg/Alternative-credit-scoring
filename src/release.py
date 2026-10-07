"""Build the small, curated set of final portfolio tables and figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FINAL = ROOT / "reports" / "final"


def run_release_assets() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(TABLES / "phase9_model_comparison.csv")
    comparison.to_csv(FINAL / "model_comparison.csv", index=False)

    cohort = pd.read_csv(TABLES / "phase13_cohort_performance.csv")
    cohort = cohort[(cohort["split"] == "test") & cohort["model"].isin(
        ["Application + Bureau", "Application + Bureau + Alternative"]
    )]
    cohort.to_csv(FINAL / "test_cohort_comparison.csv", index=False)

    calibration = pd.read_csv(TABLES / "phase10_calibration_test_metrics.csv")
    calibration = calibration[calibration["method"].isin(["uncalibrated", "sigmoid"])]
    calibration.to_csv(FINAL / "calibration_summary.csv", index=False)

    decision = pd.read_csv(TABLES / "phase11_same_approval_rate_comparison.csv")
    decision = decision[decision["target_validation_approval_rate"] == 0.7]
    decision.to_csv(FINAL / "decision_at_70pct_approval.csv", index=False)

    bootstrap = pd.read_csv(TABLES / "phase13_model_c_minus_b_bootstrap_ci.csv")
    bootstrap.to_csv(FINAL / "paired_bootstrap_intervals.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"Logistic Regression": "#3973ac", "HistGradientBoostingClassifier": "#e08b36"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    labels = comparison["model"] + "\n" + comparison["model_type"].replace(
        {"HistGradientBoostingClassifier": "HGB", "Logistic Regression": "Logistic"}
    )
    x = range(len(comparison))
    for ax, metric, title in zip(axes, ["test_roc_auc", "test_pr_auc"], ["ROC-AUC", "PR-AUC"]):
        bars = ax.barh(list(x), comparison[metric], color=[colors[v] for v in comparison["model_type"]])
        ax.set_yticks(list(x), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(comparison[metric]) * 1.16)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.suptitle("Frozen test performance by feature group and model", fontsize=14)
    fig.savefig(FINAL / "model_comparison.png", dpi=180)
    plt.close(fig)

    linear = pd.read_csv(TABLES / "phase8_ablation_summary.csv")
    nonlinear = pd.read_csv(TABLES / "phase9_nonlinear_cohort_metrics.csv")
    linear = linear.rename(columns={"model": "feature_group", "thin_file_roc_auc": "thin_roc_auc", "thin_file_pr_auc": "thin_pr_auc"})
    nonlinear = nonlinear[(nonlinear["split"] == "test") & (nonlinear["cohort"] == "Thin-file")]
    nonlinear = nonlinear.rename(columns={"model": "feature_group", "roc_auc": "thin_roc_auc", "pr_auc": "thin_pr_auc"})
    nonlinear["learner"] = "HGB"
    linear["learner"] = "Logistic"
    thin = pd.concat([
        linear[["feature_group", "learner", "thin_roc_auc", "thin_pr_auc"]],
        nonlinear[["feature_group", "learner", "thin_roc_auc", "thin_pr_auc"]],
    ], ignore_index=True)
    thin.to_csv(FINAL / "thin_file_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    short_names = thin["feature_group"].str.replace("Application + Bureau + Alternative", "A+B+Alt", regex=False).str.replace("Application + Bureau", "A+B", regex=False).str.replace("Application", "A", regex=False)
    yy = range(len(thin))
    for ax, metric, title in zip(axes, ["thin_roc_auc", "thin_pr_auc"], ["Thin-file ROC-AUC", "Thin-file PR-AUC"]):
        bars = ax.barh(list(yy), thin[metric], color=[colors["Logistic Regression"] if x == "Logistic" else colors["HistGradientBoostingClassifier"] for x in thin["learner"]])
        ax.set_yticks(list(yy), [f"{n} · {l}" for n, l in zip(short_names, thin["learner"])])
        ax.invert_yaxis()
        ax.set_xlim(0, max(thin[metric]) * 1.16)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.suptitle("Thin-file test performance across feature groups", fontsize=14)
    fig.savefig(FINAL / "thin_file_comparison.png", dpi=180)
    plt.close(fig)

    reliability = pd.read_csv(TABLES / "phase10_calibration_reliability_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, group in zip(axes, ["Overall", "Thin-file"]):
        rows = reliability[(reliability["cohort"] == group) & reliability["method"].isin(["uncalibrated", "sigmoid"])]
        for method, series in rows.groupby("method"):
            ax.plot(series["mean_predicted"], series["observed_rate"], marker="o", ms=3, label=method.capitalize())
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(xlim=(0, .6), ylim=(0, .6), xlabel="Mean predicted PD", ylabel="Observed default rate", title=group)
        ax.legend(frameon=False)
    fig.suptitle("Test-set reliability by cohort", fontsize=14)
    fig.savefig(FINAL / "calibration_reliability.png", dpi=180)
    plt.close(fig)

    decision = pd.read_csv(FINAL / "decision_at_70pct_approval.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    for ax, group in zip(axes, ["Overall", "Thin-file"]):
        rows = decision[decision["cohort"] == group]
        for _, row in rows.iterrows():
            ax.scatter(row["test_approval_rate"], row["test_default_rate_among_approved"], s=70, label=row["model"])
        ax.set(xlim=(.65, .75), ylim=(0, .08), xlabel="Test approval rate", ylabel="Default rate among approved", title=group)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Illustrative operating points near 70% validation approval", fontsize=14)
    fig.savefig(FINAL / "decision_tradeoff.png", dpi=180)
    plt.close(fig)

    importance = pd.read_csv(TABLES / "phase12_validation_feature_importance.csv").nlargest(12, "importance_mean").sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.7), constrained_layout=True)
    bars = ax.barh(importance["feature"], importance["importance_mean"], color="#3973ac")
    ax.set(xlabel="Decrease in validation average precision", title="Top validation permutation importance")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.savefig(FINAL / "feature_importance.png", dpi=180)
    plt.close(fig)

    scores = pd.read_csv(TABLES / "phase13_prediction_stability.csv")
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for cohort_name, rows in scores.groupby("cohort"):
        rows = rows.set_index("split").reindex(["train", "validation", "test"])
        ax.plot(["Train", "Validation", "Test"], rows["mean"], marker="o", label=cohort_name)
    ax.set(ylabel="Mean predicted PD", title="Mean score by split and cohort")
    ax.legend(frameon=False)
    ax.text(.01, -.23, "Training scores are in-sample; held-out splits are validation and test.", transform=ax.transAxes, fontsize=8)
    fig.savefig(FINAL / "prediction_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_release_assets()
