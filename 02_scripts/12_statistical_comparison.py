import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu

from project_paths import (
    DIABETES_RESULTS_DIR,
    STATISTICAL_RESULTS_DIR,
)

# ============================================================
# SETTINGS
# ============================================================

RL_FILE = (
    DIABETES_RESULTS_DIR
    / "10_rl_30_seed_results.csv"
)

BASELINE_FILE = (
    DIABETES_RESULTS_DIR
    / "11_30seed_baseline_results.csv"
)

LAMBDA_PENALTY = 0.02
TOTAL_FEATURES = 21


# ============================================================
# 1. LOAD RESULTS
# ============================================================

rl = pd.read_csv(RL_FILE)
baseline = pd.read_csv(BASELINE_FILE)


# ============================================================
# 2. PREPARE RL RESULTS
# ============================================================

rl["Method"] = "RL"

rl["Test_AUC"] = rl["ROC_AUC"]

rl["Test_Objective"] = (
    rl["Test_AUC"]
    - LAMBDA_PENALTY
    * (rl["Features"] / TOTAL_FEATURES)
)

rl["Reduction"] = (
    1
    - rl["Features"] / TOTAL_FEATURES
) * 100


# ============================================================
# 3. CHECK DATA
# ============================================================

print("=" * 90)
print("DATA CHECK")
print("=" * 90)

print("\nRL runs:")
print(len(rl))

print("\nBaseline runs:")
print(len(baseline))

print("\nBaseline methods:")
print(
    baseline["Method"]
    .value_counts()
)


# ============================================================
# 4. COMBINED DESCRIPTIVE SUMMARY
# ============================================================

rl_for_summary = rl[
    [
        "Method",
        "Test_AUC",
        "Features",
        "Reduction",
        "Test_Objective"
    ]
].copy()

baseline_for_summary = baseline[
    [
        "Method",
        "Test_AUC",
        "Features",
        "Reduction",
        "Test_Objective"
    ]
].copy()


combined = pd.concat(
    [
        rl_for_summary,
        baseline_for_summary
    ],
    ignore_index=True
)


summary = (
    combined
    .groupby("Method")
    .agg(
        Mean_AUC=("Test_AUC", "mean"),
        Std_AUC=("Test_AUC", "std"),
        Median_AUC=("Test_AUC", "median"),

        Mean_Features=("Features", "mean"),
        Std_Features=("Features", "std"),

        Mean_Reduction=("Reduction", "mean"),

        Mean_Objective=("Test_Objective", "mean"),
        Std_Objective=("Test_Objective", "std")
    )
    .reset_index()
)


print("\n" + "=" * 90)
print("COMBINED SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "Mean_AUC": "{:.4f}".format,
            "Std_AUC": "{:.4f}".format,
            "Median_AUC": "{:.4f}".format,
            "Mean_Features": "{:.2f}".format,
            "Std_Features": "{:.2f}".format,
            "Mean_Reduction": "{:.1f}".format,
            "Mean_Objective": "{:.4f}".format,
            "Std_Objective": "{:.4f}".format
        }
    )
)


# ============================================================
# 5. MANN-WHITNEY U TESTS
# ============================================================
#
# We use Mann-Whitney U because these are repeated stochastic
# runs. We do not assume the metric distributions are normal.
#
# H0:
# RL and baseline come from the same distribution.
#
# p < 0.05:
# evidence of a statistically significant difference.
# ============================================================

baseline_methods = [
    "RANDOM",
    "ADD",
    "REMOVE",
    "SWAP",
    "FLIP"
]


statistical_results = []


for method in baseline_methods:

    method_data = baseline[
        baseline["Method"] == method
    ]

    # --------------------------------------------------------
    # ROC-AUC comparison
    # --------------------------------------------------------

    auc_stat, auc_p = mannwhitneyu(
        rl["Test_AUC"],
        method_data["Test_AUC"],
        alternative="two-sided"
    )


    # --------------------------------------------------------
    # Number of features comparison
    # --------------------------------------------------------

    feature_stat, feature_p = mannwhitneyu(
        rl["Features"],
        method_data["Features"],
        alternative="two-sided"
    )


    # --------------------------------------------------------
    # Combined objective comparison
    # --------------------------------------------------------

    objective_stat, objective_p = mannwhitneyu(
        rl["Test_Objective"],
        method_data["Test_Objective"],
        alternative="two-sided"
    )


    statistical_results.append({
        "Comparison": f"RL vs {method}",

        "RL_Mean_AUC":
            rl["Test_AUC"].mean(),

        "Baseline_Mean_AUC":
            method_data["Test_AUC"].mean(),

        "AUC_p_value":
            auc_p,

        "RL_Mean_Features":
            rl["Features"].mean(),

        "Baseline_Mean_Features":
            method_data["Features"].mean(),

        "Features_p_value":
            feature_p,

        "RL_Mean_Objective":
            rl["Test_Objective"].mean(),

        "Baseline_Mean_Objective":
            method_data["Test_Objective"].mean(),

        "Objective_p_value":
            objective_p
    })


stats_df = pd.DataFrame(
    statistical_results
)


# ============================================================
# 6. INTERPRET SIGNIFICANCE
# ============================================================

def significance_label(p):

    if p < 0.001:
        return "Highly significant"

    elif p < 0.01:
        return "Very significant"

    elif p < 0.05:
        return "Significant"

    else:
        return "Not significant"


stats_df["AUC_Significance"] = (
    stats_df["AUC_p_value"]
    .apply(significance_label)
)

stats_df["Feature_Significance"] = (
    stats_df["Features_p_value"]
    .apply(significance_label)
)

stats_df["Objective_Significance"] = (
    stats_df["Objective_p_value"]
    .apply(significance_label)
)


# ============================================================
# 7. PRINT STATISTICAL RESULTS
# ============================================================

print("\n" + "=" * 90)
print("MANN-WHITNEY U STATISTICAL COMPARISON")
print("=" * 90)


for _, row in stats_df.iterrows():

    print("\n" + "-" * 90)

    print(row["Comparison"])

    print("-" * 90)

    print(
        f"ROC-AUC:"
        f"\n  RL mean       = {row['RL_Mean_AUC']:.4f}"
        f"\n  Baseline mean = {row['Baseline_Mean_AUC']:.4f}"
        f"\n  p-value       = {row['AUC_p_value']:.6f}"
        f"\n  Result        = {row['AUC_Significance']}"
    )

    print(
        f"\nFeatures:"
        f"\n  RL mean       = {row['RL_Mean_Features']:.2f}"
        f"\n  Baseline mean = {row['Baseline_Mean_Features']:.2f}"
        f"\n  p-value       = {row['Features_p_value']:.6f}"
        f"\n  Result        = {row['Feature_Significance']}"
    )

    print(
        f"\nObjective:"
        f"\n  RL mean       = {row['RL_Mean_Objective']:.4f}"
        f"\n  Baseline mean = {row['Baseline_Mean_Objective']:.4f}"
        f"\n  p-value       = {row['Objective_p_value']:.6f}"
        f"\n  Result        = {row['Objective_Significance']}"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

SUMMARY_FILE = (
    STATISTICAL_RESULTS_DIR
    / "12_combined_summary.csv"
)

STATS_FILE = (
    STATISTICAL_RESULTS_DIR
    / "12_statistical_comparison.csv"
)

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print("12_combined_summary.csv")
print("12_statistical_comparison.csv")