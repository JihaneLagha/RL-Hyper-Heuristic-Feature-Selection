import pandas as pd

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from project_paths import HEART_RESULTS_DIR


# ============================================================
# SETTINGS
# ============================================================

RL_FILE = (
    HEART_RESULTS_DIR
    / "19_heart_repeated_rl_results.csv"
)

BASELINE_FILE = (
    HEART_RESULTS_DIR
    / "24_heart_conventional_results.csv"
)

TOTAL_FEATURES = 13

LAMBDA_PENALTY = 0.02


COMPARISON_METHODS = [
    "ALL_FEATURES",
    "L1",
    "RFE",
    "SelectKBest",
]


# ============================================================
# 1. LOAD RESULTS
# ============================================================

rl = pd.read_csv(
    RL_FILE
)

baseline = pd.read_csv(
    BASELINE_FILE
)


# ============================================================
# 2. PREPARE RL RESULTS
# ============================================================

rl["Method"] = "RL"

rl["Test_AUC"] = (
    rl["RL_Test_AUC"]
)

rl["Features"] = (
    rl["RL_Features"]
)

rl["Objective"] = (
    rl["Test_AUC"]
    - LAMBDA_PENALTY
    * (
        rl["Features"]
        / TOTAL_FEATURES
    )
)


# ============================================================
# 3. DATA CHECK
# ============================================================

print("=" * 100)
print("FINAL HEART DISEASE METHOD COMPARISON")
print("=" * 100)

print(
    "\nRL runs:",
    len(rl),
)

print(
    "RL seeds:",
    rl["Seed"].nunique(),
)

print(
    "\nConventional runs:",
    len(baseline),
)

print(
    "Conventional methods:"
)

print(
    baseline["Method"]
    .value_counts()
)


# ============================================================
# 4. COMBINED SUMMARY
# ============================================================

rl_summary_data = rl[
    [
        "Seed",
        "Method",
        "Features",
        "Test_AUC",
        "Objective",
    ]
].copy()


baseline_summary_data = baseline[
    [
        "Seed",
        "Method",
        "Features",
        "Test_AUC",
        "Objective",
    ]
].copy()


combined = pd.concat(
    [
        rl_summary_data,
        baseline_summary_data,
    ],
    ignore_index=True,
)


combined["Reduction"] = (
    1
    - combined["Features"]
    / TOTAL_FEATURES
) * 100


summary = (
    combined
    .groupby("Method")
    .agg(
        Mean_AUC=(
            "Test_AUC",
            "mean",
        ),
        Std_AUC=(
            "Test_AUC",
            "std",
        ),
        Mean_Features=(
            "Features",
            "mean",
        ),
        Mean_Reduction=(
            "Reduction",
            "mean",
        ),
        Mean_Objective=(
            "Objective",
            "mean",
        ),
        Std_Objective=(
            "Objective",
            "std",
        ),
    )
    .reset_index()
)


print(
    "\n" + "=" * 100
)

print(
    "COMBINED HEART DISEASE SUMMARY"
)

print("=" * 100)


print(
    summary.to_string(
        index=False,
        formatters={
            "Mean_AUC":
                "{:.4f}".format,
            "Std_AUC":
                "{:.4f}".format,
            "Mean_Features":
                "{:.2f}".format,
            "Mean_Reduction":
                "{:.1f}%".format,
            "Mean_Objective":
                "{:.4f}".format,
            "Std_Objective":
                "{:.4f}".format,
        },
    )
)


# ============================================================
# 5. PAIRED TESTS
# ============================================================

results = []


for method in COMPARISON_METHODS:

    other = (
        baseline[
            baseline["Method"] == method
        ]
        .copy()
        .sort_values("Seed")
    )


    rl_sorted = (
        rl
        .copy()
        .sort_values("Seed")
    )


    paired = pd.merge(
        rl_sorted,
        other,
        on="Seed",
        suffixes=(
            "_RL",
            "_OTHER",
        ),
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    auc_difference = (
        paired["Test_AUC_RL"]
        - paired["Test_AUC_OTHER"]
    )


    auc_statistic, auc_p = wilcoxon(
        paired["Test_AUC_RL"],
        paired["Test_AUC_OTHER"],
        alternative="two-sided",
    )


    auc_wins = int(
        (auc_difference > 0).sum()
    )

    auc_losses = int(
        (auc_difference < 0).sum()
    )

    auc_ties = int(
        (auc_difference == 0).sum()
    )


    # ========================================================
    # FEATURES
    # ========================================================

    feature_difference = (
        paired["Features_RL"]
        - paired["Features_OTHER"]
    )


    feature_statistic, feature_p = wilcoxon(
        paired["Features_RL"],
        paired["Features_OTHER"],
        alternative="two-sided",
        zero_method="wilcox",
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    objective_difference = (
        paired["Objective_RL"]
        - paired["Objective_OTHER"]
    )


    objective_statistic, objective_p = wilcoxon(
        paired["Objective_RL"],
        paired["Objective_OTHER"],
        alternative="two-sided",
    )


    objective_wins = int(
        (objective_difference > 0).sum()
    )

    objective_losses = int(
        (objective_difference < 0).sum()
    )

    objective_ties = int(
        (objective_difference == 0).sum()
    )


    # ========================================================
    # PRINT DETAILED RESULTS
    # ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        f"RL vs {method}"
    )

    print("=" * 100)


    print("\nROC-AUC")
    print("-" * 50)

    print(
        f"RL mean             : "
        f"{paired['Test_AUC_RL'].mean():.4f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 13-len(method))}: "
        f"{paired['Test_AUC_OTHER'].mean():.4f}"
    )

    print(
        f"Mean difference     : "
        f"{auc_difference.mean():+.6f}"
    )

    print(
        f"Median difference   : "
        f"{auc_difference.median():+.6f}"
    )

    print(
        f"RL wins/losses/ties : "
        f"{auc_wins}/{auc_losses}/{auc_ties}"
    )

    print(
        f"Wilcoxon p-value    : "
        f"{auc_p:.8f}"
    )


    print("\nFEATURE COUNT")
    print("-" * 50)

    print(
        f"RL mean             : "
        f"{paired['Features_RL'].mean():.2f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 13-len(method))}: "
        f"{paired['Features_OTHER'].mean():.2f}"
    )

    print(
        f"Mean difference     : "
        f"{feature_difference.mean():+.2f}"
    )

    print(
        f"Wilcoxon p-value    : "
        f"{feature_p:.8f}"
    )


    print("\nOBJECTIVE")
    print("-" * 50)

    print(
        f"RL mean             : "
        f"{paired['Objective_RL'].mean():.4f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 13-len(method))}: "
        f"{paired['Objective_OTHER'].mean():.4f}"
    )

    print(
        f"Mean difference     : "
        f"{objective_difference.mean():+.6f}"
    )

    print(
        f"RL wins/losses/ties : "
        f"{objective_wins}/"
        f"{objective_losses}/"
        f"{objective_ties}"
    )

    print(
        f"Wilcoxon p-value    : "
        f"{objective_p:.8f}"
    )


    results.append({
        "Comparison":
            f"RL vs {method}",

        "RL_Mean_AUC":
            paired[
                "Test_AUC_RL"
            ].mean(),

        "Other_Mean_AUC":
            paired[
                "Test_AUC_OTHER"
            ].mean(),

        "AUC_Difference":
            auc_difference.mean(),

        "AUC_p":
            auc_p,

        "AUC_Wins":
            auc_wins,

        "AUC_Losses":
            auc_losses,

        "AUC_Ties":
            auc_ties,

        "RL_Mean_Features":
            paired[
                "Features_RL"
            ].mean(),

        "Other_Mean_Features":
            paired[
                "Features_OTHER"
            ].mean(),

        "Feature_Difference":
            feature_difference.mean(),

        "Feature_p":
            feature_p,

        "RL_Mean_Objective":
            paired[
                "Objective_RL"
            ].mean(),

        "Other_Mean_Objective":
            paired[
                "Objective_OTHER"
            ].mean(),

        "Objective_Difference":
            objective_difference.mean(),

        "Objective_p":
            objective_p,

        "Objective_Wins":
            objective_wins,

        "Objective_Losses":
            objective_losses,

        "Objective_Ties":
            objective_ties,
    })


# ============================================================
# 6. HOLM CORRECTION
# ============================================================

results_df = pd.DataFrame(
    results
)


_, corrected_auc, _, _ = (
    multipletests(
        results_df["AUC_p"],
        alpha=0.05,
        method="holm",
    )
)


_, corrected_features, _, _ = (
    multipletests(
        results_df["Feature_p"],
        alpha=0.05,
        method="holm",
    )
)


_, corrected_objective, _, _ = (
    multipletests(
        results_df["Objective_p"],
        alpha=0.05,
        method="holm",
    )
)


results_df[
    "AUC_Holm_p"
] = corrected_auc

results_df[
    "Feature_Holm_p"
] = corrected_features

results_df[
    "Objective_Holm_p"
] = corrected_objective


results_df[
    "AUC_Significant"
] = (
    results_df[
        "AUC_Holm_p"
    ] < 0.05
)

results_df[
    "Feature_Significant"
] = (
    results_df[
        "Feature_Holm_p"
    ] < 0.05
)

results_df[
    "Objective_Significant"
] = (
    results_df[
        "Objective_Holm_p"
    ] < 0.05
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 115
)

print(
    "FINAL HEART DISEASE HOLM-CORRECTED SUMMARY"
)

print("=" * 115)


print(
    results_df[
        [
            "Comparison",

            "AUC_Difference",
            "AUC_Holm_p",
            "AUC_Significant",

            "Feature_Difference",
            "Feature_Holm_p",
            "Feature_Significant",

            "Objective_Difference",
            "Objective_Holm_p",
            "Objective_Significant",
        ]
    ].to_string(
        index=False,
        formatters={
            "AUC_Difference":
                "{:+.6f}".format,

            "AUC_Holm_p":
                "{:.8f}".format,

            "Feature_Difference":
                "{:+.2f}".format,

            "Feature_Holm_p":
                "{:.8f}".format,

            "Objective_Difference":
                "{:+.6f}".format,

            "Objective_Holm_p":
                "{:.8f}".format,
        },
    )
)


# ============================================================
# 8. SAVE
# ============================================================

summary.to_csv(
    HEART_RESULTS_DIR
    / "25_heart_combined_summary.csv",
    index=False,
)

results_df.to_csv(
    HEART_RESULTS_DIR
    / "25_heart_final_statistical_results.csv",
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    HEART_RESULTS_DIR
    / "25_heart_combined_summary.csv"
)

print(
    HEART_RESULTS_DIR
    / "25_heart_final_statistical_results.csv"
)