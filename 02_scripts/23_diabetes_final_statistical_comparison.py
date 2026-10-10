import pandas as pd

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from project_paths import (
    DIABETES_RESULTS_DIR,
    STATISTICAL_RESULTS_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    DIABETES_RESULTS_DIR
    / "22_diabetes_repeated_final_results.csv"
)

REFERENCE_METHOD = "RL"

COMPARISON_METHODS = [
    "ALL_FEATURES",
    "L1",
    "RFE",
    "SelectKBest",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("=" * 95)
print("FINAL DIABETES PAIRED STATISTICAL COMPARISON")
print("=" * 95)

print(
    "\nNumber of seeds:",
    df["Seed"].nunique(),
)


# ============================================================
# RL DATA
# ============================================================

rl = (
    df[
        df["Method"] == REFERENCE_METHOD
    ]
    .sort_values("Seed")
    .copy()
)


# ============================================================
# RUN PAIRED TESTS
# ============================================================

results = []


for method in COMPARISON_METHODS:

    baseline = (
        df[
            df["Method"] == method
        ]
        .sort_values("Seed")
        .copy()
    )


    paired = pd.merge(
        rl,
        baseline,
        on="Seed",
        suffixes=("_RL", "_BASE"),
    )


    # ========================================================
    # AUC
    # ========================================================

    auc_difference = (
        paired["Test_AUC_RL"]
        - paired["Test_AUC_BASE"]
    )


    auc_stat, auc_p = wilcoxon(
        paired["Test_AUC_RL"],
        paired["Test_AUC_BASE"],
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
    # OBJECTIVE
    # ========================================================

    objective_difference = (
        paired["Objective_RL"]
        - paired["Objective_BASE"]
    )


    objective_stat, objective_p = wilcoxon(
        paired["Objective_RL"],
        paired["Objective_BASE"],
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
    # FEATURE COUNT
    # ========================================================

    feature_difference = (
        paired["Features_RL"]
        - paired["Features_BASE"]
    )


    feature_stat, feature_p = wilcoxon(
        paired["Features_RL"],
        paired["Features_BASE"],
        alternative="two-sided",
    )


    # ========================================================
    # PRINT DETAILS
    # ========================================================

    print(
        "\n" + "=" * 95
    )

    print(
        f"RL vs {method}"
    )

    print("=" * 95)


    print("\nROC-AUC")
    print("-" * 45)

    print(
        f"RL mean            : "
        f"{paired['Test_AUC_RL'].mean():.4f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 10-len(method))}: "
        f"{paired['Test_AUC_BASE'].mean():.4f}"
    )

    print(
        f"Mean difference    : "
        f"{auc_difference.mean():+.6f}"
    )

    print(
        f"Median difference  : "
        f"{auc_difference.median():+.6f}"
    )

    print(
        f"RL wins/losses/ties: "
        f"{auc_wins}/{auc_losses}/{auc_ties}"
    )

    print(
        f"Wilcoxon p-value   : "
        f"{auc_p:.6f}"
    )


    print("\nFEATURE COUNT")
    print("-" * 45)

    print(
        f"RL mean            : "
        f"{paired['Features_RL'].mean():.2f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 10-len(method))}: "
        f"{paired['Features_BASE'].mean():.2f}"
    )

    print(
        f"Mean difference    : "
        f"{feature_difference.mean():+.2f}"
    )

    print(
        f"Wilcoxon p-value   : "
        f"{feature_p:.6f}"
    )


    print("\nOBJECTIVE")
    print("-" * 45)

    print(
        f"RL mean            : "
        f"{paired['Objective_RL'].mean():.4f}"
    )

    print(
        f"{method} mean"
        f"{' ' * max(1, 10-len(method))}: "
        f"{paired['Objective_BASE'].mean():.4f}"
    )

    print(
        f"Mean difference    : "
        f"{objective_difference.mean():+.6f}"
    )

    print(
        f"RL wins/losses/ties: "
        f"{objective_wins}/"
        f"{objective_losses}/"
        f"{objective_ties}"
    )

    print(
        f"Wilcoxon p-value   : "
        f"{objective_p:.6f}"
    )


    results.append({
        "Comparison":
            f"RL vs {method}",

        "RL_Mean_AUC":
            paired["Test_AUC_RL"].mean(),

        "Other_Mean_AUC":
            paired["Test_AUC_BASE"].mean(),

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
            paired["Features_RL"].mean(),

        "Other_Mean_Features":
            paired["Features_BASE"].mean(),

        "Feature_Difference":
            feature_difference.mean(),

        "Feature_p":
            feature_p,

        "RL_Mean_Objective":
            paired["Objective_RL"].mean(),

        "Other_Mean_Objective":
            paired["Objective_BASE"].mean(),

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
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# HOLM CORRECTION
# ============================================================

_, auc_corrected, _, _ = multipletests(
    results_df["AUC_p"],
    alpha=0.05,
    method="holm",
)

_, feature_corrected, _, _ = multipletests(
    results_df["Feature_p"],
    alpha=0.05,
    method="holm",
)

_, objective_corrected, _, _ = multipletests(
    results_df["Objective_p"],
    alpha=0.05,
    method="holm",
)


results_df[
    "AUC_Holm_p"
] = auc_corrected

results_df[
    "Feature_Holm_p"
] = feature_corrected

results_df[
    "Objective_Holm_p"
] = objective_corrected


results_df[
    "AUC_Significant"
] = (
    results_df["AUC_Holm_p"]
    < 0.05
)

results_df[
    "Feature_Significant"
] = (
    results_df["Feature_Holm_p"]
    < 0.05
)

results_df[
    "Objective_Significant"
] = (
    results_df["Objective_Holm_p"]
    < 0.05
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "FINAL HOLM-CORRECTED SUMMARY"
)

print("=" * 110)


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
                "{:.6f}".format,

            "Feature_Difference":
                "{:+.2f}".format,

            "Feature_Holm_p":
                "{:.6f}".format,

            "Objective_Difference":
                "{:+.6f}".format,

            "Objective_Holm_p":
                "{:.6f}".format,
        },
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    STATISTICAL_RESULTS_DIR
    / "23_diabetes_final_statistical_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)