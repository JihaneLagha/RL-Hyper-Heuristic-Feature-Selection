import pandas as pd
import numpy as np

from scipy.stats import wilcoxon

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

OUTPUT_FILE = (
    STATISTICAL_RESULTS_DIR
    / "13_paired_statistical_results.csv"
)

TOTAL_FEATURES = 21
LAMBDA_PENALTY = 0.02

ALPHA = 0.05


# ============================================================
# 1. LOAD DATA
# ============================================================

rl = pd.read_csv(RL_FILE)
baseline = pd.read_csv(BASELINE_FILE)


# ============================================================
# 2. PREPARE RL DATA
# ============================================================

rl["Test_AUC"] = rl["ROC_AUC"]

rl["Test_Objective"] = (
    rl["Test_AUC"]
    - LAMBDA_PENALTY
    * (rl["Features"] / TOTAL_FEATURES)
)


# ============================================================
# 3. METHODS TO COMPARE
# ============================================================

METHODS = [
    "RANDOM",
    "ADD",
    "REMOVE",
    "SWAP",
    "FLIP"
]


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def significance_label(p_value):

    if p_value < 0.001:
        return "Highly significant"

    elif p_value < 0.01:
        return "Very significant"

    elif p_value < 0.05:
        return "Significant"

    else:
        return "Not significant"


# ============================================================
# 5. PAIRED WILCOXON TESTS
# ============================================================

results = []


for method in METHODS:

    method_df = (
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

    # --------------------------------------------------------
    # Ensure exact seed pairing
    # --------------------------------------------------------

    paired = pd.merge(
        rl_sorted,
        method_df,
        on="Seed",
        suffixes=("_RL", "_BASE")
    )

    print("\n" + "=" * 90)
    print(f"RL vs {method}")
    print("=" * 90)

    print(
        "Paired observations:",
        len(paired)
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    auc_difference = (
        paired["Test_AUC_RL"]
        - paired["Test_AUC_BASE"]
    )

    auc_stat, auc_p = wilcoxon(
        paired["Test_AUC_RL"],
        paired["Test_AUC_BASE"],
        alternative="two-sided",
        zero_method="wilcox"
    )


    # ========================================================
    # NUMBER OF FEATURES
    # ========================================================

    feature_difference = (
        paired["Features_RL"]
        - paired["Features_BASE"]
    )

    feature_stat, feature_p = wilcoxon(
        paired["Features_RL"],
        paired["Features_BASE"],
        alternative="two-sided",
        zero_method="wilcox"
    )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    objective_difference = (
        paired["Test_Objective_RL"]
        - paired["Test_Objective_BASE"]
    )

    objective_stat, objective_p = wilcoxon(
        paired["Test_Objective_RL"],
        paired["Test_Objective_BASE"],
        alternative="two-sided",
        zero_method="wilcox"
    )


    # ========================================================
    # WIN / LOSS / TIE COUNTS
    # ========================================================

    auc_wins = int(
        (auc_difference > 0).sum()
    )

    auc_losses = int(
        (auc_difference < 0).sum()
    )

    auc_ties = int(
        (auc_difference == 0).sum()
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
    # PRINT RESULT
    # ========================================================

    print("\nROC-AUC")
    print("-" * 40)

    print(
        f"RL mean          : "
        f"{paired['Test_AUC_RL'].mean():.4f}"
    )

    print(
        f"{method} mean      : "
        f"{paired['Test_AUC_BASE'].mean():.4f}"
    )

    print(
        f"Mean difference  : "
        f"{auc_difference.mean():+.6f}"
    )

    print(
        f"Median difference: "
        f"{auc_difference.median():+.6f}"
    )

    print(
        f"RL wins/losses/ties: "
        f"{auc_wins}/{auc_losses}/{auc_ties}"
    )

    print(
        f"Wilcoxon p-value : "
        f"{auc_p:.6f}"
    )

    print(
        "Result           :",
        significance_label(auc_p)
    )


    print("\nFEATURE COUNT")
    print("-" * 40)

    print(
        f"RL mean          : "
        f"{paired['Features_RL'].mean():.2f}"
    )

    print(
        f"{method} mean      : "
        f"{paired['Features_BASE'].mean():.2f}"
    )

    print(
        f"Mean difference  : "
        f"{feature_difference.mean():+.2f}"
    )

    print(
        f"Wilcoxon p-value : "
        f"{feature_p:.6f}"
    )

    print(
        "Result           :",
        significance_label(feature_p)
    )


    print("\nCOMBINED OBJECTIVE")
    print("-" * 40)

    print(
        f"RL mean          : "
        f"{paired['Test_Objective_RL'].mean():.4f}"
    )

    print(
        f"{method} mean      : "
        f"{paired['Test_Objective_BASE'].mean():.4f}"
    )

    print(
        f"Mean difference  : "
        f"{objective_difference.mean():+.6f}"
    )

    print(
        f"RL wins/losses/ties: "
        f"{objective_wins}/"
        f"{objective_losses}/"
        f"{objective_ties}"
    )

    print(
        f"Wilcoxon p-value : "
        f"{objective_p:.6f}"
    )

    print(
        "Result           :",
        significance_label(objective_p)
    )


    results.append({
        "Comparison":
            f"RL vs {method}",

        "RL_Mean_AUC":
            paired["Test_AUC_RL"].mean(),

        "Baseline_Mean_AUC":
            paired["Test_AUC_BASE"].mean(),

        "AUC_Mean_Difference":
            auc_difference.mean(),

        "AUC_p_value":
            auc_p,

        "AUC_Result":
            significance_label(auc_p),

        "RL_Mean_Features":
            paired["Features_RL"].mean(),

        "Baseline_Mean_Features":
            paired["Features_BASE"].mean(),

        "Feature_p_value":
            feature_p,

        "Feature_Result":
            significance_label(feature_p),

        "RL_Mean_Objective":
            paired["Test_Objective_RL"].mean(),

        "Baseline_Mean_Objective":
            paired["Test_Objective_BASE"].mean(),

        "Objective_Mean_Difference":
            objective_difference.mean(),

        "Objective_p_value":
            objective_p,

        "Objective_Result":
            significance_label(objective_p),

        "AUC_RL_Wins":
            auc_wins,

        "AUC_RL_Losses":
            auc_losses,

        "AUC_Ties":
            auc_ties,

        "Objective_RL_Wins":
            objective_wins,

        "Objective_RL_Losses":
            objective_losses,

        "Objective_Ties":
            objective_ties
    })


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    results_df[
        [
            "Comparison",
            "AUC_Mean_Difference",
            "AUC_p_value",
            "AUC_Result",
            "Objective_Mean_Difference",
            "Objective_p_value",
            "Objective_Result"
        ]
    ].to_string(
        index=False,
        formatters={
            "AUC_Mean_Difference":
                "{:+.6f}".format,

            "AUC_p_value":
                "{:.6f}".format,

            "Objective_Mean_Difference":
                "{:+.6f}".format,

            "Objective_p_value":
                "{:.6f}".format
        }
    )
)


print("\nResults saved to:")
print(OUTPUT_FILE)