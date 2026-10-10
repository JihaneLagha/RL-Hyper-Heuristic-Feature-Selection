import pandas as pd
import numpy as np

from scipy.stats import wilcoxon

from project_paths import HEART_RESULTS_DIR

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    HEART_RESULTS_DIR
    / "19_heart_repeated_rl_results.csv"
)

TOTAL_FEATURES = 13

LAMBDA_PENALTY = 0.02


# ============================================================
# 1. LOAD RESULTS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("=" * 90)
print("HEART DISEASE STATISTICAL ANALYSIS")
print("=" * 90)

print(
    "Number of paired runs:",
    len(df)
)


# ============================================================
# 2. COMPUTE OBJECTIVES
# ============================================================

df["RL_Objective"] = (
    df["RL_Test_AUC"]
    - LAMBDA_PENALTY
    * (
        df["RL_Features"]
        / TOTAL_FEATURES
    )
)

df["All_Feature_Objective"] = (
    df["All_Feature_AUC"]
    - LAMBDA_PENALTY
)


df["Objective_Difference"] = (
    df["RL_Objective"]
    - df["All_Feature_Objective"]
)


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("DESCRIPTIVE RESULTS")
print("=" * 90)


print(
    f"Mean RL ROC-AUC          : "
    f"{df['RL_Test_AUC'].mean():.4f}"
)

print(
    f"Std RL ROC-AUC           : "
    f"{df['RL_Test_AUC'].std():.4f}"
)

print(
    f"Mean all-feature ROC-AUC : "
    f"{df['All_Feature_AUC'].mean():.4f}"
)

print(
    f"Std all-feature ROC-AUC  : "
    f"{df['All_Feature_AUC'].std():.4f}"
)

print(
    f"Mean AUC difference      : "
    f"{df['AUC_Difference'].mean():+.4f}"
)

print(
    f"Median AUC difference    : "
    f"{df['AUC_Difference'].median():+.4f}"
)

print(
    f"Mean selected features   : "
    f"{df['RL_Features'].mean():.2f}"
)

print(
    f"Mean feature reduction   : "
    f"{df['Reduction'].mean():.1f}%"
)


# ============================================================
# 4. WILCOXON TEST FOR ROC-AUC
# ============================================================
#
# H0:
# RL subset and all-feature model have the same paired
# performance distribution.
# ============================================================

auc_statistic, auc_p = wilcoxon(
    df["RL_Test_AUC"],
    df["All_Feature_AUC"],
    alternative="two-sided"
)


print("\n" + "=" * 90)
print("PAIRED ROC-AUC TEST")
print("=" * 90)

print(
    f"Wilcoxon statistic : "
    f"{auc_statistic:.4f}"
)

print(
    f"p-value            : "
    f"{auc_p:.8f}"
)


if auc_p < 0.05:

    print(
        "Result             : "
        "Statistically significant difference"
    )

else:

    print(
        "Result             : "
        "No statistically significant difference"
    )


# ============================================================
# 5. DIRECTION OF DIFFERENCE
# ============================================================

wins = (
    df["AUC_Difference"] > 0
).sum()

losses = (
    df["AUC_Difference"] < 0
).sum()

ties = (
    df["AUC_Difference"] == 0
).sum()


print(
    f"\nRL wins/losses/ties: "
    f"{wins}/{losses}/{ties}"
)


# ============================================================
# 6. PERFORMANCE-COMPACTNESS OBJECTIVE
# ============================================================

objective_statistic, objective_p = wilcoxon(
    df["RL_Objective"],
    df["All_Feature_Objective"],
    alternative="two-sided"
)


print("\n" + "=" * 90)
print("PERFORMANCE-COMPACTNESS OBJECTIVE")
print("=" * 90)


print(
    f"Mean RL objective          : "
    f"{df['RL_Objective'].mean():.4f}"
)

print(
    f"Mean all-feature objective : "
    f"{df['All_Feature_Objective'].mean():.4f}"
)

print(
    f"Mean objective difference  : "
    f"{df['Objective_Difference'].mean():+.4f}"
)

print(
    f"Wilcoxon p-value           : "
    f"{objective_p:.8f}"
)


if objective_p < 0.05:

    print(
        "Result                     : "
        "Statistically significant difference"
    )

else:

    print(
        "Result                     : "
        "No statistically significant difference"
    )


# ============================================================
# 7. FEATURE FREQUENCY
# ============================================================

feature_counts = {}


for feature_string in df[
    "Selected_Features"
].dropna():

    features = feature_string.split(",")

    for feature in features:

        feature = feature.strip()

        feature_counts[feature] = (
            feature_counts.get(
                feature,
                0
            ) + 1
        )


frequency_df = (
    pd.DataFrame(
        list(
            feature_counts.items()
        ),
        columns=[
            "Feature",
            "Selection_Count"
        ]
    )
    .sort_values(
        "Selection_Count",
        ascending=False
    )
)


frequency_df["Selection_Percentage"] = (
    frequency_df[
        "Selection_Count"
    ]
    / len(df)
    * 100
)


print("\n" + "=" * 90)
print("FEATURE SELECTION FREQUENCY")
print("=" * 90)


print(
    frequency_df.to_string(
        index=False,
        formatters={
            "Selection_Percentage":
                "{:.1f}%".format
        }
    )
)


# ============================================================
# 8. SAVE
# ============================================================

summary = pd.DataFrame([
    {
        "Mean_RL_AUC":
            df["RL_Test_AUC"].mean(),

        "Mean_All_Feature_AUC":
            df["All_Feature_AUC"].mean(),

        "Mean_AUC_Difference":
            df["AUC_Difference"].mean(),

        "AUC_Wilcoxon_p":
            auc_p,

        "Mean_RL_Features":
            df["RL_Features"].mean(),

        "Mean_Reduction":
            df["Reduction"].mean(),

        "Mean_RL_Objective":
            df["RL_Objective"].mean(),

        "Mean_All_Feature_Objective":
            df["All_Feature_Objective"].mean(),

        "Objective_Wilcoxon_p":
            objective_p,

        "RL_Wins":
            wins,

        "RL_Losses":
            losses,

        "Ties":
            ties
    }
])


SUMMARY_FILE = (
    HEART_RESULTS_DIR
    / "20_heart_statistical_summary.csv"
)

FREQUENCY_FILE = (
    HEART_RESULTS_DIR
    / "20_heart_feature_frequency.csv"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

frequency_df.to_csv(
    FREQUENCY_FILE,
    index=False,
)


print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    SUMMARY_FILE
)

print(
    FREQUENCY_FILE
)