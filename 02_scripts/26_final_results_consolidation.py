import pandas as pd
import matplotlib.pyplot as plt

from project_paths import (
    DIABETES_RESULTS_DIR,
    HEART_RESULTS_DIR,
    STATISTICAL_RESULTS_DIR,
    DIABETES_FIGURES_DIR,
    HEART_FIGURES_DIR,
    CROSS_DATASET_FIGURES_DIR,
)


# ============================================================
# FILES
# ============================================================

DIABETES_RESULTS = (
    DIABETES_RESULTS_DIR
    / "22_diabetes_repeated_final_results.csv"
)

DIABETES_STATS = (
    STATISTICAL_RESULTS_DIR
    / "23_diabetes_final_statistical_results.csv"
)

HEART_RESULTS = (
    HEART_RESULTS_DIR
    / "25_heart_combined_summary.csv"
)

HEART_STATS = (
    HEART_RESULTS_DIR
    / "25_heart_final_statistical_results.csv"
)

HEART_FREQUENCY = (
    HEART_RESULTS_DIR
    / "20_heart_feature_frequency.csv"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

diabetes = pd.read_csv(
    DIABETES_RESULTS
)

diabetes_stats = pd.read_csv(
    DIABETES_STATS
)

heart_summary = pd.read_csv(
    HEART_RESULTS
)

heart_stats = pd.read_csv(
    HEART_STATS
)

heart_frequency = pd.read_csv(
    HEART_FREQUENCY
)


# ============================================================
# 2. DIABETES SUMMARY
# ============================================================

diabetes["Reduction"] = (
    1
    - diabetes["Features"] / 21
) * 100


diabetes_summary = (
    diabetes
    .groupby("Method")
    .agg(
        Mean_AUC=("Test_AUC", "mean"),
        Std_AUC=("Test_AUC", "std"),
        Mean_Features=("Features", "mean"),
        Mean_Reduction=("Reduction", "mean"),
        Mean_Objective=("Objective", "mean"),
        Std_Objective=("Objective", "std"),
    )
    .reset_index()
)


# ============================================================
# 3. CROSS-DATASET SUMMARY
# ============================================================

diabetes_cross = (
    diabetes_summary[
        diabetes_summary[
            "Method"
        ].isin(
            [
                "ALL_FEATURES",
                "RL",
                "L1",
                "RFE",
                "SelectKBest",
            ]
        )
    ]
    .copy()
)

diabetes_cross[
    "Dataset"
] = "Diabetes"


heart_cross = heart_summary.copy()

heart_cross[
    "Dataset"
] = "Heart Disease"


cross_dataset = pd.concat(
    [
        diabetes_cross,
        heart_cross,
    ],
    ignore_index=True,
)


cross_dataset = cross_dataset[
    [
        "Dataset",
        "Method",
        "Mean_AUC",
        "Std_AUC",
        "Mean_Features",
        "Mean_Reduction",
        "Mean_Objective",
        "Std_Objective",
    ]
]


# ============================================================
# 4. PRINT TABLES
# ============================================================

print("=" * 100)
print("FINAL DIABETES SUMMARY")
print("=" * 100)

print(
    diabetes_summary.to_string(
        index=False,
        formatters={
            "Mean_AUC": "{:.4f}".format,
            "Std_AUC": "{:.4f}".format,
            "Mean_Features": "{:.2f}".format,
            "Mean_Reduction": "{:.1f}%".format,
            "Mean_Objective": "{:.4f}".format,
            "Std_Objective": "{:.4f}".format,
        },
    )
)


print("\n" + "=" * 100)
print("FINAL HEART DISEASE SUMMARY")
print("=" * 100)

print(
    heart_summary.to_string(
        index=False,
        formatters={
            "Mean_AUC": "{:.4f}".format,
            "Std_AUC": "{:.4f}".format,
            "Mean_Features": "{:.2f}".format,
            "Mean_Reduction": "{:.1f}%".format,
            "Mean_Objective": "{:.4f}".format,
            "Std_Objective": "{:.4f}".format,
        },
    )
)


print("\n" + "=" * 100)
print("CROSS-DATASET SUMMARY")
print("=" * 100)

print(
    cross_dataset.to_string(
        index=False,
        formatters={
            "Mean_AUC": "{:.4f}".format,
            "Std_AUC": "{:.4f}".format,
            "Mean_Features": "{:.2f}".format,
            "Mean_Reduction": "{:.1f}%".format,
            "Mean_Objective": "{:.4f}".format,
            "Std_Objective": "{:.4f}".format,
        },
    )
)


# ============================================================
# 5. DIABETES AUC FIGURE
# ============================================================

plot_df = (
    diabetes_summary
    .sort_values(
        "Mean_AUC",
        ascending=False,
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    plot_df["Method"],
    plot_df["Mean_AUC"],
    yerr=plot_df["Std_AUC"],
    capsize=5,
)

plt.ylabel(
    "Mean ROC-AUC"
)

plt.xlabel(
    "Feature-Selection Method"
)

plt.title(
    "Diabetes: Mean ROC-AUC Across 10 Repeated Splits"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    DIABETES_FIGURES_DIR
    / "26_diabetes_auc_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# 6. DIABETES FEATURE REDUCTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    diabetes_summary["Method"],
    diabetes_summary[
        "Mean_Reduction"
    ],
)

plt.ylabel(
    "Mean Feature Reduction (%)"
)

plt.xlabel(
    "Method"
)

plt.title(
    "Diabetes: Feature Reduction"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    DIABETES_FIGURES_DIR
    / "26_diabetes_feature_reduction.png",
    dpi=300,
)

plt.close()


# ============================================================
# 7. DIABETES OBJECTIVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    diabetes_summary["Method"],
    diabetes_summary[
        "Mean_Objective"
    ],
    yerr=diabetes_summary[
        "Std_Objective"
    ],
    capsize=5,
)

plt.ylabel(
    "Mean Performance-Compactness Objective"
)

plt.xlabel(
    "Method"
)

plt.title(
    "Diabetes: Performance-Compactness Objective"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    DIABETES_FIGURES_DIR
    / "26_diabetes_objective.png",
    dpi=300,
)

plt.close()


# ============================================================
# 8. HEART DISEASE AUC
# ============================================================

heart_plot = (
    heart_summary
    .sort_values(
        "Mean_AUC",
        ascending=False,
    )
)


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    heart_plot["Method"],
    heart_plot["Mean_AUC"],
    yerr=heart_plot["Std_AUC"],
    capsize=5,
)

plt.ylabel(
    "Mean ROC-AUC"
)

plt.xlabel(
    "Feature-Selection Method"
)

plt.title(
    "Heart Disease: Mean ROC-AUC Across 30 Splits"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    HEART_FIGURES_DIR
    / "26_heart_auc_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# 9. HEART FEATURE REDUCTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    heart_summary["Method"],
    heart_summary[
        "Mean_Reduction"
    ],
)

plt.ylabel(
    "Mean Feature Reduction (%)"
)

plt.xlabel(
    "Method"
)

plt.title(
    "Heart Disease: Feature Reduction"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    HEART_FIGURES_DIR
    / "26_heart_feature_reduction.png",
    dpi=300,
)

plt.close()


# ============================================================
# 10. HEART OBJECTIVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    heart_summary["Method"],
    heart_summary[
        "Mean_Objective"
    ],
    yerr=heart_summary[
        "Std_Objective"
    ],
    capsize=5,
)

plt.ylabel(
    "Mean Performance-Compactness Objective"
)

plt.xlabel(
    "Method"
)

plt.title(
    "Heart Disease: Performance-Compactness Objective"
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    HEART_FIGURES_DIR
    / "26_heart_objective.png",
    dpi=300,
)

plt.close()


# ============================================================
# 11. HEART FEATURE FREQUENCY
# ============================================================

freq_plot = heart_frequency.sort_values(
    "Selection_Percentage",
    ascending=True,
)


plt.figure(
    figsize=(9, 7)
)

plt.barh(
    freq_plot["Feature"],
    freq_plot[
        "Selection_Percentage"
    ],
)

plt.xlabel(
    "Selection Frequency (%)"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Heart Disease: RL Feature Selection Frequency"
)

plt.tight_layout()

plt.savefig(
    HEART_FIGURES_DIR
    / "26_heart_feature_frequency.png",
    dpi=300,
)

plt.close()


# ============================================================
# 12. CROSS-DATASET RL COMPARISON
# ============================================================

rl_cross = cross_dataset[
    cross_dataset["Method"] == "RL"
].copy()


plt.figure(
    figsize=(7, 6)
)

plt.bar(
    rl_cross["Dataset"],
    rl_cross["Mean_AUC"],
    yerr=rl_cross["Std_AUC"],
    capsize=5,
)

plt.ylabel(
    "Mean ROC-AUC"
)

plt.title(
    "RL Hyper-Heuristic Across the Two Datasets"
)

plt.tight_layout()

plt.savefig(
    CROSS_DATASET_FIGURES_DIR
    / "26_rl_cross_dataset_auc.png",
    dpi=300,
)

plt.close()


# ============================================================
# 13. SAVE TABLES
# ============================================================

diabetes_summary.to_csv(
    DIABETES_RESULTS_DIR
    / "26_final_diabetes_summary.csv",
    index=False,
)

heart_summary.to_csv(
    HEART_RESULTS_DIR
    / "26_final_heart_summary.csv",
    index=False,
)

cross_dataset.to_csv(
    STATISTICAL_RESULTS_DIR
    / "26_cross_dataset_summary.csv",
    index=False,
)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 100)
print("OUTPUT FILES CREATED")
print("=" * 100)

print(
    DIABETES_RESULTS_DIR
    / "26_final_diabetes_summary.csv"
)

print(
    HEART_RESULTS_DIR
    / "26_final_heart_summary.csv"
)

print(
    STATISTICAL_RESULTS_DIR
    / "26_cross_dataset_summary.csv"
)

print(
    DIABETES_FIGURES_DIR
    / "26_diabetes_auc_comparison.png"
)

print(
    DIABETES_FIGURES_DIR
    / "26_diabetes_feature_reduction.png"
)

print(
    DIABETES_FIGURES_DIR
    / "26_diabetes_objective.png"
)

print(
    HEART_FIGURES_DIR
    / "26_heart_auc_comparison.png"
)

print(
    HEART_FIGURES_DIR
    / "26_heart_feature_reduction.png"
)

print(
    HEART_FIGURES_DIR
    / "26_heart_objective.png"
)

print(
    HEART_FIGURES_DIR
    / "26_heart_feature_frequency.png"
)

print(
    CROSS_DATASET_FIGURES_DIR
    / "26_rl_cross_dataset_auc.png"
)