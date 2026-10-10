import pandas as pd

from statsmodels.stats.multitest import multipletests

from project_paths import STATISTICAL_RESULTS_DIR


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    STATISTICAL_RESULTS_DIR
    / "13_paired_statistical_results.csv"
)

OUTPUT_FILE = (
    STATISTICAL_RESULTS_DIR
    / "14_holm_corrected_results.csv"
)


# ============================================================
# 1. LOAD PAIRED TEST RESULTS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("=" * 90)
print("HOLM MULTIPLE-COMPARISON CORRECTION")
print("=" * 90)


# ============================================================
# 2. CORRECT ROC-AUC P-VALUES
# ============================================================

auc_reject, auc_corrected, _, _ = multipletests(
    df["AUC_p_value"],
    alpha=0.05,
    method="holm"
)

df["AUC_Holm_p"] = auc_corrected
df["AUC_Holm_Significant"] = auc_reject


# ============================================================
# 3. CORRECT FEATURE-COUNT P-VALUES
# ============================================================

feature_reject, feature_corrected, _, _ = multipletests(
    df["Feature_p_value"],
    alpha=0.05,
    method="holm"
)

df["Feature_Holm_p"] = feature_corrected
df["Feature_Holm_Significant"] = feature_reject


# ============================================================
# 4. CORRECT OBJECTIVE P-VALUES
# ============================================================

objective_reject, objective_corrected, _, _ = multipletests(
    df["Objective_p_value"],
    alpha=0.05,
    method="holm"
)

df["Objective_Holm_p"] = objective_corrected
df["Objective_Holm_Significant"] = objective_reject


# ============================================================
# 5. PRINT RESULTS
# ============================================================

for _, row in df.iterrows():

    print("\n" + "-" * 90)
    print(row["Comparison"])
    print("-" * 90)

    print(
        f"ROC-AUC:"
        f"\n  Original p = {row['AUC_p_value']:.6f}"
        f"\n  Holm p     = {row['AUC_Holm_p']:.6f}"
        f"\n  Significant after correction = "
        f"{row['AUC_Holm_Significant']}"
    )

    print(
        f"\nFeatures:"
        f"\n  Original p = {row['Feature_p_value']:.6f}"
        f"\n  Holm p     = {row['Feature_Holm_p']:.6f}"
        f"\n  Significant after correction = "
        f"{row['Feature_Holm_Significant']}"
    )

    print(
        f"\nObjective:"
        f"\n  Original p = {row['Objective_p_value']:.6f}"
        f"\n  Holm p     = {row['Objective_Holm_p']:.6f}"
        f"\n  Significant after correction = "
        f"{row['Objective_Holm_Significant']}"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print(
    df[
        [
            "Comparison",
            "AUC_Holm_p",
            "AUC_Holm_Significant",
            "Feature_Holm_p",
            "Feature_Holm_Significant",
            "Objective_Holm_p",
            "Objective_Holm_Significant"
        ]
    ].to_string(
        index=False,
        formatters={
            "AUC_Holm_p":
                "{:.6f}".format,
            "Feature_Holm_p":
                "{:.6f}".format,
            "Objective_Holm_p":
                "{:.6f}".format
        }
    )
)


print("\nResults saved to:")
print(OUTPUT_FILE)