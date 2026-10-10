import subprocess
import re
import os
import pandas as pd

from project_paths import (
    SCRIPTS_DIR,
    DIABETES_RESULTS_DIR,
)


SEEDS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
    21, 22, 23, 24, 25,
    26, 27, 28, 29, 30
]


# ============================================================
# PATHS
# ============================================================

SOURCE_SCRIPT = (
    SCRIPTS_DIR
    / "06_qlearning_hyperheuristic.py"
)

OUTPUT_FILE = (
    DIABETES_RESULTS_DIR
    / "10_rl_30_seed_results.csv"
)


print("=" * 80)
print("REPEATED RL RUNS")
print("=" * 80)


results = []


for seed in SEEDS:

    print(f"\nRunning seed {seed}...")

    # --------------------------------------------------------
    # 1. Read the existing RL script
    # --------------------------------------------------------

    with open(
        SOURCE_SCRIPT,
        "r",
        encoding="utf-8"
    ) as f:
        code = f.read()


    # --------------------------------------------------------
    # 2. Replace RANDOM_SEED with the current seed
    # --------------------------------------------------------

    modified_code = re.sub(
        r"RANDOM_SEED\s*=\s*\d+",
        f"RANDOM_SEED = {seed}",
        code
    )


    # --------------------------------------------------------
    # 3. Save a temporary RL script
    # --------------------------------------------------------

    temp_file = (
        SCRIPTS_DIR
        / f"temp_rl_seed_{seed}.py"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(modified_code)


    # --------------------------------------------------------
    # 4. Run the temporary script
    # --------------------------------------------------------

    completed = subprocess.run(
        [
            "python",
            str(temp_file)
        ],
        capture_output=True,
        text=True
    )

    output = completed.stdout


    # --------------------------------------------------------
    # 5. Remove temporary file
    # --------------------------------------------------------

    if os.path.exists(temp_file):
        os.remove(temp_file)


    # --------------------------------------------------------
    # 6. Extract final results
    # --------------------------------------------------------

    feature_match = re.search(
        r"RL number of features\s*:\s*(\d+)",
        output
    )

    auc_match = re.search(
        r"RL subset ROC-AUC\s*:\s*([0-9.]+)",
        output
    )

    reduction_match = re.search(
        r"Feature reduction\s*:\s*([0-9.]+)%",
        output
    )


    # --------------------------------------------------------
    # 7. Save results
    # --------------------------------------------------------

    if (
        feature_match
        and auc_match
        and reduction_match
    ):

        n_features = int(
            feature_match.group(1)
        )

        auc = float(
            auc_match.group(1)
        )

        reduction = float(
            reduction_match.group(1)
        )

        results.append(
            (
                seed,
                n_features,
                auc,
                reduction
            )
        )

        print(
            f"Completed seed {seed}: "
            f"{n_features} features, "
            f"ROC-AUC = {auc:.4f}, "
            f"reduction = {reduction:.1f}%"
        )

    else:

        print(
            f"Could not extract result "
            f"for seed {seed}."
        )

        if completed.stderr:

            print("Error:")
            print(completed.stderr)


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)


print(
    f"{'Seed':<10}"
    f"{'Features':<12}"
    f"{'ROC-AUC':<12}"
    f"{'Reduction':<12}"
)

print("-" * 46)


for (
    seed,
    n_features,
    auc,
    reduction
) in results:

    print(
        f"{seed:<10}"
        f"{n_features:<12}"
        f"{auc:<12.4f}"
        f"{reduction:<11.1f}%"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results,
    columns=[
        "Seed",
        "Features",
        "ROC_AUC",
        "Reduction"
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nResults saved to:")
print(OUTPUT_FILE)


# ============================================================
# OVERALL STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)


if len(df) > 0:

    print(
        f"Successful runs : "
        f"{len(df)} / {len(SEEDS)}"
    )

    print(
        f"Mean ROC-AUC    : "
        f"{df['ROC_AUC'].mean():.4f}"
    )

    print(
        f"Std ROC-AUC     : "
        f"{df['ROC_AUC'].std():.4f}"
    )

    print(
        f"Min ROC-AUC     : "
        f"{df['ROC_AUC'].min():.4f}"
    )

    print(
        f"Max ROC-AUC     : "
        f"{df['ROC_AUC'].max():.4f}"
    )

    print(
        f"Mean features   : "
        f"{df['Features'].mean():.2f}"
    )

    print(
        f"Min features    : "
        f"{df['Features'].min()}"
    )

    print(
        f"Max features    : "
        f"{df['Features'].max()}"
    )

    print(
        f"Mean reduction  : "
        f"{df['Reduction'].mean():.1f}%"
    )

    print(
        f"Std reduction   : "
        f"{df['Reduction'].std():.1f}%"
    )

else:

    print(
        "No successful RL runs were recorded."
    )