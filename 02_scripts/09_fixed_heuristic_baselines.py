import zipfile
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project_paths import (
    DIABETES_ZIP,
    DIABETES_RESULTS_DIR,
)

# ============================================================
# SETTINGS
# ============================================================

ZIP_FILE = DIABETES_ZIP
CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

SEEDS = [1, 7, 21, 42, 100]

EPISODES = 10
STEPS_PER_EPISODE = 15

LAMBDA_PENALTY = 0.02


# ============================================================
# 1. LOAD DATA
# ============================================================

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(CSV_FILE) as f:
        df = pd.read_csv(f)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"].astype(int)

FEATURES = X.columns.tolist()
N_FEATURES = len(FEATURES)


# ============================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval
)


# ============================================================
# 3. DEVELOPMENT SAMPLE
# ============================================================

X_search, _, y_search, _ = train_test_split(
    X_train,
    y_train,
    train_size=25000,
    random_state=123,
    stratify=y_train
)


# ============================================================
# 4. SCALE SEARCH DATA
# ============================================================

search_scaler = StandardScaler()

X_search_scaled = search_scaler.fit_transform(
    X_search
)

X_val_scaled = search_scaler.transform(
    X_val
)


# ============================================================
# 5. SUBSET HELPERS
# ============================================================

def random_subset(min_features=5, max_features=18):

    n_selected = random.randint(
        min_features,
        max_features
    )

    indices = random.sample(
        range(N_FEATURES),
        n_selected
    )

    subset = [0] * N_FEATURES

    for i in indices:
        subset[i] = 1

    return subset


def selected_indices(subset):

    return [
        i for i, value in enumerate(subset)
        if value == 1
    ]


def selected_names(subset):

    return [
        FEATURES[i]
        for i in selected_indices(subset)
    ]


# ============================================================
# 6. LOW-LEVEL HEURISTICS
# ============================================================

def add_feature(subset):

    new_subset = subset.copy()

    available = [
        i for i, value in enumerate(new_subset)
        if value == 0
    ]

    if available:
        new_subset[random.choice(available)] = 1

    return new_subset


def remove_feature(subset):

    new_subset = subset.copy()

    selected = selected_indices(new_subset)

    if len(selected) > 1:
        new_subset[random.choice(selected)] = 0

    return new_subset


def swap_feature(subset):

    new_subset = subset.copy()

    selected = selected_indices(new_subset)

    unselected = [
        i for i, value in enumerate(new_subset)
        if value == 0
    ]

    if selected and unselected:

        remove_index = random.choice(selected)
        add_index = random.choice(unselected)

        new_subset[remove_index] = 0
        new_subset[add_index] = 1

    return new_subset


def flip_feature(subset):

    new_subset = subset.copy()

    index = random.randrange(N_FEATURES)

    if (
        new_subset[index] == 1
        and sum(new_subset) == 1
    ):
        return new_subset

    new_subset[index] = 1 - new_subset[index]

    return new_subset


HEURISTICS = {
    "ADD": add_feature,
    "REMOVE": remove_feature,
    "SWAP": swap_feature,
    "FLIP": flip_feature
}


# ============================================================
# 7. SUBSET EVALUATION
# ============================================================

evaluation_cache = {}


def evaluate_subset(subset):

    key = tuple(subset)

    if key in evaluation_cache:
        return evaluation_cache[key]

    indices = selected_indices(subset)

    if len(indices) == 0:
        return 0.5, 0.5

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_search_scaled[:, indices],
        y_search
    )

    val_probability = model.predict_proba(
        X_val_scaled[:, indices]
    )[:, 1]

    auc = roc_auc_score(
        y_val,
        val_probability
    )

    objective = (
        auc
        - LAMBDA_PENALTY
        * (len(indices) / N_FEATURES)
    )

    evaluation_cache[key] = (
        auc,
        objective
    )

    return auc, objective


# ============================================================
# 8. FINAL HELD-OUT TEST EVALUATION
# ============================================================

def evaluate_on_test(feature_names):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_trainval[feature_names]
    )

    X_test_scaled = scaler.transform(
        X_test[feature_names]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_trainval
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    return roc_auc_score(
        y_test,
        probabilities
    )


# ============================================================
# 9. ALL-FEATURE BASELINE
# ============================================================

all_feature_auc = evaluate_on_test(
    FEATURES
)

print("=" * 90)
print("FIXED HEURISTIC BASELINE EXPERIMENT")
print("=" * 90)

print(
    f"\nAll 21 features Test ROC-AUC: "
    f"{all_feature_auc:.4f}"
)


# ============================================================
# 10. FIXED-HEURISTIC EXPERIMENTS
# ============================================================

all_results = []

for heuristic_name, heuristic_function in HEURISTICS.items():

    print("\n" + "=" * 90)
    print(f"FIXED HEURISTIC: {heuristic_name}")
    print("=" * 90)

    for seed in SEEDS:

        random.seed(seed)
        np.random.seed(seed)

        best_subset = None
        best_validation_auc = -1
        best_objective = -1

        # --------------------------------------------
        # Same search budget:
        # 10 episodes x 15 steps
        # --------------------------------------------

        for episode in range(EPISODES):

            current_subset = random_subset()

            current_auc, current_objective = evaluate_subset(
                current_subset
            )

            if current_objective > best_objective:

                best_subset = current_subset.copy()
                best_validation_auc = current_auc
                best_objective = current_objective

            for step in range(STEPS_PER_EPISODE):

                # Always use the SAME heuristic
                new_subset = heuristic_function(
                    current_subset
                )

                new_auc, new_objective = evaluate_subset(
                    new_subset
                )

                current_subset = new_subset
                current_auc = new_auc
                current_objective = new_objective

                if current_objective > best_objective:

                    best_subset = current_subset.copy()
                    best_validation_auc = current_auc
                    best_objective = current_objective

        # --------------------------------------------
        # Final test evaluation
        # --------------------------------------------

        best_features = selected_names(
            best_subset
        )

        test_auc = evaluate_on_test(
            best_features
        )

        reduction = (
            1
            - len(best_features) / N_FEATURES
        ) * 100

        result = {
            "Heuristic": heuristic_name,
            "Seed": seed,
            "Features": len(best_features),
            "Validation_AUC": best_validation_auc,
            "Test_AUC": test_auc,
            "Reduction": reduction
        }

        all_results.append(result)

        print(
            f"Seed {seed:3d} | "
            f"Features: {len(best_features):2d} | "
            f"Validation AUC: {best_validation_auc:.4f} | "
            f"Test AUC: {test_auc:.4f} | "
            f"Reduction: {reduction:.1f}%"
        )


# ============================================================
# 11. FULL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print("\n" + "=" * 90)
print("FULL FIXED-HEURISTIC RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Validation_AUC": "{:.4f}".format,
            "Test_AUC": "{:.4f}".format,
            "Reduction": "{:.1f}".format
        }
    )
)


# ============================================================
# 12. SUMMARY BY HEURISTIC
# ============================================================

summary = (
    results_df
    .groupby("Heuristic")
    .agg(
        Mean_Test_AUC=("Test_AUC", "mean"),
        Std_Test_AUC=("Test_AUC", "std"),
        Mean_Features=("Features", "mean"),
        Mean_Reduction=("Reduction", "mean")
    )
    .reset_index()
)


print("\n" + "=" * 90)
print("SUMMARY BY FIXED HEURISTIC")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "Mean_Test_AUC": "{:.4f}".format,
            "Std_Test_AUC": "{:.4f}".format,
            "Mean_Features": "{:.2f}".format,
            "Mean_Reduction": "{:.1f}".format
        }
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

RESULTS_FILE = (
    DIABETES_RESULTS_DIR
    / "09_fixed_heuristic_results.csv"
)

SUMMARY_FILE = (
    DIABETES_RESULTS_DIR
    / "09_fixed_heuristic_summary.csv"
)

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

print("\nResults saved to:")
print(" - 09_fixed_heuristic_results.csv")
print(" - 09_fixed_heuristic_summary.csv")