import zipfile
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project_paths import (
    HEART_DISEASE_ZIP,
    HEART_RESULTS_DIR,
)

# ============================================================
# SETTINGS
# ============================================================

ZIP_FILE = HEART_DISEASE_ZIP
DATA_FILE = "processed.cleveland.data"

SEEDS = list(range(1, 31))

LAMBDA_PENALTY = 0.02

EPISODES = 20
STEPS_PER_EPISODE = 15


COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]


# ============================================================
# 1. LOAD DATA
# ============================================================

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(DATA_FILE) as f:

        df = pd.read_csv(
            f,
            header=None,
            names=COLUMNS,
            na_values="?"
        )


df["HeartDisease_binary"] = (
    df["num"] > 0
).astype(int)


X = df.drop(
    columns=[
        "num",
        "HeartDisease_binary"
    ]
)

y = df["HeartDisease_binary"]

FEATURES = X.columns.tolist()
N_FEATURES = len(FEATURES)


# ============================================================
# HELPERS
# ============================================================

def selected_indices(subset):

    return [
        i
        for i, value in enumerate(subset)
        if value == 1
    ]


def selected_names(subset):

    return [
        FEATURES[i]
        for i in selected_indices(subset)
    ]


def random_subset(
    min_features=3,
    max_features=11
):

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


# ============================================================
# LOW-LEVEL HEURISTICS
# ============================================================

def add_feature(subset):

    new_subset = subset.copy()

    available = [
        i
        for i, value in enumerate(new_subset)
        if value == 0
    ]

    if available:

        new_subset[
            random.choice(available)
        ] = 1

    return new_subset


def remove_feature(subset):

    new_subset = subset.copy()

    selected = selected_indices(
        new_subset
    )

    if len(selected) > 1:

        new_subset[
            random.choice(selected)
        ] = 0

    return new_subset


def swap_feature(subset):

    new_subset = subset.copy()

    selected = selected_indices(
        new_subset
    )

    unselected = [
        i
        for i, value in enumerate(new_subset)
        if value == 0
    ]

    if selected and unselected:

        remove_index = random.choice(
            selected
        )

        add_index = random.choice(
            unselected
        )

        new_subset[remove_index] = 0
        new_subset[add_index] = 1

    return new_subset


def flip_feature(subset):

    new_subset = subset.copy()

    index = random.randrange(
        N_FEATURES
    )

    if (
        new_subset[index] == 1
        and sum(new_subset) == 1
    ):
        return new_subset

    new_subset[index] = (
        1 - new_subset[index]
    )

    return new_subset


ACTIONS = [
    add_feature,
    remove_feature,
    swap_feature,
    flip_feature
]


# ============================================================
# STATE
# ============================================================

def get_state(subset):

    n = sum(subset)

    if n <= 4:
        return 0

    elif n <= 8:
        return 1

    else:
        return 2


# ============================================================
# RUN ONE COMPLETE EXPERIMENT
# ============================================================

def run_experiment(seed):

    random.seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------------
    # Different train/test split for every seed
    # --------------------------------------------------------

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed,
        stratify=y_trainval
    )


    # ========================================================
    # IMPUTATION
    # ========================================================

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_val_imp = imputer.transform(
        X_val
    )


    # ========================================================
    # SCALING
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imp
    )

    X_val_scaled = scaler.transform(
        X_val_imp
    )


    # ========================================================
    # SUBSET EVALUATION
    # ========================================================

    cache = {}

    def evaluate_subset(subset):

        key = tuple(subset)

        if key in cache:
            return cache[key]

        indices = selected_indices(
            subset
        )

        model = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=seed
        )

        model.fit(
            X_train_scaled[:, indices],
            y_train
        )

        probability = model.predict_proba(
            X_val_scaled[:, indices]
        )[:, 1]

        auc = roc_auc_score(
            y_val,
            probability
        )

        objective = (
            auc
            - LAMBDA_PENALTY
            * (
                len(indices)
                / N_FEATURES
            )
        )

        cache[key] = (
            auc,
            objective
        )

        return (
            auc,
            objective
        )


    # ========================================================
    # Q-LEARNING
    # ========================================================

    Q = np.zeros(
        (3, 4)
    )

    alpha = 0.20
    gamma = 0.80

    epsilon = 0.30
    epsilon_decay = 0.95
    minimum_epsilon = 0.05


    best_subset = None
    best_validation_auc = -1
    best_objective = -1


    for episode in range(
        EPISODES
    ):

        current_subset = (
            random_subset()
        )

        (
            current_auc,
            current_objective
        ) = evaluate_subset(
            current_subset
        )

        if (
            current_objective
            > best_objective
        ):

            best_subset = (
                current_subset.copy()
            )

            best_validation_auc = (
                current_auc
            )

            best_objective = (
                current_objective
            )


        for step in range(
            STEPS_PER_EPISODE
        ):

            state = get_state(
                current_subset
            )

            if random.random() < epsilon:

                action_index = (
                    random.randrange(4)
                )

            else:

                action_index = int(
                    np.argmax(
                        Q[state]
                    )
                )


            new_subset = ACTIONS[
                action_index
            ](
                current_subset
            )


            (
                new_auc,
                new_objective
            ) = evaluate_subset(
                new_subset
            )


            reward = (
                new_objective
                - current_objective
            )


            next_state = get_state(
                new_subset
            )


            Q[
                state,
                action_index
            ] += (
                alpha
                * (
                    reward
                    + gamma
                    * np.max(
                        Q[next_state]
                    )
                    - Q[
                        state,
                        action_index
                    ]
                )
            )


            current_subset = (
                new_subset
            )

            current_auc = (
                new_auc
            )

            current_objective = (
                new_objective
            )


            if (
                current_objective
                > best_objective
            ):

                best_subset = (
                    current_subset.copy()
                )

                best_validation_auc = (
                    current_auc
                )

                best_objective = (
                    current_objective
                )


        epsilon = max(
            minimum_epsilon,
            epsilon
            * epsilon_decay
        )


    # ========================================================
    # SELECTED FEATURES
    # ========================================================

    best_features = selected_names(
        best_subset
    )


    # ========================================================
    # FINAL RL TEST MODEL
    # ========================================================

    final_imputer = SimpleImputer(
        strategy="median"
    )

    X_trainval_best = (
        final_imputer.fit_transform(
            X_trainval[
                best_features
            ]
        )
    )

    X_test_best = (
        final_imputer.transform(
            X_test[
                best_features
            ]
        )
    )


    final_scaler = StandardScaler()

    X_trainval_best = (
        final_scaler.fit_transform(
            X_trainval_best
        )
    )

    X_test_best = (
        final_scaler.transform(
            X_test_best
        )
    )


    final_model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=seed
    )

    final_model.fit(
        X_trainval_best,
        y_trainval
    )


    rl_probability = (
        final_model.predict_proba(
            X_test_best
        )[:, 1]
    )

    rl_auc = roc_auc_score(
        y_test,
        rl_probability
    )


    # ========================================================
    # ALL-FEATURE BASELINE
    # ========================================================

    baseline_imputer = SimpleImputer(
        strategy="median"
    )

    X_trainval_all = (
        baseline_imputer.fit_transform(
            X_trainval
        )
    )

    X_test_all = (
        baseline_imputer.transform(
            X_test
        )
    )


    baseline_scaler = StandardScaler()

    X_trainval_all = (
        baseline_scaler.fit_transform(
            X_trainval_all
        )
    )

    X_test_all = (
        baseline_scaler.transform(
            X_test_all
        )
    )


    baseline_model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=seed
    )

    baseline_model.fit(
        X_trainval_all,
        y_trainval
    )


    baseline_probability = (
        baseline_model.predict_proba(
            X_test_all
        )[:, 1]
    )


    baseline_auc = roc_auc_score(
        y_test,
        baseline_probability
    )


    reduction = (
        1
        - len(best_features)
        / N_FEATURES
    ) * 100


    return {
        "Seed": seed,

        "RL_Features":
            len(best_features),

        "Validation_AUC":
            best_validation_auc,

        "RL_Test_AUC":
            rl_auc,

        "All_Feature_AUC":
            baseline_auc,

        "AUC_Difference":
            rl_auc
            - baseline_auc,

        "Reduction":
            reduction,

        "Selected_Features":
            ",".join(
                best_features
            )
    }


# ============================================================
# RUN 30 EXPERIMENTS
# ============================================================

print("=" * 90)
print("REPEATED HEART DISEASE RL VALIDATION")
print("=" * 90)


results = []


for seed in SEEDS:

    result = run_experiment(
        seed
    )

    results.append(
        result
    )

    print(
        f"Seed {seed:2d}"
        f" | Features: "
        f"{result['RL_Features']:2d}"
        f" | RL AUC: "
        f"{result['RL_Test_AUC']:.4f}"
        f" | All AUC: "
        f"{result['All_Feature_AUC']:.4f}"
        f" | Difference: "
        f"{result['AUC_Difference']:+.4f}"
        f" | Reduction: "
        f"{result['Reduction']:.1f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n" + "=" * 90
)

print(
    "HEART DISEASE 30-RUN SUMMARY"
)

print("=" * 90)


print(
    f"Mean RL ROC-AUC       : "
    f"{results_df['RL_Test_AUC'].mean():.4f}"
)

print(
    f"Std RL ROC-AUC        : "
    f"{results_df['RL_Test_AUC'].std():.4f}"
)

print(
    f"Mean all-feature AUC  : "
    f"{results_df['All_Feature_AUC'].mean():.4f}"
)

print(
    f"Std all-feature AUC   : "
    f"{results_df['All_Feature_AUC'].std():.4f}"
)

print(
    f"Mean AUC difference   : "
    f"{results_df['AUC_Difference'].mean():+.4f}"
)

print(
    f"Mean selected features: "
    f"{results_df['RL_Features'].mean():.2f}"
)

print(
    f"Mean feature reduction: "
    f"{results_df['Reduction'].mean():.1f}%"
)


wins = (
    results_df[
        "AUC_Difference"
    ] > 0
).sum()

losses = (
    results_df[
        "AUC_Difference"
    ] < 0
).sum()

ties = (
    results_df[
        "AUC_Difference"
    ] == 0
).sum()


print(
    f"RL wins/losses/ties   : "
    f"{wins}/{losses}/{ties}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    HEART_RESULTS_DIR
    / "19_heart_repeated_rl_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nResults saved to:")
print(
    OUTPUT_FILE
)