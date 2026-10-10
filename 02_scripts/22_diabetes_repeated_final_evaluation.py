import zipfile
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    RFE,
    SelectFromModel
)
from sklearn.metrics import roc_auc_score

from project_paths import (
    DIABETES_ZIP,
    DIABETES_RESULTS_DIR,
)


# ============================================================
# SETTINGS
# ============================================================

ZIP_FILE = DIABETES_ZIP

DATA_FILE = (
    "diabetes_binary_health_indicators_BRFSS2015.csv"
)

TARGET = "Diabetes_binary"

SEEDS = list(range(1, 11))

SEARCH_SAMPLE_SIZE = 25000

LAMBDA_PENALTY = 0.02

RL_EPISODES = 10
RL_STEPS = 15


# ============================================================
# LOAD DATA
# ============================================================

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(DATA_FILE) as f:
        df = pd.read_csv(f)


X = df.drop(columns=[TARGET])
y = df[TARGET]

FEATURE_NAMES = X.columns.tolist()

N_FEATURES = len(FEATURE_NAMES)


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
        FEATURE_NAMES[i]
        for i in selected_indices(subset)
    ]


def random_subset():

    n = random.randint(
        5,
        18
    )

    indices = random.sample(
        range(N_FEATURES),
        n
    )

    subset = [0] * N_FEATURES

    for i in indices:
        subset[i] = 1

    return subset


# ============================================================
# LOW-LEVEL HEURISTICS
# ============================================================

def add_feature(subset):

    new = subset.copy()

    available = [
        i
        for i, value in enumerate(new)
        if value == 0
    ]

    if available:
        new[random.choice(available)] = 1

    return new


def remove_feature(subset):

    new = subset.copy()

    selected = selected_indices(new)

    if len(selected) > 1:
        new[random.choice(selected)] = 0

    return new


def swap_feature(subset):

    new = subset.copy()

    selected = selected_indices(new)

    unselected = [
        i
        for i, value in enumerate(new)
        if value == 0
    ]

    if selected and unselected:

        old = random.choice(selected)
        new_feature = random.choice(unselected)

        new[old] = 0
        new[new_feature] = 1

    return new


def flip_feature(subset):

    new = subset.copy()

    index = random.randrange(
        N_FEATURES
    )

    if (
        new[index] == 1
        and sum(new) == 1
    ):
        return new

    new[index] = 1 - new[index]

    return new


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

    if n <= 7:
        return 0

    elif n <= 14:
        return 1

    else:
        return 2


# ============================================================
# FINAL EVALUATION FUNCTION
# ============================================================

def evaluate_final_subset(
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    features
):

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_trainval[features]
    )

    test_scaled = scaler.transform(
        X_test[features]
    )

    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        train_scaled,
        y_trainval
    )

    probability = model.predict_proba(
        test_scaled
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probability
    )

    objective = (
        auc
        - LAMBDA_PENALTY
        * (
            len(features)
            / N_FEATURES
        )
    )

    return auc, objective


# ============================================================
# ONE OUTER SPLIT
# ============================================================

def run_seed(seed):

    random.seed(seed)
    np.random.seed(seed)


    # --------------------------------------------------------
    # 80% development / 20% untouched outer test
    # --------------------------------------------------------

    (
        X_trainval,
        X_test,
        y_trainval,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )


    # --------------------------------------------------------
    # Development split:
    # 75% search train / 25% validation
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed,
        stratify=y_trainval
    )


    # --------------------------------------------------------
    # Use only training data for RL search sample
    # --------------------------------------------------------

    (
        X_search,
        _,
        y_search,
        _
    ) = train_test_split(
        X_train,
        y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        random_state=seed,
        stratify=y_train
    )


    search_scaler = StandardScaler()

    X_search_scaled = (
        search_scaler.fit_transform(
            X_search
        )
    )

    X_val_scaled = (
        search_scaler.transform(
            X_val
        )
    )


    # ========================================================
    # ALL FEATURES
    # ========================================================

    all_auc, all_objective = evaluate_final_subset(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        FEATURE_NAMES
    )


    # ========================================================
    # RL HYPER-HEURISTIC
    # ========================================================

    cache = {}


    def evaluate_rl_subset(subset):

        key = tuple(subset)

        if key in cache:
            return cache[key]

        indices = selected_indices(
            subset
        )

        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )

        model.fit(
            X_search_scaled[:, indices],
            y_search
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

        return auc, objective


    Q = np.zeros(
        (3, 4)
    )

    alpha = 0.20
    gamma = 0.80

    epsilon = 0.30

    best_subset = None
    best_objective = -1


    for episode in range(
        RL_EPISODES
    ):

        current = random_subset()

        (
            current_auc,
            current_objective
        ) = evaluate_rl_subset(
            current
        )


        if current_objective > best_objective:

            best_objective = current_objective
            best_subset = current.copy()


        for step in range(
            RL_STEPS
        ):

            state = get_state(
                current
            )


            if random.random() < epsilon:

                action = random.randrange(
                    4
                )

            else:

                action = int(
                    np.argmax(
                        Q[state]
                    )
                )


            candidate = ACTIONS[
                action
            ](
                current
            )


            (
                candidate_auc,
                candidate_objective
            ) = evaluate_rl_subset(
                candidate
            )


            reward = (
                candidate_objective
                - current_objective
            )


            next_state = get_state(
                candidate
            )


            Q[state, action] += (
                alpha
                * (
                    reward
                    + gamma
                    * np.max(
                        Q[next_state]
                    )
                    - Q[state, action]
                )
            )


            current = candidate

            current_auc = candidate_auc

            current_objective = (
                candidate_objective
            )


            if (
                current_objective
                > best_objective
            ):

                best_objective = (
                    current_objective
                )

                best_subset = (
                    current.copy()
                )


        epsilon = max(
            0.05,
            epsilon * 0.95
        )


    rl_features = selected_names(
        best_subset
    )


    rl_auc, rl_objective = evaluate_final_subset(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        rl_features
    )


    # ========================================================
    # SELECTKBEST
    # ========================================================

    best_k_objective = -1
    best_k_features = None


    for k in [
        5,
        8,
        10,
        12,
        15,
        18
    ]:

        selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

        selector.fit(
            X_train,
            y_train
        )

        indices = selector.get_support(
            indices=True
        )

        features = [
            FEATURE_NAMES[i]
            for i in indices
        ]


        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(
            X_train[features]
        )

        val_scaled = scaler.transform(
            X_val[features]
        )


        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )

        model.fit(
            train_scaled,
            y_train
        )

        prob = model.predict_proba(
            val_scaled
        )[:, 1]


        val_auc = roc_auc_score(
            y_val,
            prob
        )


        objective = (
            val_auc
            - LAMBDA_PENALTY
            * (
                k / N_FEATURES
            )
        )


        if objective > best_k_objective:

            best_k_objective = objective
            best_k_features = features


    skb_auc, skb_objective = evaluate_final_subset(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        best_k_features
    )


    # ========================================================
    # L1
    # ========================================================

    l1_scaler = StandardScaler()

    X_train_l1 = l1_scaler.fit_transform(
        X_train
    )


    l1_model = LogisticRegression(
        penalty="l1",
        C=0.05,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )


    l1_model.fit(
        X_train_l1,
        y_train
    )


    selector = SelectFromModel(
        l1_model,
        prefit=True,
        threshold="median"
    )


    indices = selector.get_support(
        indices=True
    )


    l1_features = [
        FEATURE_NAMES[i]
        for i in indices
    ]


    l1_auc, l1_objective = evaluate_final_subset(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        l1_features
    )


    # ========================================================
    # RFE
    # ========================================================

    X_train_rfe_scaler = StandardScaler()

    X_train_rfe = (
        X_train_rfe_scaler.fit_transform(
            X_train
        )
    )

    X_val_rfe = (
        X_train_rfe_scaler.transform(
            X_val
        )
    )


    best_rfe_objective = -1
    best_rfe_features = None


    for k in [
        5,
        8,
        10,
        12,
        15
    ]:

        estimator = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )


        rfe = RFE(
            estimator=estimator,
            n_features_to_select=k,
            step=1
        )


        rfe.fit(
            X_train_rfe,
            y_train
        )


        indices = rfe.get_support(
            indices=True
        )


        features = [
            FEATURE_NAMES[i]
            for i in indices
        ]


        selected_indices_array = np.array(
            indices
        )


        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )


        model.fit(
            X_train_rfe[
                :,
                selected_indices_array
            ],
            y_train
        )


        probability = model.predict_proba(
            X_val_rfe[
                :,
                selected_indices_array
            ]
        )[:, 1]


        val_auc = roc_auc_score(
            y_val,
            probability
        )


        objective = (
            val_auc
            - LAMBDA_PENALTY
            * (
                k / N_FEATURES
            )
        )


        if objective > best_rfe_objective:

            best_rfe_objective = objective
            best_rfe_features = features


    rfe_auc, rfe_objective = evaluate_final_subset(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        best_rfe_features
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return [
        {
            "Seed": seed,
            "Method": "ALL_FEATURES",
            "Features": N_FEATURES,
            "Test_AUC": all_auc,
            "Objective": all_objective
        },

        {
            "Seed": seed,
            "Method": "RL",
            "Features": len(rl_features),
            "Test_AUC": rl_auc,
            "Objective": rl_objective
        },

        {
            "Seed": seed,
            "Method": "SelectKBest",
            "Features": len(best_k_features),
            "Test_AUC": skb_auc,
            "Objective": skb_objective
        },

        {
            "Seed": seed,
            "Method": "L1",
            "Features": len(l1_features),
            "Test_AUC": l1_auc,
            "Objective": l1_objective
        },

        {
            "Seed": seed,
            "Method": "RFE",
            "Features": len(best_rfe_features),
            "Test_AUC": rfe_auc,
            "Objective": rfe_objective
        }
    ]


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []


print("=" * 95)
print("CLEAN REPEATED DIABETES FINAL EVALUATION")
print("=" * 95)


for seed in SEEDS:

    seed_results = run_seed(
        seed
    )

    all_results.extend(
        seed_results
    )


    print(
        f"\nSeed {seed}"
    )


    for result in seed_results:

        reduction = (
            1
            - result["Features"]
            / N_FEATURES
        ) * 100


        print(
            f"  {result['Method']:12s}"
            f" | Features: "
            f"{result['Features']:2d}"
            f" | AUC: "
            f"{result['Test_AUC']:.4f}"
            f" | Reduction: "
            f"{reduction:5.1f}%"
            f" | Objective: "
            f"{result['Objective']:.4f}"
        )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df["Reduction"] = (
    1
    - results_df["Features"]
    / N_FEATURES
) * 100


summary = (
    results_df
    .groupby("Method")
    .agg(
        Mean_AUC=(
            "Test_AUC",
            "mean"
        ),

        Std_AUC=(
            "Test_AUC",
            "std"
        ),

        Mean_Features=(
            "Features",
            "mean"
        ),

        Mean_Reduction=(
            "Reduction",
            "mean"
        ),

        Mean_Objective=(
            "Objective",
            "mean"
        ),

        Std_Objective=(
            "Objective",
            "std"
        )
    )
    .reset_index()
)


print(
    "\n" + "=" * 95
)

print(
    "FINAL 10-SPLIT SUMMARY"
)

print("=" * 95)


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
                "{:.4f}".format
        }
    )
)


# ============================================================
# SAVE
# ============================================================

RESULTS_FILE = (
    DIABETES_RESULTS_DIR
    / "22_diabetes_repeated_final_results.csv"
)

SUMMARY_FILE = (
    DIABETES_RESULTS_DIR
    / "22_diabetes_repeated_final_summary.csv"
)


results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_FILE
)

print(
    SUMMARY_FILE
)