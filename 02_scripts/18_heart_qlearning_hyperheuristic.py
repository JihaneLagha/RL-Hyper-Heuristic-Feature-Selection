import zipfile
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project_paths import HEART_DISEASE_ZIP

# ============================================================
# SETTINGS
# ============================================================

ZIP_FILE = HEART_DISEASE_ZIP
DATA_FILE = "processed.cleveland.data"

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


# ============================================================
# 2. CREATE BINARY TARGET
# ============================================================

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
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# 60% train
# 20% validation
# 20% final test
#
# Test remains untouched during RL search.
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


print("=" * 80)
print("HEART DISEASE DATA SPLIT")
print("=" * 80)

print(
    "Training   :",
    X_train.shape
)

print(
    "Validation :",
    X_val.shape
)

print(
    "Test       :",
    X_test.shape
)


# ============================================================
# 4. IMPUTE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imp = imputer.fit_transform(
    X_train
)

X_val_imp = imputer.transform(
    X_val
)


# ============================================================
# 5. STANDARDIZE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imp
)

X_val_scaled = scaler.transform(
    X_val_imp
)


# ============================================================
# 6. FEATURE SUBSET REPRESENTATION
# ============================================================

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


# ============================================================
# 7. LOW-LEVEL HEURISTICS
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

ACTION_NAMES = [
    "ADD",
    "REMOVE",
    "SWAP",
    "FLIP"
]


# ============================================================
# 8. SUBSET EVALUATION
# ============================================================

evaluation_cache = {}


def evaluate_subset(subset):

    key = tuple(subset)

    if key in evaluation_cache:

        return evaluation_cache[key]

    indices = selected_indices(
        subset
    )

    if len(indices) == 0:

        return 0.5, 0.5

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train_scaled[:, indices],
        y_train
    )

    probabilities = model.predict_proba(
        X_val_scaled[:, indices]
    )[:, 1]

    auc = roc_auc_score(
        y_val,
        probabilities
    )

    # --------------------------------------------------------
    # Objective:
    # reward prediction performance,
    # mildly penalize number of features
    # --------------------------------------------------------

    lambda_penalty = 0.02

    objective = (
        auc
        - lambda_penalty
        * (
            len(indices)
            / N_FEATURES
        )
    )

    evaluation_cache[key] = (
        auc,
        objective
    )

    return (
        auc,
        objective
    )


# ============================================================
# 9. DEFINE STATES
# ============================================================
#
# SMALL  = 1-4 features
# MEDIUM = 5-8 features
# LARGE  = 9-13 features
# ============================================================

def get_state(subset):

    n = sum(subset)

    if n <= 4:
        return 0

    elif n <= 8:
        return 1

    else:
        return 2


STATE_NAMES = [
    "SMALL",
    "MEDIUM",
    "LARGE"
]


# ============================================================
# 10. Q-LEARNING SETTINGS
# ============================================================

N_STATES = 3
N_ACTIONS = 4

Q = np.zeros(
    (
        N_STATES,
        N_ACTIONS
    )
)

alpha = 0.20
gamma = 0.80

epsilon = 0.30
epsilon_decay = 0.95
minimum_epsilon = 0.05

episodes = 20
steps_per_episode = 15


# ============================================================
# 11. TRACK BEST SOLUTION
# ============================================================

best_subset = None
best_auc = -1
best_objective = -1


print(
    "\n" + "=" * 80
)

print(
    "STARTING HEART DISEASE "
    "Q-LEARNING HYPER-HEURISTIC"
)

print("=" * 80)


# ============================================================
# 12. Q-LEARNING LOOP
# ============================================================

for episode in range(
    1,
    episodes + 1
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

    print(
        f"\nEpisode {episode:02d}"
        f" | Start features: "
        f"{sum(current_subset):2d}"
        f" | AUC: "
        f"{current_auc:.4f}"
        f" | Objective: "
        f"{current_objective:.4f}"
    )

    for step in range(
        steps_per_episode
    ):

        state = get_state(
            current_subset
        )

        # ----------------------------------------------------
        # EPSILON-GREEDY ACTION SELECTION
        # ----------------------------------------------------

        if random.random() < epsilon:

            action_index = random.randrange(
                N_ACTIONS
            )

        else:

            action_index = int(
                np.argmax(
                    Q[state]
                )
            )


        # ----------------------------------------------------
        # APPLY HEURISTIC
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = (
            new_objective
            - current_objective
        )


        next_state = get_state(
            new_subset
        )


        # ----------------------------------------------------
        # Q UPDATE
        # ----------------------------------------------------

        old_q = Q[
            state,
            action_index
        ]

        best_future_q = np.max(
            Q[next_state]
        )

        Q[
            state,
            action_index
        ] = (
            old_q
            + alpha
            * (
                reward
                + gamma
                * best_future_q
                - old_q
            )
        )


        # ----------------------------------------------------
        # TRANSITION
        # ----------------------------------------------------

        current_subset = (
            new_subset
        )

        current_auc = (
            new_auc
        )

        current_objective = (
            new_objective
        )


        # ----------------------------------------------------
        # TRACK BEST
        # ----------------------------------------------------

        if (
            current_objective
            > best_objective
        ):

            best_objective = (
                current_objective
            )

            best_auc = (
                current_auc
            )

            best_subset = (
                current_subset.copy()
            )


    epsilon = max(
        minimum_epsilon,
        epsilon
        * epsilon_decay
    )


# ============================================================
# 13. FINAL Q-TABLE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FINAL Q-TABLE"
)

print("=" * 80)


q_table = pd.DataFrame(
    Q,
    index=STATE_NAMES,
    columns=ACTION_NAMES
)

print(
    q_table.round(6)
)


# ============================================================
# 14. BEST SUBSET
# ============================================================

best_features = selected_names(
    best_subset
)


print(
    "\n" + "=" * 80
)

print(
    "BEST HEART DISEASE SUBSET"
)

print("=" * 80)


print(
    "Number of features:",
    len(best_features)
)

print(
    f"Validation ROC-AUC: "
    f"{best_auc:.4f}"
)

print(
    f"Objective: "
    f"{best_objective:.4f}"
)


print(
    "\nSelected features:"
)

for feature in best_features:

    print(
        " -",
        feature
    )


# ============================================================
# 15. FINAL TEST EVALUATION - RL SUBSET
# ============================================================

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
    random_state=42
)

final_model.fit(
    X_trainval_best,
    y_trainval
)

rl_prob = final_model.predict_proba(
    X_test_best
)[:, 1]

rl_test_auc = roc_auc_score(
    y_test,
    rl_prob
)


# ============================================================
# 16. ALL-FEATURE TEST BASELINE
# ============================================================

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
    random_state=42
)

baseline_model.fit(
    X_trainval_all,
    y_trainval
)

baseline_prob = (
    baseline_model.predict_proba(
        X_test_all
    )[:, 1]
)

baseline_auc = roc_auc_score(
    y_test,
    baseline_prob
)


# ============================================================
# 17. FINAL COMPARISON
# ============================================================

reduction = (
    1
    - len(best_features)
    / N_FEATURES
) * 100


print(
    "\n" + "=" * 80
)

print(
    "FINAL HEART DISEASE "
    "HELD-OUT TEST COMPARISON"
)

print("=" * 80)


print(
    f"All 13 features ROC-AUC : "
    f"{baseline_auc:.4f}"
)

print(
    f"RL subset ROC-AUC        : "
    f"{rl_test_auc:.4f}"
)

print(
    f"RL number of features    : "
    f"{len(best_features)}"
)

print(
    f"Feature reduction         : "
    f"{reduction:.1f}%"
)

print(
    f"Unique subsets evaluated  : "
    f"{len(evaluation_cache)}"
)