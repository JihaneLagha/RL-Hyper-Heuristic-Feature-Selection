import zipfile
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project_paths import DIABETES_ZIP

# ============================================================
# SETTINGS
# ============================================================

ZIP_FILE = DIABETES_ZIP
CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


print("=" * 80)
print("DATA SPLIT")
print("=" * 80)

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# ============================================================
# 3. USE A DEVELOPMENT SAMPLE DURING RL SEARCH
# ============================================================
#
# Retraining logistic regression hundreds of times on all
# 152,208 training samples is unnecessary for the search.
#
# We therefore use a stratified sample from TRAINING ONLY.
# The test set remains untouched.
# ============================================================

X_search, _, y_search, _ = train_test_split(
    X_train,
    y_train,
    train_size=25000,
    random_state=123,
    stratify=y_train
)

print("\nRL search training sample:", X_search.shape)


# ============================================================
# 4. STANDARDIZE ONCE
# ============================================================

scaler = StandardScaler()

X_search_scaled = scaler.fit_transform(X_search)

# Use the SAME scaler learned from search training data
X_val_scaled = scaler.transform(X_val)


# ============================================================
# 5. FEATURE SUBSET REPRESENTATION
# ============================================================

def random_subset(min_features=5, max_features=18):

    number_selected = random.randint(
        min_features,
        max_features
    )

    indices = random.sample(
        range(N_FEATURES),
        number_selected
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

    # Prevent empty subset
    if (
        new_subset[index] == 1
        and sum(new_subset) == 1
    ):
        return new_subset

    new_subset[index] = 1 - new_subset[index]

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

    validation_prob = model.predict_proba(
        X_val_scaled[:, indices]
    )[:, 1]

    roc_auc = roc_auc_score(
        y_val,
        validation_prob
    )

    # Reward objective
    lambda_penalty = 0.02

    objective = (
        roc_auc
        - lambda_penalty
        * (len(indices) / N_FEATURES)
    )

    evaluation_cache[key] = (
        roc_auc,
        objective
    )

    return roc_auc, objective


# ============================================================
# 8. DEFINE RL STATES
# ============================================================
#
# First prototype:
#
# State 0 = small subset      1-7 features
# State 1 = medium subset     8-14 features
# State 2 = large subset      15-21 features
#
# Later we can design a richer state representation.
# ============================================================

def get_state(subset):

    number_features = sum(subset)

    if number_features <= 7:
        return 0

    elif number_features <= 14:
        return 1

    else:
        return 2


STATE_NAMES = [
    "SMALL",
    "MEDIUM",
    "LARGE"
]


# ============================================================
# 9. Q-LEARNING SETTINGS
# ============================================================

N_STATES = 3
N_ACTIONS = 4

Q = np.zeros(
    (N_STATES, N_ACTIONS)
)

alpha = 0.20
gamma = 0.80

epsilon = 0.30
epsilon_decay = 0.95
minimum_epsilon = 0.05

episodes = 10
steps_per_episode = 15


# ============================================================
# 10. STORE GLOBAL BEST
# ============================================================

best_subset = None
best_auc = -1
best_objective = -1


print("\n" + "=" * 80)
print("STARTING Q-LEARNING HYPER-HEURISTIC")
print("=" * 80)


# ============================================================
# 11. Q-LEARNING LOOP
# ============================================================

for episode in range(1, episodes + 1):

    current_subset = random_subset()

    current_auc, current_objective = evaluate_subset(
        current_subset
    )

    print(
        f"\nEpisode {episode:02d} "
        f"| Start features: {sum(current_subset):2d} "
        f"| AUC: {current_auc:.4f} "
        f"| Objective: {current_objective:.4f}"
    )

    for step in range(steps_per_episode):

        state = get_state(current_subset)

        # ---------------------------------------------
        # Epsilon-greedy action selection
        # ---------------------------------------------

        if random.random() < epsilon:

            action_index = random.randrange(
                N_ACTIONS
            )

        else:

            action_index = int(
                np.argmax(Q[state])
            )

        # ---------------------------------------------
        # Apply chosen heuristic
        # ---------------------------------------------

        action_function = ACTIONS[action_index]

        new_subset = action_function(
            current_subset
        )

        new_auc, new_objective = evaluate_subset(
            new_subset
        )

        # ---------------------------------------------
        # Reward = change in optimization objective
        # ---------------------------------------------

        reward = (
            new_objective
            - current_objective
        )

        next_state = get_state(
            new_subset
        )

        # ---------------------------------------------
        # Q-learning update
        # ---------------------------------------------

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
                + gamma * best_future_q
                - old_q
            )
        )

        # ---------------------------------------------
        # Transition to new solution
        # ---------------------------------------------

        current_subset = new_subset

        current_auc = new_auc

        current_objective = new_objective

        # ---------------------------------------------
        # Track global best
        # ---------------------------------------------

        if current_objective > best_objective:

            best_objective = current_objective

            best_auc = current_auc

            best_subset = current_subset.copy()

    # Reduce exploration gradually
    epsilon = max(
        minimum_epsilon,
        epsilon * epsilon_decay
    )


# ============================================================
# 12. SHOW LEARNED Q-TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL Q-TABLE")
print("=" * 80)

q_table = pd.DataFrame(
    Q,
    index=STATE_NAMES,
    columns=ACTION_NAMES
)

print(q_table.round(6))


# ============================================================
# 13. BEST FEATURE SUBSET
# ============================================================

print("\n" + "=" * 80)
print("BEST SUBSET FOUND DURING RL SEARCH")
print("=" * 80)

print("Number of features:", sum(best_subset))

print(
    f"Validation ROC-AUC: {best_auc:.4f}"
)

print(
    f"Optimization objective: "
    f"{best_objective:.4f}"
)

print("\nSelected features:")

for feature in selected_names(best_subset):
    print(" -", feature)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================
#
# IMPORTANT:
# Test data have not been used during RL search.
# ============================================================

best_features = selected_names(
    best_subset
)

final_scaler = StandardScaler()

X_trainval_best = final_scaler.fit_transform(
    X_trainval[best_features]
)

X_test_best = final_scaler.transform(
    X_test[best_features]
)

final_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

final_model.fit(
    X_trainval_best,
    y_trainval
)

test_probability = final_model.predict_proba(
    X_test_best
)[:, 1]

rl_test_auc = roc_auc_score(
    y_test,
    test_probability
)


# ============================================================
# 15. ALL-FEATURE TEST BASELINE
# ============================================================

baseline_scaler = StandardScaler()

X_trainval_all = baseline_scaler.fit_transform(
    X_trainval
)

X_test_all = baseline_scaler.transform(
    X_test
)

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

baseline_model.fit(
    X_trainval_all,
    y_trainval
)

baseline_probability = baseline_model.predict_proba(
    X_test_all
)[:, 1]

baseline_test_auc = roc_auc_score(
    y_test,
    baseline_probability
)


# ============================================================
# 16. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FINAL HELD-OUT TEST COMPARISON")
print("=" * 80)

print(
    f"All 21 features ROC-AUC : "
    f"{baseline_test_auc:.4f}"
)

print(
    f"RL subset ROC-AUC        : "
    f"{rl_test_auc:.4f}"
)

print(
    f"RL number of features    : "
    f"{len(best_features)}"
)

feature_reduction = (
    1
    - len(best_features) / N_FEATURES
) * 100

print(
    f"Feature reduction         : "
    f"{feature_reduction:.1f}%"
)

print("\nUnique subsets evaluated:")
print(len(evaluation_cache))