import zipfile
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    RFE,
    SelectFromModel,
)
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
    "num",
]


# ============================================================
# LOAD DATA
# ============================================================

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(DATA_FILE) as f:

        df = pd.read_csv(
            f,
            header=None,
            names=COLUMNS,
            na_values="?",
        )


df["HeartDisease_binary"] = (
    df["num"] > 0
).astype(int)


X = df.drop(
    columns=[
        "num",
        "HeartDisease_binary",
    ]
)

y = df["HeartDisease_binary"]

FEATURE_NAMES = X.columns.tolist()

N_FEATURES = len(FEATURE_NAMES)


# ============================================================
# FINAL SUBSET EVALUATION
# ============================================================

def evaluate_final(
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    features,
):

    imputer = SimpleImputer(
        strategy="median",
    )

    train_imp = imputer.fit_transform(
        X_trainval[features]
    )

    test_imp = imputer.transform(
        X_test[features]
    )


    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_imp
    )

    test_scaled = scaler.transform(
        test_imp
    )


    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        train_scaled,
        y_trainval,
    )


    probability = model.predict_proba(
        test_scaled
    )[:, 1]


    auc = roc_auc_score(
        y_test,
        probability,
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
# RUN ONE SPLIT
# ============================================================

def run_seed(seed):

    (
        X_trainval,
        X_test,
        y_trainval,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )


    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=seed,
        stratify=y_trainval,
    )


    # ========================================================
    # TRAIN-SPLIT IMPUTATION + SCALING
    # ========================================================

    imputer = SimpleImputer(
        strategy="median",
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_val_imp = imputer.transform(
        X_val
    )


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imp
    )

    X_val_scaled = scaler.transform(
        X_val_imp
    )


    results = []


    # ========================================================
    # ALL FEATURES
    # ========================================================

    auc, objective = evaluate_final(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        FEATURE_NAMES,
    )


    results.append({
        "Seed": seed,
        "Method": "ALL_FEATURES",
        "Features": N_FEATURES,
        "Test_AUC": auc,
        "Objective": objective,
    })


    # ========================================================
    # SELECTKBEST
    # ========================================================

    best_objective = -1
    best_features = None


    for k in [
        3,
        5,
        8,
        10,
        12,
    ]:

        selector = SelectKBest(
            score_func=f_classif,
            k=k,
        )

        selector.fit(
            X_train_imp,
            y_train,
        )


        indices = selector.get_support(
            indices=True,
        )


        features = [
            FEATURE_NAMES[i]
            for i in indices
        ]


        model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )


        model.fit(
            X_train_scaled[:, indices],
            y_train,
        )


        probability = model.predict_proba(
            X_val_scaled[:, indices]
        )[:, 1]


        auc_val = roc_auc_score(
            y_val,
            probability,
        )


        objective_val = (
            auc_val
            - LAMBDA_PENALTY
            * (
                k / N_FEATURES
            )
        )


        if objective_val > best_objective:

            best_objective = (
                objective_val
            )

            best_features = (
                features
            )


    auc, objective = evaluate_final(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        best_features,
    )


    results.append({
        "Seed": seed,
        "Method": "SelectKBest",
        "Features": len(best_features),
        "Test_AUC": auc,
        "Objective": objective,
    })


    # ========================================================
    # L1 FEATURE SELECTION
    # ========================================================

    l1 = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )


    l1.fit(
        X_train_scaled,
        y_train,
    )


    l1_selector = SelectFromModel(
        l1,
        prefit=True,
        threshold="median",
    )


    indices = l1_selector.get_support(
        indices=True,
    )


    l1_features = [
        FEATURE_NAMES[i]
        for i in indices
    ]


    auc, objective = evaluate_final(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        l1_features,
    )


    results.append({
        "Seed": seed,
        "Method": "L1",
        "Features": len(l1_features),
        "Test_AUC": auc,
        "Objective": objective,
    })


    # ========================================================
    # RFE
    # ========================================================

    best_objective = -1
    best_features = None


    for k in [
        3,
        5,
        8,
        10,
        12,
    ]:

        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )


        rfe = RFE(
            estimator=estimator,
            n_features_to_select=k,
            step=1,
        )


        rfe.fit(
            X_train_scaled,
            y_train,
        )


        indices = rfe.get_support(
            indices=True,
        )


        features = [
            FEATURE_NAMES[i]
            for i in indices
        ]


        model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )


        model.fit(
            X_train_scaled[:, indices],
            y_train,
        )


        probability = model.predict_proba(
            X_val_scaled[:, indices]
        )[:, 1]


        auc_val = roc_auc_score(
            y_val,
            probability,
        )


        objective_val = (
            auc_val
            - LAMBDA_PENALTY
            * (
                k / N_FEATURES
            )
        )


        if objective_val > best_objective:

            best_objective = (
                objective_val
            )

            best_features = (
                features
            )


    auc, objective = evaluate_final(
        X_trainval,
        y_trainval,
        X_test,
        y_test,
        best_features,
    )


    results.append({
        "Seed": seed,
        "Method": "RFE",
        "Features": len(best_features),
        "Test_AUC": auc,
        "Objective": objective,
    })


    return results


# ============================================================
# RUN 30 SPLITS
# ============================================================

all_results = []


print("=" * 95)
print("HEART DISEASE CONVENTIONAL BASELINES - 30 SPLITS")
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
            "mean",
        ),

        Std_AUC=(
            "Test_AUC",
            "std",
        ),

        Mean_Features=(
            "Features",
            "mean",
        ),

        Mean_Reduction=(
            "Reduction",
            "mean",
        ),

        Mean_Objective=(
            "Objective",
            "mean",
        ),

        Std_Objective=(
            "Objective",
            "std",
        ),
    )
    .reset_index()
)


print(
    "\n" + "=" * 95
)

print(
    "HEART DISEASE 30-SPLIT CONVENTIONAL SUMMARY"
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
                "{:.4f}".format,
        },
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    HEART_RESULTS_DIR
    / "24_heart_conventional_results.csv",
    index=False,
)

summary.to_csv(
    HEART_RESULTS_DIR
    / "24_heart_conventional_summary.csv",
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    HEART_RESULTS_DIR
    / "24_heart_conventional_results.csv"
)

print(
    HEART_RESULTS_DIR
    / "24_heart_conventional_summary.csv"
)