import zipfile
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    RFE,
    SelectFromModel
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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

RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

with zipfile.ZipFile(ZIP_FILE, "r") as z:

    with z.open(DATA_FILE) as f:

        df = pd.read_csv(f)


X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

FEATURE_NAMES = X.columns.tolist()

N_FEATURES = X.shape[1]


print("=" * 90)
print("DIABETES CONVENTIONAL FEATURE-SELECTION BASELINES")
print("=" * 90)

print(
    "Dataset:",
    X.shape
)


# ============================================================
# 2. SAME TRAIN / VALIDATION / TEST STRUCTURE
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


print(
    "\nTrain:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# 3. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)


# ============================================================
# 4. SUBSET EVALUATION
# ============================================================

def evaluate_feature_set(
    feature_names,
    method_name
):

    final_scaler = StandardScaler()

    X_trainval_selected = (
        final_scaler.fit_transform(
            X_trainval[
                feature_names
            ]
        )
    )

    X_test_selected = (
        final_scaler.transform(
            X_test[
                feature_names
            ]
        )
    )


    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_trainval_selected,
        y_trainval
    )


    predictions = model.predict(
        X_test_selected
    )

    probabilities = model.predict_proba(
        X_test_selected
    )[:, 1]


    auc = roc_auc_score(
        y_test,
        probabilities
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    specificity = tn / (
        tn + fp
    )


    reduction = (
        1
        - len(feature_names)
        / N_FEATURES
    ) * 100


    objective = (
        auc
        - 0.02
        * (
            len(feature_names)
            / N_FEATURES
        )
    )


    return {
        "Method":
            method_name,

        "Features":
            len(feature_names),

        "Reduction":
            reduction,

        "ROC_AUC":
            auc,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Specificity":
            specificity,

        "Objective":
            objective,

        "Selected_Features":
            ",".join(
                feature_names
            )
    }


results = []


# ============================================================
# 5. ALL FEATURES
# ============================================================

results.append(
    evaluate_feature_set(
        FEATURE_NAMES,
        "ALL_FEATURES"
    )
)


# ============================================================
# 6. SELECTKBEST
# ============================================================
#
# Test several candidate values using validation only.
# ============================================================

best_k_auc = -1
best_k_features = None
best_k = None


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
        X_train_scaled,
        y_train
    )


    selected_indices = (
        selector.get_support(
            indices=True
        )
    )

    selected_features = [
        FEATURE_NAMES[i]
        for i in selected_indices
    ]


    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train[
            selected_features
        ],
        y_train
    )


    val_prob = model.predict_proba(
        X_val[
            selected_features
        ]
    )[:, 1]


    auc = roc_auc_score(
        y_val,
        val_prob
    )


    objective = (
        auc
        - 0.02
        * (
            k / N_FEATURES
        )
    )


    print(
        f"SelectKBest k={k:2d}"
        f" | Validation AUC: "
        f"{auc:.4f}"
        f" | Objective: "
        f"{objective:.4f}"
    )


    if objective > best_k_auc:

        best_k_auc = objective

        best_k_features = (
            selected_features
        )

        best_k = k


print(
    "\nBest SelectKBest k:",
    best_k
)


results.append(
    evaluate_feature_set(
        best_k_features,
        f"SelectKBest_{best_k}"
    )
)


# ============================================================
# 7. L1 FEATURE SELECTION
# ============================================================

l1_model = LogisticRegression(
    penalty="l1",
    C=0.05,
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

l1_model.fit(
    X_train_scaled,
    y_train
)


selector_l1 = SelectFromModel(
    l1_model,
    prefit=True,
    threshold="median"
)


l1_indices = selector_l1.get_support(
    indices=True
)


l1_features = [
    FEATURE_NAMES[i]
    for i in l1_indices
]


print(
    "\nL1 selected features:",
    len(l1_features)
)


results.append(
    evaluate_feature_set(
        l1_features,
        "L1"
    )
)


# ============================================================
# 8. RFE
# ============================================================

best_rfe_objective = -1
best_rfe_features = None
best_rfe_k = None


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
        X_train_scaled,
        y_train
    )


    indices = rfe.get_support(
        indices=True
    )


    selected_features = [
        FEATURE_NAMES[i]
        for i in indices
    ]


    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )


    model.fit(
        X_train[
            selected_features
        ],
        y_train
    )


    probabilities = model.predict_proba(
        X_val[
            selected_features
        ]
    )[:, 1]


    auc = roc_auc_score(
        y_val,
        probabilities
    )


    objective = (
        auc
        - 0.02
        * (
            k / N_FEATURES
        )
    )


    print(
        f"RFE k={k:2d}"
        f" | Validation AUC: "
        f"{auc:.4f}"
        f" | Objective: "
        f"{objective:.4f}"
    )


    if objective > best_rfe_objective:

        best_rfe_objective = (
            objective
        )

        best_rfe_features = (
            selected_features
        )

        best_rfe_k = k


print(
    "\nBest RFE k:",
    best_rfe_k
)


results.append(
    evaluate_feature_set(
        best_rfe_features,
        f"RFE_{best_rfe_k}"
    )
)


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "Objective",
    ascending=False
)


print(
    "\n" + "=" * 90
)

print(
    "FINAL CONVENTIONAL BASELINE COMPARISON"
)

print("=" * 90)


print(
    results_df[
        [
            "Method",
            "Features",
            "Reduction",
            "ROC_AUC",
            "Accuracy",
            "Recall",
            "F1",
            "Specificity",
            "Objective"
        ]
    ].to_string(
        index=False,
        formatters={
            "Reduction":
                "{:.1f}%".format,

            "ROC_AUC":
                "{:.4f}".format,

            "Accuracy":
                "{:.4f}".format,

            "Recall":
                "{:.4f}".format,

            "F1":
                "{:.4f}".format,

            "Specificity":
                "{:.4f}".format,

            "Objective":
                "{:.4f}".format
        }
    )
)


# ============================================================
# 10. PRINT FEATURE SUBSETS
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "SELECTED FEATURE SUBSETS"
)

print("=" * 90)


for _, row in results_df.iterrows():

    print(
        f"\n{row['Method']}"
        f" ({row['Features']} features)"
    )

    print(
        row[
            "Selected_Features"
        ]
    )


# ============================================================
# 11. SAVE
# ============================================================

OUTPUT_FILE = (
    DIABETES_RESULTS_DIR
    / "21_diabetes_conventional_baselines.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)