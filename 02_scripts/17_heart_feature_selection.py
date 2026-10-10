import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from project_paths import HEART_DISEASE_ZIP

ZIP_FILE = HEART_DISEASE_ZIP
DATA_FILE = "processed.cleveland.data"

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


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def evaluate_model(k=None):

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]

    if k is not None:
        steps.append(
            (
                "feature_selection",
                SelectKBest(
                    score_func=f_classif,
                    k=k
                )
            )
        )

    steps.extend([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=42
            )
        )
    ])

    model = Pipeline(
        steps
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    results = {
        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred
            ),

        "F1":
            f1_score(
                y_test,
                y_pred
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob
            ),

        "Specificity":
            tn / (tn + fp)
    }

    selected_features = None

    if k is not None:

        selector = model.named_steps[
            "feature_selection"
        ]

        selected_features = X.columns[
            selector.get_support()
        ].tolist()

    return (
        results,
        selected_features
    )


# ============================================================
# 5. EXPERIMENTS
# ============================================================

experiments = [
    ("All features", None),
    ("Top 5 features", 5),
    ("Top 8 features", 8),
    ("Top 10 features", 10)
]


print("=" * 90)
print("HEART DISEASE FEATURE-SELECTION BASELINE")
print("=" * 90)


for label, k in experiments:

    results, selected_features = evaluate_model(
        k
    )

    print("\n" + "-" * 90)
    print(label)
    print("-" * 90)

    print(
        f"Accuracy    : "
        f"{results['Accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{results['Precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{results['Recall']:.4f}"
    )

    print(
        f"F1-score    : "
        f"{results['F1']:.4f}"
    )

    print(
        f"ROC-AUC     : "
        f"{results['ROC_AUC']:.4f}"
    )

    print(
        f"Specificity : "
        f"{results['Specificity']:.4f}"
    )

    if selected_features is not None:

        print("\nSelected features:")

        for feature in selected_features:
            print(" -", feature)