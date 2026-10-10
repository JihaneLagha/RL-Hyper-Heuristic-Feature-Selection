import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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


print("=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print(
    y_train.value_counts(
        normalize=True
    ) * 100
)

print("\nTest class distribution:")
print(
    y_test.value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# 4. BASELINE PIPELINE
# ============================================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
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


print("\n" + "=" * 80)
print("TRAINING HEART DISEASE BASELINE")
print("=" * 80)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 5. PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

specificity = tn / (tn + fp)


print("\n" + "=" * 80)
print("HEART DISEASE BASELINE RESULTS")
print("=" * 80)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"F1-score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {roc_auc:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)


print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)