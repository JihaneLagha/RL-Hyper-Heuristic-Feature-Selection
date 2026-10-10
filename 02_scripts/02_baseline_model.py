import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
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

from project_paths import DIABETES_ZIP

ZIP_FILE = DIABETES_ZIP
CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(CSV_FILE) as f:
        df = pd.read_csv(f)

# ------------------------------------------------------------
# 2. Separate predictors and target
# ------------------------------------------------------------
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"].astype(int)

print("=" * 80)
print("DATASET")
print("=" * 80)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 4. Baseline Logistic Regression
# ------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

print("\n" + "=" * 80)
print("TRAINING BASELINE MODEL")
print("=" * 80)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. Predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 6. Metrics
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\n" + "=" * 80)
print("BASELINE RESULTS - ALL 21 FEATURES")
print("=" * 80)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\nConfusion Matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")