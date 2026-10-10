import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
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

from project_paths import DIABETES_ZIP

ZIP_FILE = DIABETES_ZIP
CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(CSV_FILE) as f:
        df = pd.read_csv(f)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"].astype(int)

# ------------------------------------------------------------
# 2. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 3. Helper function
# ------------------------------------------------------------
def evaluate_model(k=None):

    steps = []

    if k is not None:
        steps.append(
            ("feature_selection", SelectKBest(score_func=f_classif, k=k))
        )

    steps.extend([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])

    model = Pipeline(steps)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Specificity": tn / (tn + fp)
    }

    selected_features = None

    if k is not None:
        selector = model.named_steps["feature_selection"]
        selected_features = X.columns[
            selector.get_support()
        ].tolist()

    return results, selected_features


# ------------------------------------------------------------
# 4. Test multiple feature counts
# ------------------------------------------------------------
experiments = [
    ("All features", None),
    ("Top 5 features", 5),
    ("Top 10 features", 10),
    ("Top 15 features", 15)
]

print("=" * 90)
print("FEATURE SELECTION BASELINE")
print("=" * 90)

for label, k in experiments:

    results, selected_features = evaluate_model(k)

    print("\n" + "-" * 90)
    print(label)
    print("-" * 90)

    print(f"Accuracy    : {results['Accuracy']:.4f}")
    print(f"Precision   : {results['Precision']:.4f}")
    print(f"Recall      : {results['Recall']:.4f}")
    print(f"F1-score    : {results['F1']:.4f}")
    print(f"ROC-AUC     : {results['ROC_AUC']:.4f}")
    print(f"Specificity : {results['Specificity']:.4f}")

    if selected_features is not None:
        print("\nSelected features:")
        for feature in selected_features:
            print(" -", feature)