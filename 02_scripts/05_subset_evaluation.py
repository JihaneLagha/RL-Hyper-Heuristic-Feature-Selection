import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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

FEATURES = X.columns.tolist()


# ------------------------------------------------------------
# 2. Split into train / validation / test
# ------------------------------------------------------------
# First keep 20% completely untouched for final testing
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Split remaining 80% into:
# 75% training
# 25% validation
#
# This gives:
# 60% train
# 20% validation
# 20% test overall
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


# ------------------------------------------------------------
# 3. Function to evaluate a feature subset
# ------------------------------------------------------------
def evaluate_subset(selected_features):

    if len(selected_features) == 0:
        return None

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])

    model.fit(
        X_train[selected_features],
        y_train
    )

    val_prob = model.predict_proba(
        X_val[selected_features]
    )[:, 1]

    roc_auc = roc_auc_score(
        y_val,
        val_prob
    )

    return roc_auc


# ------------------------------------------------------------
# 4. Test several manually defined subsets
# ------------------------------------------------------------

all_features = FEATURES.copy()

subset_15 = [
    'HighBP',
    'HighChol',
    'CholCheck',
    'BMI',
    'Smoker',
    'Stroke',
    'HeartDiseaseorAttack',
    'PhysActivity',
    'GenHlth',
    'MentHlth',
    'PhysHlth',
    'DiffWalk',
    'Age',
    'Education',
    'Income'
]

subset_10 = [
    'HighBP',
    'HighChol',
    'BMI',
    'HeartDiseaseorAttack',
    'GenHlth',
    'PhysHlth',
    'DiffWalk',
    'Age',
    'Education',
    'Income'
]

subset_5 = [
    'HighBP',
    'HighChol',
    'BMI',
    'GenHlth',
    'DiffWalk'
]


experiments = {
    "All 21 features": all_features,
    "15 features": subset_15,
    "10 features": subset_10,
    "5 features": subset_5
}


print("\n" + "=" * 80)
print("VALIDATION ROC-AUC BY FEATURE SUBSET")
print("=" * 80)

for name, feature_list in experiments.items():

    score = evaluate_subset(feature_list)

    print(f"\n{name}")
    print("-" * 40)

    print("Number of features:", len(feature_list))
    print(f"Validation ROC-AUC: {score:.4f}")


# ------------------------------------------------------------
# 5. Example reward function
# ------------------------------------------------------------

def calculate_reward(roc_auc, number_features, total_features):

    lambda_penalty = 0.02

    feature_penalty = (
        number_features / total_features
    )

    reward = (
        roc_auc
        - lambda_penalty * feature_penalty
    )

    return reward


print("\n" + "=" * 80)
print("EXAMPLE REWARD VALUES")
print("=" * 80)

for name, feature_list in experiments.items():

    roc_auc = evaluate_subset(feature_list)

    reward = calculate_reward(
        roc_auc,
        len(feature_list),
        len(FEATURES)
    )

    print(f"\n{name}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Features: {len(feature_list)}")
    print(f"Reward  : {reward:.4f}")