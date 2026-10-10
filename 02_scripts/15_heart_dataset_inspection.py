import zipfile
import pandas as pd
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
# 2. BASIC INSPECTION
# ============================================================

print("=" * 80)
print("DATASET SHAPE")
print("=" * 80)

print(df.shape)


print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)

print(df.columns.tolist())


print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)

print(df.head())


print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

print(df.isnull().sum())

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 4. ORIGINAL TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL TARGET DISTRIBUTION")
print("=" * 80)

print(
    df["num"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. CREATE BINARY TARGET
# ============================================================

df["HeartDisease_binary"] = (
    df["num"] > 0
).astype(int)


print("\n" + "=" * 80)
print("BINARY TARGET DISTRIBUTION")
print("=" * 80)

print(
    df["HeartDisease_binary"]
    .value_counts()
    .sort_index()
)


print("\nTarget percentages:")

print(
    df["HeartDisease_binary"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


# ============================================================
# 6. FEATURE COUNT
# ============================================================

X = df.drop(
    columns=[
        "num",
        "HeartDisease_binary"
    ]
)

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print(
    "Number of input features:",
    X.shape[1]
)

print(
    "Features:",
    X.columns.tolist()
)