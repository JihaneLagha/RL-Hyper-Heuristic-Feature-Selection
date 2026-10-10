import zipfile
import pandas as pd

from project_paths import DIABETES_ZIP


ZIP_FILE = DIABETES_ZIP
CSV_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"


with zipfile.ZipFile(ZIP_FILE, "r") as z:
    with z.open(CSV_FILE) as f:
        df = pd.read_csv(f)


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

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(df.isnull().sum())

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)
print(df["Diabetes_binary"].value_counts())

print("\nTarget percentages:")
print(
    df["Diabetes_binary"]
    .value_counts(normalize=True)
    * 100
)