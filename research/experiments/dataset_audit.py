
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/ddos.csv"

df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()

print("=== DATASET AUDIT ===")

print("\nShape:", df.shape)

print("\nDuplicate rows:", df.duplicated().sum())

print("\nMissing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nInfinite values:")
numeric = df.select_dtypes(include=np.number)
print(np.isinf(numeric.to_numpy()).sum())

print("\nConstant columns:")
print([
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
])

print("\nClass distribution:")
print(df["Label"].astype(str).str.strip().value_counts())

print("\nPossible identifier/time columns:")
print([
    col for col in df.columns
    if any(term in col.lower() for term in [
        "timestamp", "flow id", "source ip", "destination ip"
    ])
])
