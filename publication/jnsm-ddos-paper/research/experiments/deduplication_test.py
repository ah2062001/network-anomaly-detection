
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/ddos.csv"

df = pd.read_csv(DATA, nrows=100_000, low_memory=False)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

def evaluate(data, name):
    X = data.drop(columns=["Label"]).select_dtypes(
        include=np.number
    ).copy()

    y = (data["Label"] == "DDoS").astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove constant and entirely missing features
    keep = X.columns[
        (X.nunique(dropna=True) > 1)
    ]
    X = X[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n=== {name} ===")
    print("Rows:", len(data))
    print("Features:", X.shape[1])
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("F1:", f1_score(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

evaluate(df, "Original")

deduplicated = df.drop_duplicates().copy()

print("\nRemoved duplicate rows:", len(df) - len(deduplicated))

evaluate(deduplicated, "Deduplicated")
