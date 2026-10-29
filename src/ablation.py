
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/ddos.csv"

df = pd.read_csv(DATA, nrows=100_000, low_memory=False)
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()].copy()
df["Label"] = df["Label"].astype(str).str.strip()

df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

y = (df["Label"] == "DDoS").astype(int)

X = df.drop(columns=["Label"])
X = X.select_dtypes(include=np.number)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

# Fit baseline using training data
baseline = create_model()
baseline.fit(X_train, y_train)

rf = baseline.named_steps["classifier"]
feature_names = baseline.named_steps["imputer"].get_feature_names_out()

ranked = pd.Series(
    rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 15 features:")
print(ranked.head(15))

experiments = {
    "All features": [],
    "Remove top 1": ranked.head(1).index.tolist(),
    "Remove top 5": ranked.head(5).index.tolist(),
    "Remove top 15": ranked.head(15).index.tolist()
}

results = []

for name, removed in experiments.items():
    print(f"\nRunning: {name}")

    if name == "All features":
        model = baseline
    else:
        model = create_model()
        model.fit(
            X_train.drop(columns=removed),
            y_train
        )

    X_eval = X_test.drop(columns=removed)
    pred = model.predict(X_eval)

    results.append({
        "Experiment": name,
        "Features": X_eval.shape[1],
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    })

results_df = pd.DataFrame(results)

print("\nAblation results:")
print(results_df.to_string(index=False))

output = ROOT / "reports"
output.mkdir(exist_ok=True)

results_df.to_csv(output / "ablation_results.csv", index=False)
