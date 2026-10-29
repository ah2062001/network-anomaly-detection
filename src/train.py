
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/ddos.csv"
MODEL_PATH = ROOT / "models/random_forest.joblib"

# 1. Load a manageable subset
df = pd.read_csv(DATA_PATH, nrows=100_000, low_memory=False)

# Normalize column names
df.columns = df.columns.str.strip()

# Remove duplicate column names
df = df.loc[:, ~df.columns.duplicated()].copy()

# 2. Prepare labels
df["Label"] = df["Label"].astype(str).str.strip()

print("Original labels:")
print(df["Label"].value_counts())

# 0 = BENIGN, 1 = DDoS
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()
df["target"] = (df["Label"] == "DDoS").astype(int)

if df["target"].nunique() < 2:
    raise ValueError(
        "The loaded rows contain only one class. "
        "Load more rows or sample across the CSV."
    )

# 3. Select numeric features
X = df.drop(columns=["Label", "target"])
X = X.select_dtypes(include=np.number).copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Remove entirely empty columns
X = X.dropna(axis=1, how="all")

y = df["target"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Preprocessing and model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# 6. Train
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# 7. Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["BENIGN", "DDoS"],
    zero_division=0,
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nROC-AUC:", roc_auc_score(y_test, probabilities))

# 8. Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
