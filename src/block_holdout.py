
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/ddos.csv"

# Load full dataset
df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()].copy()

df["Label"] = df["Label"].astype(str).str.strip()

# Retain the two expected classes
df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

# Preserve original row order
split_index = 175_000

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining labels:")
print(train_df["Label"].value_counts())

print("\nTesting labels:")
print(test_df["Label"].value_counts())

# Binary classification
y_train = (train_df["Label"] == "DDoS").astype(int)
y_test = (test_df["Label"] == "DDoS").astype(int)

# Numeric flow features
X_train = train_df.drop(columns=["Label"])
X_test = test_df.drop(columns=["Label"])

X_train = X_train.select_dtypes(include=np.number).copy()
X_test = X_test[X_train.columns].copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Select columns based only on training data
valid_cols = X_train.columns[X_train.notna().any()]
X_train = X_train[valid_cols]
X_test = X_test[valid_cols]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(
    y_test, pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

if y_test.nunique() == 2:
    print("\nROC-AUC:", roc_auc_score(y_test, prob))
else:
    print("\nROC-AUC: Undefined (only one test class)")
