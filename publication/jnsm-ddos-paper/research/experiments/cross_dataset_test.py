
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]

train_path = ROOT / "data/raw/ddos.csv"
test_path = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"

print("Loading datasets...")

train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

train["Label"] = train["Label"].astype(str).str.strip()
test["Class"] = test["Class"].astype(str).str.strip()
test["Label"] = test["Label"].astype(str).str.strip()

train = train[train["Label"].isin(["BENIGN", "DDoS"])].copy()
test = test[test["Class"].isin(["Attack", "Benign"])].copy()

# Only shared, numeric flow features.
# Explicitly exclude labels, identifiers and metadata.
excluded = {
    "Label", "Class", "Protocol",
    "Unnamed: 0", "Flow ID",
    "Source IP", "Destination IP", "Timestamp",
}

shared = sorted(
    (set(train.columns) & set(test.columns)) - excluded
)

shared = [
    col for col in shared
    if pd.api.types.is_numeric_dtype(train[col])
    and pd.api.types.is_numeric_dtype(test[col])
]

X_train = train[shared].replace([np.inf, -np.inf], np.nan)
X_test = test[shared].replace([np.inf, -np.inf], np.nan)

y_train = (train["Label"] == "DDoS").astype(int)
y_test = (test["Class"] == "Attack").astype(int)

# Feature selection based on training data only.
usable = X_train.columns[X_train.nunique(dropna=True) > 1]

X_train = X_train[usable]
X_test = X_test[usable]

print("\nTraining rows:", len(X_train))
print("External test rows:", len(X_test))
print("Common usable features:", len(usable))

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])


from sklearn.model_selection import train_test_split

# Internal evaluation using the exact same 57 features
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

internal_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

print("\n=== MATCHED-FEATURE INTERNAL BASELINE ===")
internal_model.fit(Xi_train, yi_train)
internal_pred = internal_model.predict(Xi_test)

print("Accuracy:", accuracy_score(yi_test, internal_pred))
print("Precision:", precision_score(yi_test, internal_pred))
print("Recall:", recall_score(yi_test, internal_pred))
print("F1:", f1_score(yi_test, internal_pred))
print("Confusion matrix:")
print(confusion_matrix(yi_test, internal_pred))


print("\nTraining model...")
model.fit(X_train, y_train)

print("Evaluating on CIC-DDoS2019...")
pred = model.predict(X_test)

print("\n=== EXTERNAL VALIDATION ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1:", f1_score(y_test, pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test, pred,
    target_names=["Benign", "Attack"],
    zero_division=0,
))

# Diagnose performance for individual attack types.
results = pd.DataFrame({
    "Attack Type": test["Label"].to_numpy(),
    "Correct": pred == y_test.to_numpy(),
})

by_type = (
    results.groupby("Attack Type")["Correct"]
    .agg(["count", "mean"])
    .sort_values("mean")
)

print("\n=== PERFORMANCE BY TRAFFIC TYPE ===")
print(by_type.to_string())

report_dir = ROOT / "reports"
report_dir.mkdir(exist_ok=True)

by_type.to_csv(report_dir / "cross_dataset_by_type.csv")
print("\nSaved reports/cross_dataset_by_type.csv")


# Diagnostic: detection rate by backward-packet count
diagnostic = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": pred,
    "BackwardPackets": X_test["Total Backward Packets"].to_numpy(),
    "AttackType": test["Label"].to_numpy(),
})

attacks = diagnostic[diagnostic["Actual"] == 1].copy()

attacks["BackwardTraffic"] = np.where(
    attacks["BackwardPackets"] == 0,
    "Zero backward packets",
    "At least one backward packet"
)

summary = attacks.groupby("BackwardTraffic").agg(
    Flows=("Predicted", "size"),
    Detected=("Predicted", "sum"),
    DetectionRate=("Predicted", "mean")
)

print("\n=== DETECTION BY BACKWARD TRAFFIC ===")
print(summary.to_string())

# Separate analysis for SYN and non-SYN attacks
attacks["AttackGroup"] = np.where(
    attacks["AttackType"] == "Syn",
    "Syn",
    "Other attacks"
)

print("\n=== DETECTION BY ATTACK GROUP ===")
print(
    attacks.groupby("AttackGroup")["Predicted"]
    .agg(["count", "sum", "mean"])
    .to_string()
)
