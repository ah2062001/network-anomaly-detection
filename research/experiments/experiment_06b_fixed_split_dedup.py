from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/experiment_06"
OUT.mkdir(parents=True, exist_ok=True)

features = json.loads(
    (ROOT / "reports/score_diagnostics/metadata.json").read_text()
)["features"]

source = pd.read_csv(ROOT / "data/raw/ddos.csv", low_memory=False)
source.columns = source.columns.str.strip()

y = source["Label"].astype(str).str.strip().eq("DDoS").to_numpy()
idx = np.arange(len(source))

dev_idx, test_idx = train_test_split(
    idx, test_size=0.20, stratify=y, random_state=42
)
train_idx, cal_idx = train_test_split(
    dev_idx, test_size=0.125,
    stratify=y[dev_idx], random_state=42
)

# Fixed evaluation partitions for both models.
X_cal = source.iloc[cal_idx][features].replace(
    [np.inf, -np.inf], np.nan
)
X_test = source.iloc[test_idx][features].replace(
    [np.inf, -np.inf], np.nan
)

# Deduplicate entire source rows, including labels, within
# the training partition only.
original_train = source.iloc[train_idx]
unique_train = original_train.drop_duplicates()

print("Original train rows:", len(original_train))
print("Unique train rows:", len(unique_train))
print("Removed from train:", len(original_train) - len(unique_train))
print("Calibration rows:", len(cal_idx))
print("Holdout rows:", len(test_idx))

results = []

for variant, training in [
    ("original_training", original_train),
    ("training_only_dedup", unique_train),
]:
    print("\nTraining:", variant, flush=True)

    X_train = training[features].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = training["Label"].astype(str).str.strip().eq("DDoS")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=2
        ))
    ])

    model.fit(X_train, y_train)

    cal_scores = model.predict_proba(X_cal)[:, 1]
    threshold = float(
        np.quantile(cal_scores[~y[cal_idx]], 0.99)
    )

    source_scores = model.predict_proba(X_test)[:, 1]

    target_scores = []
    target_labels = []

    target_file = ROOT / (
        "data/raw/cicddos2019/cicddos2019_dataset.csv"
    )

    for chunk in pd.read_csv(
        target_file, chunksize=50000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        X_external = chunk[features].replace(
            [np.inf, -np.inf], np.nan
        )

        target_scores.append(
            model.predict_proba(X_external)[:, 1]
        )
        target_labels.append(
            chunk["Class"].astype(str).str.strip()
            .eq("Attack").to_numpy()
        )

    p = np.concatenate(target_scores)
    yt = np.concatenate(target_labels)

    predicted = p > threshold

    row = {
        "variant": variant,
        "train_rows": len(training),
        "source_holdout_auc": roc_auc_score(
            y[test_idx], source_scores
        ),
        "threshold": threshold,
        "source_calibration_fpr": float(
            (cal_scores[~y[cal_idx]] > threshold).mean()
        ),
        "external_auc": roc_auc_score(yt, p),
        "external_ap": average_precision_score(yt, p),
        "external_recall": float(predicted[yt].mean()),
        "external_fpr": float(predicted[~yt].mean()),
    }

    results.append(row)
    print(pd.DataFrame([row]).to_string(index=False), flush=True)

result = pd.DataFrame(results)
result.to_csv(
    OUT / "fixed_split_dedup_comparison.csv", index=False
)

print("\n=== FIXED-SPLIT COMPARISON ===")
print(result.to_string(index=False))
