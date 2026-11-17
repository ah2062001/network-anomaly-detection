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

SOURCE = ROOT / "data/raw/ddos.csv"
TARGET = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"

design = json.loads(
    (ROOT / "reports/experiment_05/design.json").read_text()
)
metadata = json.loads(
    (ROOT / "reports/score_diagnostics/metadata.json").read_text()
)

features = design["feature_sets"]["A_common"]

assert features == metadata["features"], (
    "Feature lists do not match."
)

source = pd.read_csv(SOURCE, low_memory=False)
source.columns = source.columns.str.strip()

results = []

def evaluate(name, df):
    print("\n" + "=" * 55, flush=True)
    print(name, "| source rows:", len(df), flush=True)

    y = (
        df["Label"].astype(str).str.strip()
        .eq("DDoS").to_numpy(dtype=int)
    )

    X = df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    indices = np.arange(len(df))

    dev_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    train_idx, cal_idx = train_test_split(
        dev_idx,
        test_size=0.125,
        stratify=y[dev_idx],
        random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=2
        ))
    ])

    model.fit(X.iloc[train_idx], y[train_idx])

    cal_scores = model.predict_proba(
        X.iloc[cal_idx]
    )[:, 1]

    benign_cal = cal_scores[y[cal_idx] == 0]
    threshold = float(np.quantile(benign_cal, 0.99))

    holdout_scores = model.predict_proba(
        X.iloc[test_idx]
    )[:, 1]

    source_auc = roc_auc_score(
        y[test_idx], holdout_scores
    )

    ext_scores = []
    ext_y = []

    for chunk in pd.read_csv(
        TARGET, chunksize=50000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        target_y = (
            chunk["Class"].astype(str).str.strip()
            .eq("Attack").to_numpy(dtype=int)
        )

        target_x = chunk[features].replace(
            [np.inf, -np.inf], np.nan
        )

        ext_scores.append(
            model.predict_proba(target_x)[:, 1]
        )
        ext_y.append(target_y)

    scores = np.concatenate(ext_scores)
    target_y = np.concatenate(ext_y)

    pred = scores > threshold
    attack = target_y == 1
    benign = ~attack

    row = {
        "variant": name,
        "source_rows": len(df),
        "training_rows": len(train_idx),
        "source_holdout_auc": source_auc,
        "threshold": threshold,
        "source_calibration_fpr": float(
            (benign_cal > threshold).mean()
        ),
        "external_auc": roc_auc_score(target_y, scores),
        "external_ap": average_precision_score(
            target_y, scores
        ),
        "external_recall": float(pred[attack].mean()),
        "external_fpr": float(pred[benign].mean()),
    }

    results.append(row)

    print("\nRESULT")
    for key, value in row.items():
        print(f"{key}: {value}", flush=True)

    pd.DataFrame(results).to_csv(
        OUT / "baseline_reproduction.csv",
        index=False
    )

    del model

evaluate("original_split", source)

deduplicated = source.drop_duplicates().reset_index(drop=True)
evaluate("deduplicated_split", deduplicated)

print("\n=== COMPARISON ===")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved:", OUT / "baseline_reproduction.csv")
