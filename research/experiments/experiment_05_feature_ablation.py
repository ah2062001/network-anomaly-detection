from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/ddos.csv"
TARGET = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"
OUT = ROOT / "reports/experiment_05"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TREES = 100
SOURCE_FPR_BUDGET = 0.01

# Candidate feature aliases established from schema inspection.
# These are NOT yet proof of identical extraction semantics.
ALIASES = {
    "Average Packet Size": "Avg Packet Size",
    "Max Packet Length": "Packet Length Max",
    "Min Packet Length": "Packet Length Min",
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

# Explicitly selected traffic-oriented features.
# Keep this list fixed across evaluation runs.
CORE_CANDIDATES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Down/Up Ratio",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
]

EXCLUDE = {
    "Label", "Class", "Flow ID",
    "Source IP", "Destination IP",
    "Source Port", "Destination Port",
    "Timestamp", "Protocol", "Unnamed: 0",
}

def as_numeric_matrix(df, names):
    columns = []
    for name in names:
        x = pd.to_numeric(
            df[name], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        x = x.clip(-1e20, 1e20)
        columns.append(x.to_numpy(dtype=np.float32))

    return np.column_stack(columns)

def fill_missing(x, medians):
    return np.where(np.isfinite(x), x, medians).astype(
        np.float32, copy=False
    )

def calculate_metrics(y, scores, threshold):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores)
    pred = scores > threshold

    tp = int(((y == 1) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "balanced_accuracy": (
            (tp / (tp + fn) + tn / (tn + fp)) / 2
            if tp + fn and tn + fp else np.nan
        ),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
    }

print("Loading source data...")
source = pd.read_csv(SOURCE, low_memory=False)
source.columns = source.columns.str.strip()

print("Source rows before deduplication:", len(source))
source = source.drop_duplicates().reset_index(drop=True)
print("Source rows after deduplication:", len(source))

source_labels = set(source["Label"].astype(str).str.strip())
assert source_labels == {"BENIGN", "DDoS"}, source_labels

y = (
    source["Label"].astype(str).str.strip()
    .eq("DDoS").to_numpy(dtype=np.int8)
)

target_columns = set(
    pd.read_csv(TARGET, nrows=0).columns.str.strip()
)

shared = sorted(
    (set(source.columns) & target_columns) - EXCLUDE
)

# Use source data only to determine numeric usability and
# remove source-constant columns.
base = []

for name in shared:
    v = pd.to_numeric(
        source[name], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    if v.notna().mean() >= 0.95 and v.nunique(dropna=True) > 1:
        base.append(name)

core = [name for name in CORE_CANDIDATES if name in base]

extra = [
    name for name, target_name in ALIASES.items()
    if name in source.columns and target_name in target_columns
    and pd.to_numeric(
        source[name], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    .nunique(dropna=True) > 1
]

feature_sets = {
    "A_common": base,
    "B_core": core,
    "C_expanded": base + extra,
}

assert len(core) >= 20, "Core set unexpectedly small."
assert len(extra) >= 3, "Candidate aliases unexpectedly missing."

for name, features in feature_sets.items():
    assert len(features) == len(set(features))
    print(name, "features:", len(features))

# Deduplicated, label-stratified, index-based random split.
# No chronological interpretation is justified.
idx = np.arange(len(source))

dev_idx, test_idx = train_test_split(
    idx, test_size=0.20, stratify=y, random_state=SEED
)

fit_idx, cal_idx = train_test_split(
    dev_idx,
    test_size=0.125,
    stratify=y[dev_idx],
    random_state=SEED
)

print("Train:", len(fit_idx))
print("Calibration:", len(cal_idx))
print("Source holdout:", len(test_idx))

design = {
    "experiment": "05_feature_representation_ablation",
    "exploratory_target": True,
    "source_exact_deduplication": True,
    "split_seed": SEED,
    "train_n": len(fit_idx),
    "calibration_n": len(cal_idx),
    "source_holdout_n": len(test_idx),
    "model": "RandomForestClassifier",
    "n_estimators": TREES,
    "class_weight": "balanced",
    "source_fpr_budget": SOURCE_FPR_BUDGET,
    "threshold_rule": "attack_probability > threshold",
    "threshold_selected_from": "source calibration benign only",
    "feature_sets": feature_sets,
    "candidate_aliases": ALIASES,
    "excluded_from_expansion": [
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets"
    ]
}

with open(OUT / "design.json", "w") as f:
    json.dump(design, f, indent=2)

metric_rows = []
family_rows = []

for arm, features in feature_sets.items():
    print("\n" + "=" * 55)
    print("MODEL:", arm)
    print("=" * 55)

    X = as_numeric_matrix(source, features)

    # Imputation values come exclusively from source training.
    medians = np.nanmedian(X[fit_idx], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)

    X_fit = fill_missing(X[fit_idx], medians)
    X_cal = fill_missing(X[cal_idx], medians)
    X_test = fill_missing(X[test_idx], medians)

    model = RandomForestClassifier(
        n_estimators=TREES,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=2
    )

    print("Training Random Forest...")
    model.fit(X_fit, y[fit_idx])

    cal_scores = model.predict_proba(X_cal)[:, 1]

    # Strict > rule and the empirical 99th percentile ensure
    # calibration FPR is no greater than 1%, including ties.
    cal_benign = cal_scores[y[cal_idx] == 0]
    threshold = float(
        np.quantile(cal_benign, 1 - SOURCE_FPR_BUDGET)
    )

    cal_fpr = float((cal_benign > threshold).mean())
    assert cal_fpr <= SOURCE_FPR_BUDGET + 1e-12

    print("Source threshold:", threshold)
    print("Calibration benign FPR:", round(cal_fpr, 5))

    source_scores = model.predict_proba(X_test)[:, 1]

    for label, cutoff in [
        ("source_calibrated", threshold),
        ("default_0.5", 0.5)
    ]:
        result = calculate_metrics(
            y[test_idx], source_scores, cutoff
        )
        metric_rows.append({
            "arm": arm,
            "dataset": "source_holdout",
            "operating_point": label,
            "calibration_fpr": cal_fpr,
            **result
        })

    del X, X_fit, X_cal, X_test

    external_y = []
    external_scores = []
    external_families = []

    print("Scoring external target...")

    # The model never fits or calibrates on target observations.
    target_names = [
        ALIASES.get(name, name) for name in features
    ]

    for chunk in pd.read_csv(
        TARGET, chunksize=50000, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        classes = chunk["Class"].astype(str).str.strip()
        if not classes.isin(["Attack", "Benign"]).all():
            raise ValueError("Unexpected external class label.")

        target_y = classes.eq("Attack").to_numpy(dtype=np.int8)

        target_x = as_numeric_matrix(chunk, target_names)
        target_x = fill_missing(target_x, medians)

        scores = model.predict_proba(target_x)[:, 1]

        external_y.append(target_y)
        external_scores.append(scores)
        external_families.append(
            chunk["Label"].astype(str).str.strip().to_numpy()
        )

    ty = np.concatenate(external_y)
    ts = np.concatenate(external_scores)
    tf = np.concatenate(external_families)

    for label, cutoff in [
        ("source_calibrated", threshold),
        ("default_0.5", 0.5)
    ]:
        result = calculate_metrics(ty, ts, cutoff)
        metric_rows.append({
            "arm": arm,
            "dataset": "external_2019",
            "operating_point": label,
            "calibration_fpr": cal_fpr,
            **result
        })

        print(
            label,
            "| recall:", round(result["recall"], 4),
            "| FPR:", round(result["fpr"], 4),
            "| AUC:", round(result["roc_auc"], 4)
        )

    # Per-family recall at the source-calibrated threshold.
    for family in sorted(set(tf[ty == 1])):
        mask = (tf == family) & (ty == 1)
        family_rows.append({
            "arm": arm,
            "family": family,
            "n": int(mask.sum()),
            "source_threshold": threshold,
            "recall": float((ts[mask] > threshold).mean()),
            "median_attack_score": float(np.median(ts[mask]))
        })

    pd.DataFrame(metric_rows).to_csv(
        OUT / "metrics.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        OUT / "family_recall.csv", index=False
    )

    del model, ty, ts, tf

metrics = pd.DataFrame(metric_rows)

print("\n=== EXTERNAL RESULTS: SOURCE-CALIBRATED THRESHOLD ===")

selection = metrics[
    (metrics["dataset"] == "external_2019")
    & (metrics["operating_point"] == "source_calibrated")
]

print(
    selection[
        ["arm", "threshold", "recall", "fpr",
         "roc_auc", "average_precision"]
    ].to_string(index=False)
)

print("\nOutputs saved in:", OUT)
print("Target results are exploratory, not confirmatory.")
