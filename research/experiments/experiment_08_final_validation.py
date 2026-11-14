from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/experiment_08"
SOURCE = ROOT / "data/raw/ddos.csv"
TARGET = ROOT / "data/raw/cicids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
OUT.mkdir(parents=True, exist_ok=True)

design = json.loads(
    (ROOT / "reports/experiment_05/design.json").read_text()
)
ARMS = {
    "A_common": design["feature_sets"]["A_common"],
    "B_core": design["feature_sets"]["B_core"]
}
SEEDS = [42, 43, 44]

# Explicit mapping from CIC-IDS2017 names to CIC-IDS2018 names.
ALIASES = {
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s",
    "Min Packet Length": "Pkt Len Min",
    "Max Packet Length": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
    "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Byts/b Avg",
    "Fwd Avg Packets/Bulk": "Fwd Pkts/b Avg",
    "Fwd Avg Bulk Rate": "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Byts/b Avg",
    "Bwd Avg Packets/Bulk": "Bwd Pkts/b Avg",
    "Bwd Avg Bulk Rate": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

all_features = ARMS["A_common"]
assert set(ARMS["B_core"]).issubset(set(all_features))

target_header = pd.read_csv(TARGET, nrows=0).columns.str.strip()
available = set(target_header)

missing = [
    (f, ALIASES.get(f, f))
    for f in all_features
    if ALIASES.get(f, f) not in available
]

if missing:
    print("MISSING FEATURE MAPPINGS:")
    for original, mapped in missing:
        print(original, "=>", mapped)
    raise SystemExit("Schema check failed; no models trained.")

mapped_names = [ALIASES.get(f, f) for f in all_features]
assert len(mapped_names) == len(set(mapped_names))

print("Schema check passed.")
print("Model A features:", len(ARMS["A_common"]))
print("Model B features:", len(ARMS["B_core"]))

source = pd.read_csv(SOURCE, low_memory=False)
source.columns = source.columns.str.strip()

y = source["Label"].astype(str).str.strip().eq("DDoS").to_numpy()
core = ARMS["B_core"]

groups = pd.util.hash_pandas_object(
    source[core].replace([np.inf, -np.inf], np.nan),
    index=False
).to_numpy(dtype=np.uint64)

def group_split(indices, fraction, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=fraction,
        random_state=seed
    )
    a, b = next(
        splitter.split(
            np.zeros(len(indices)),
            groups=groups[indices]
        )
    )
    return indices[a], indices[b]

rows = []
family_rows = []

for seed in SEEDS:
    print("\nSEED:", seed, flush=True)

    idx = np.arange(len(source))
    dev, holdout = group_split(idx, 0.20, seed)
    train, cal = group_split(dev, 0.125, seed + 1000)

    assert not (
        set(groups[train]) & set(groups[cal])
    )
    assert not (
        set(groups[train]) & set(groups[holdout])
    )
    assert not (
        set(groups[cal]) & set(groups[holdout])
    )

    for arm, features in ARMS.items():
        print("Training:", arm, flush=True)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                class_weight="balanced",
                n_jobs=2
            ))
        ])

        X_train = source.iloc[train][features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_cal = source.iloc[cal][features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_holdout = source.iloc[holdout][features].replace(
            [np.inf, -np.inf], np.nan
        )

        model.fit(X_train, y[train])

        cal_scores = model.predict_proba(X_cal)[:, 1]
        benign_cal = cal_scores[~y[cal]]

        threshold = float(np.quantile(benign_cal, 0.99))
        cal_fpr = float((benign_cal > threshold).mean())

        holdout_scores = model.predict_proba(X_holdout)[:, 1]
        holdout_auc = roc_auc_score(y[holdout], holdout_scores)

        # Check reproduction against Experiment 7 before using
        # the new target evaluation.
        earlier = pd.read_csv(
            ROOT / "reports/experiment_07/metrics_by_seed.csv"
        )
        previous = earlier[
            (earlier["seed"] == seed) &
            (earlier["arm"] == arm)
        ]
        if len(previous) != 1:
            raise ValueError("Missing Experiment 7 reference.")

        old = previous.iloc[0]
        if not np.isclose(
            holdout_auc, old["source_auc"], atol=1e-6
        ):
            raise ValueError(
                "Source reproduction mismatch. Stop evaluation."
            )
        if not np.isclose(
            threshold, old["threshold"], atol=1e-9
        ):
            raise ValueError(
                "Threshold reproduction mismatch."
            )

        scores_parts = []
        labels_parts = []

        needed_target = {
            ALIASES.get(f, f) for f in features
        } | {"Label"}

        for chunk in pd.read_csv(
            TARGET,
            usecols=lambda c: c.strip() in needed_target,
            chunksize=50000,
            low_memory=False
        ):
            chunk.columns = chunk.columns.str.strip()

            rename = {
                ALIASES.get(f, f): f for f in features
            }
            X_external = chunk.rename(columns=rename)[features]

            for f in features:
                X_external[f] = pd.to_numeric(
                    X_external[f], errors="coerce"
                )

            X_external = X_external.replace(
                [np.inf, -np.inf], np.nan
            )

            scores_parts.append(
                model.predict_proba(X_external)[:, 1]
            )
            labels_parts.append(
                chunk["Label"].astype(str).str.strip().to_numpy()
            )

        scores = np.concatenate(scores_parts)
        labels = np.concatenate(labels_parts)

        expected = {
            "Benign",
            "DDOS attack-HOIC",
            "DDOS attack-LOIC-UDP"
        }
        if set(labels) != expected:
            raise ValueError(
                "Unexpected 2018 labels: " + str(set(labels))
            )

        attack = labels != "Benign"
        predicted = scores > threshold

        tp = int((predicted & attack).sum())
        fp = int((predicted & ~attack).sum())
        fn = int((~predicted & attack).sum())
        tn = int((~predicted & ~attack).sum())

        result = {
            "seed": seed,
            "arm": arm,
            "threshold": threshold,
            "calibration_fpr": cal_fpr,
            "source_auc": holdout_auc,
            "external_auc": roc_auc_score(attack, scores),
            "external_ap": average_precision_score(attack, scores),
            "external_recall": tp / (tp + fn),
            "external_fpr": fp / (fp + tn),
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        }

        rows.append(result)

        for family in [
            "DDOS attack-HOIC",
            "DDOS attack-LOIC-UDP"
        ]:
            mask = labels == family
            family_rows.append({
                "seed": seed,
                "arm": arm,
                "family": family,
                "n": int(mask.sum()),
                "recall": float(predicted[mask].mean())
            })

        print(
            "AUC:", round(result["external_auc"], 5),
            "| Recall:", round(result["external_recall"], 5),
            "| FPR:", round(result["external_fpr"], 5),
            flush=True
        )

        pd.DataFrame(rows).to_csv(
            OUT / "final_metrics.csv", index=False
        )
        pd.DataFrame(family_rows).to_csv(
            OUT / "final_family_recall.csv", index=False
        )

        del model, scores, labels

result = pd.DataFrame(rows)

print("\n=== FINAL VALIDATION RESULTS ===")
print(result[
    ["seed", "arm", "external_auc",
     "external_ap", "external_recall", "external_fpr"]
].to_string(index=False))

wide = result.pivot(
    index="seed", columns="arm",
    values=["external_auc", "external_recall", "external_fpr"]
)

print("\n=== PAIRED DIFFERENCES: B MINUS A ===")

for metric in [
    "external_auc",
    "external_recall",
    "external_fpr"
]:
    difference = (
        wide[(metric, "B_core")]
        - wide[(metric, "A_common")]
    )
    print(
        metric,
        "| by seed:",
        np.round(difference.to_numpy(), 6),
        "| mean:", round(float(difference.mean()), 6)
    )

print("\n=== ATTACK-FAMILY RECALL ===")
family = pd.DataFrame(family_rows)
print(family.to_string(index=False))

print("\nSaved results in:", OUT)
