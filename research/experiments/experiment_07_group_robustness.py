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
OUT = ROOT / "reports/experiment_07"
OUT.mkdir(parents=True, exist_ok=True)

design = json.loads(
    (ROOT / "reports/experiment_05/design.json").read_text()
)

FEATURE_SETS = {
    "A_common": design["feature_sets"]["A_common"],
    "B_core": design["feature_sets"]["B_core"],
}

SEEDS = [42, 43, 44]
SOURCE_FPR = 0.01

source_file = ROOT / "data/raw/ddos.csv"
target_file = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"

needed = set(FEATURE_SETS["A_common"]) | {"Label"}

source = pd.read_csv(
    source_file,
    usecols=lambda c: c.strip() in needed,
    low_memory=False
)
source.columns = source.columns.str.strip()

y = source["Label"].astype(str).str.strip().eq("DDoS").to_numpy()

assert set(source["Label"].astype(str).str.strip()) == {
    "BENIGN", "DDoS"
}

# Group on the smaller feature representation so identical
# 37-feature vectors cannot cross any source split.
core = FEATURE_SETS["B_core"]

group_frame = source[core].replace(
    [np.inf, -np.inf], np.nan
)

groups = pd.util.hash_pandas_object(
    group_frame, index=False
).to_numpy(dtype=np.uint64)

print("Source rows:", len(source), flush=True)
print("Unique core-feature groups:", len(np.unique(groups)),
      flush=True)

results = []
family_results = []

def split_once(indices, group_ids, fraction, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=fraction,
        random_state=seed
    )
    first, second = next(
        splitter.split(
            np.zeros(len(indices)),
            groups=group_ids
        )
    )
    return indices[first], indices[second]

for seed in SEEDS:
    print("\n" + "=" * 55, flush=True)
    print("SEED:", seed, flush=True)

    indices = np.arange(len(source))

    dev, holdout = split_once(
        indices, groups, 0.20, seed
    )

    train, calibration = split_once(
        dev, groups[dev], 0.125, seed + 1000
    )

    # Verify group disjointness across all partitions.
    g_train = set(groups[train].tolist())
    g_cal = set(groups[calibration].tolist())
    g_test = set(groups[holdout].tolist())

    assert not (g_train & g_cal)
    assert not (g_train & g_test)
    assert not (g_cal & g_test)

    print(
        "Train / calibration / holdout:",
        len(train), len(calibration), len(holdout),
        flush=True
    )

    for partition in [train, calibration, holdout]:
        assert y[partition].any()
        assert (~y[partition]).any()

    for arm, features in FEATURE_SETS.items():
        print("\nTraining:", arm, flush=True)

        X_train = source.iloc[train][features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_cal = source.iloc[calibration][features].replace(
            [np.inf, -np.inf], np.nan
        )
        X_test = source.iloc[holdout][features].replace(
            [np.inf, -np.inf], np.nan
        )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                class_weight="balanced",
                n_jobs=2
            ))
        ])

        model.fit(X_train, y[train])

        cal_scores = model.predict_proba(X_cal)[:, 1]
        benign_cal = cal_scores[~y[calibration]]

        threshold = float(
            np.quantile(benign_cal, 1 - SOURCE_FPR)
        )

        cal_fpr = float((benign_cal > threshold).mean())
        assert cal_fpr <= SOURCE_FPR + 1e-12

        source_scores = model.predict_proba(X_test)[:, 1]

        source_pred = source_scores > threshold

        ext_y_parts = []
        ext_score_parts = []
        ext_family_parts = []

        needed_target = set(features) | {"Class", "Label"}

        for chunk in pd.read_csv(
            target_file,
            usecols=lambda c: c.strip() in needed_target,
            chunksize=50000,
            low_memory=False
        ):
            chunk.columns = chunk.columns.str.strip()

            labels = chunk["Class"].astype(str).str.strip()
            assert labels.isin(["Attack", "Benign"]).all()

            external_x = chunk[features].replace(
                [np.inf, -np.inf], np.nan
            )

            ext_y_parts.append(
                labels.eq("Attack").to_numpy()
            )
            ext_score_parts.append(
                model.predict_proba(external_x)[:, 1]
            )
            ext_family_parts.append(
                chunk["Label"].astype(str).str.strip().to_numpy()
            )

        yt = np.concatenate(ext_y_parts)
        scores = np.concatenate(ext_score_parts)
        families = np.concatenate(ext_family_parts)

        predicted = scores > threshold

        row = {
            "seed": seed,
            "arm": arm,
            "train_rows": len(train),
            "cal_rows": len(calibration),
            "holdout_rows": len(holdout),
            "threshold": threshold,
            "calibration_fpr": cal_fpr,
            "source_auc": roc_auc_score(
                y[holdout], source_scores
            ),
            "source_recall": float(
                source_pred[y[holdout]].mean()
            ),
            "source_fpr": float(
                source_pred[~y[holdout]].mean()
            ),
            "external_auc": roc_auc_score(yt, scores),
            "external_ap": average_precision_score(yt, scores),
            "external_recall": float(predicted[yt].mean()),
            "external_fpr": float(predicted[~yt].mean())
        }

        results.append(row)

        print(
            "Source AUC:", round(row["source_auc"], 5),
            "| External AUC:", round(row["external_auc"], 5),
            "| External recall:", round(row["external_recall"], 5),
            "| External FPR:", round(row["external_fpr"], 5),
            flush=True
        )

        for family in sorted(set(families[yt])):
            mask = yt & (families == family)
            family_results.append({
                "seed": seed,
                "arm": arm,
                "family": family,
                "n": int(mask.sum()),
                "recall": float(predicted[mask].mean())
            })

        pd.DataFrame(results).to_csv(
            OUT / "metrics_by_seed.csv", index=False
        )
        pd.DataFrame(family_results).to_csv(
            OUT / "family_recall_by_seed.csv", index=False
        )

        del model

result = pd.DataFrame(results)

print("\n=== RESULTS BY SEED ===")
print(result[
    ["seed", "arm", "source_auc", "source_fpr",
     "external_auc", "external_recall", "external_fpr"]
].to_string(index=False))

wide = result.pivot(
    index="seed",
    columns="arm",
    values=["external_auc", "external_recall", "external_fpr"]
)

print("\n=== PAIRED DIFFERENCES: B MINUS A ===")

for metric in ["external_auc", "external_recall", "external_fpr"]:
    diff = wide[(metric, "B_core")] - wide[(metric, "A_common")]
    print(
        metric,
        "| by seed:", np.round(diff.to_numpy(), 6),
        "| mean:", round(float(diff.mean()), 6)
    )

print("\nThese results are exploratory.")
print("CIC-DDoS2019 has already informed representation design.")
print("Reports saved to:", OUT)
