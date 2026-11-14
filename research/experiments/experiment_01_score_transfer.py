"""Experiment 01: source-only score and threshold transfer, CIC-IDS2017 -> CIC-DDoS2019.

Run from any directory: python path/to/experiment_01_score_transfer.py
Requires pandas, numpy, scikit-learn.
Uses fixed source train/calibration/test partitions; target labels only for final evaluation.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data/raw/ddos.csv"
TARGET_PATH = ROOT / "data/raw/cicddos2019/cicddos2019_dataset.csv"
OUTPUT = ROOT / "reports/score_diagnostics"
SEED = 42


def report_scores(name, y, p, threshold, strict=False):
    pred = (p > threshold).astype(int) if strict else (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(np.mean(y))
    return {
        "dataset": name,
        "threshold_name": "source_1pct_FPR" if strict else "fixed_0.50",
        "threshold": float(threshold),
        "n": len(y),
        "attack_prevalence": prevalence,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "attack_recall": recall_score(y, pred, zero_division=0),
        "attack_f1": f1_score(y, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "predicted_attack_fraction": float(np.mean(pred)),
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "ap_minus_prevalence": average_precision_score(y, p) - prevalence,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    for path in (SOURCE_PATH, TARGET_PATH):
        if not path.is_file():
            sys.exit(f"Missing CSV: {path}\nCheck your project directory structure.")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading source and target data...")
    source = pd.read_csv(SOURCE_PATH, low_memory=False)
    target = pd.read_csv(TARGET_PATH, low_memory=False)
    source.columns = source.columns.str.strip()
    target.columns = target.columns.str.strip()

    source_labels = source["Label"].astype(str).str.strip()
    target_classes = target["Class"].astype(str).str.strip()
    target_types = target["Label"].astype(str).str.strip()
    if set(source_labels.unique()) != {"BENIGN", "DDoS"}:
        raise ValueError(f"Unexpected source labels: {source_labels.value_counts().to_dict()}")
    if set(target_classes.unique()) != {"Attack", "Benign"}:
        raise ValueError(f"Unexpected target classes: {target_classes.value_counts().to_dict()}")

    excluded = {"Label", "Class", "Protocol", "Unnamed: 0", "Flow ID",
                "Source IP", "Destination IP", "Timestamp"}
    common = sorted((set(source.columns) & set(target.columns)) - excluded)
    numeric = [c for c in common if pd.api.types.is_numeric_dtype(source[c])
               and pd.api.types.is_numeric_dtype(target[c])]
    X_source = source[numeric].replace([np.inf, -np.inf], np.nan)
    X_target = target[numeric].replace([np.inf, -np.inf], np.nan)
    y_source = (source_labels == "DDoS").astype(int).to_numpy()
    y_target = (target_classes == "Attack").astype(int).to_numpy()

    # 70% training, 10% threshold-calibration, 20% held-out internal test.
    indices = np.arange(len(source))
    traincal_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=y_source, random_state=SEED)
    train_idx, cal_idx = train_test_split(
        traincal_idx, test_size=0.125,
        stratify=y_source[traincal_idx], random_state=SEED)

    # Feature choice and all fitted transformations use source TRAIN only.
    keep = X_source.iloc[train_idx].columns[
        X_source.iloc[train_idx].nunique(dropna=True) > 1].tolist()
    X_train = X_source.iloc[train_idx][keep]
    X_cal = X_source.iloc[cal_idx][keep]
    X_test = X_source.iloc[test_idx][keep]
    X_external = X_target[keep]

    print(f"Source train/calibration/test: {len(train_idx):,}/{len(cal_idx):,}/{len(test_idx):,}")
    print(f"External test: {len(y_target):,} | Shared usable features: {len(keep)}")
    print("Training source-only Random Forest...")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=SEED,
            class_weight="balanced", n_jobs=-1)),
    ])
    model.fit(X_train, y_source[train_idx])
    p_cal = model.predict_proba(X_cal)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    p_external = model.predict_proba(X_external)[:, 1]

    # The only threshold selection uses benign source calibration flows.
    # Use > to avoid excessive false positives at tied score quantiles.
    benign_cal = p_cal[y_source[cal_idx] == 0]
    threshold = float(np.quantile(benign_cal, 0.99))
    calibration_fpr = float(np.mean(benign_cal > threshold))
    print(f"Source-only threshold for nominal 1% FPR: {threshold:.6f}")
    print(f"Actual calibration benign FPR: {calibration_fpr:.4%}")

    results = []
    for dataset_name, truth, proba in [
        ("CIC-IDS2017_holdout", y_source[test_idx], p_test),
        ("CIC-DDoS2019_external", y_target, p_external),
    ]:
        results.append(report_scores(dataset_name, truth, proba, 0.5))
        results.append(report_scores(dataset_name, truth, proba, threshold, strict=True))
    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    cols = ["dataset", "threshold_name", "threshold", "accuracy", "balanced_accuracy",
            "precision", "attack_recall", "attack_f1", "false_positive_rate",
            "roc_auc", "average_precision", "ap_minus_prevalence", "TN", "FP", "FN", "TP"]
    print("\n=== SAME MODEL: INTERNAL HOLDOUT VS EXTERNAL ===")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Score quantiles by TRUE class, for diagnostic use only.
    qrows = []
    for dataset_name, truth, proba in [
        ("CIC-IDS2017_holdout", y_source[test_idx], p_test),
        ("CIC-DDoS2019_external", y_target, p_external),
    ]:
        for label, code in [("Benign", 0), ("Attack", 1)]:
            v = proba[truth == code]
            qrows.append({"dataset": dataset_name, "true_class": label,
                          "n": len(v), "mean": float(np.mean(v)),
                          "p10": float(np.quantile(v, 0.10)),
                          "median": float(np.quantile(v, 0.50)),
                          "p90": float(np.quantile(v, 0.90)),
                          "p99": float(np.quantile(v, 0.99))})
    distributions = pd.DataFrame(qrows)
    distributions.to_csv(OUTPUT / "score_quantiles.csv", index=False)
    print("\n=== SCORE DISTRIBUTIONS BY ACTUAL CLASS ===")
    print(distributions.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Attack-type breakdown is an explanatory TARGET analysis, never used for model/threshold selection.
    detail = pd.DataFrame({"attack_type": target_types.to_numpy(),
                           "true_attack": y_target,
                           "attack_probability": p_external,
                           "pred_fixed_0p5": (p_external >= 0.5).astype(int),
                           "pred_source_fpr1": (p_external > threshold).astype(int)})
    groups = detail.groupby("attack_type", dropna=False).agg(
        count=("attack_probability", "size"),
        actual_attack_fraction=("true_attack", "mean"),
        mean_score=("attack_probability", "mean"),
        median_score=("attack_probability", "median"),
        fraction_flagged_at_0p5=("pred_fixed_0p5", "mean"),
        fraction_flagged_at_source_threshold=("pred_source_fpr1", "mean"),
    ).sort_values("count", ascending=False)
    groups.to_csv(OUTPUT / "by_traffic_type.csv")
    print("\n=== TARGET SCORE/DETECTION BY TRAFFIC TYPE ===")
    print(groups.to_string(float_format=lambda x: f"{x:.5f}"))

    # Keep per-flow scores for plots, later paired comparisons, uncertainty intervals.
    detail.to_csv(OUTPUT / "external_per_flow_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"source_row_index": test_idx,
                  "true_attack": y_source[test_idx],
                  "attack_probability": p_test}).to_csv(
                      OUTPUT / "source_holdout_scores.csv.gz", index=False, compression="gzip")

    importance = pd.DataFrame({"feature": keep,
                               "rf_importance": model.named_steps["rf"].feature_importances_})
    importance.sort_values("rf_importance", ascending=False).to_csv(
        OUTPUT / "source_feature_importance.csv", index=False)
    metadata = {"experiment": "01_source_only_score_and_threshold_transfer",
                "seed": SEED, "scikit_learn_version": sklearn.__version__,
                "python_version": sys.version.split()[0],
                "source_csv": str(SOURCE_PATH), "target_csv": str(TARGET_PATH),
                "source_train_n": len(train_idx), "source_calibration_n": len(cal_idx),
                "source_test_n": len(test_idx), "target_n": len(y_target),
                "features": keep, "threshold_selection": "99th percentile of BENIGN SOURCE calibration scores, predict strictly greater",
                "source_threshold": threshold,
                "source_calibration_fpr": calibration_fpr,
                "model": "RandomForestClassifier(n_estimators=100, class_weight=balanced, random_state=42)",
                "note": "Only target columns/schema used before evaluation; target labels used for metrics and post-hoc grouping, never fit or tune."}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved experiment outputs in: {OUTPUT}")
    print("Next: share summary.csv and score_quantiles.csv (or paste terminal output).")


if __name__ == "__main__":
    main()
