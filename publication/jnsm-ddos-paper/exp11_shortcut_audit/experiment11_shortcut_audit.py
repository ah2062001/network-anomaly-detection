#!/usr/bin/env python3
"""Experiment 11 follow-up: single-feature shortcut screen on existing grouped partitions.

Descriptive diagnostic only. It does NOT re-fit the original Random Forest.
Select the direction of each feature using the saved TRAIN subset only,
and evaluate that fixed direction on the saved TEST subset. High AUC is
evidence of strong predictive association, not proof of leakage or valid labels.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "exp11_within_target"))
import experiment11_within_target as exp11

def take_saved_rows(csv_path, dataset, mapping, features, train_idx, test_idx, chunk_size):
    label_col = "Class" if dataset == "2019" else "Label"
    wanted = set(mapping.values()) | {label_col}
    ordered = [mapping[f] for f in features]
    sorted_train, sorted_test = np.sort(train_idx), np.sort(test_idx)
    xtr, ytr, xte, yte = [], [], [], []
    start = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False,
                             usecols=lambda c: str(c).strip() in wanted):
        chunk.columns = [str(s).strip() for s in chunk.columns]
        if set(chunk.columns) != wanted:
            raise ValueError(f"Expected fields missing: {wanted-set(chunk.columns)}")
        stop = start + len(chunk)
        for indices, xdst, ydst in ((sorted_train, xtr, ytr), (sorted_test, xte, yte)):
            lo = np.searchsorted(indices, start)
            hi = np.searchsorted(indices, stop)
            if hi > lo:
                selected = chunk.iloc[indices[lo:hi] - start]
                arr = np.column_stack([
                    pd.to_numeric(selected[f], errors="coerce").to_numpy(dtype=np.float32)
                    for f in ordered])
                arr[~np.isfinite(arr)] = np.nan
                xdst.append(arr)
                ydst.append(exp11.parse_y(selected[label_col], dataset))
        start = stop
    if not xtr or not xte:
        raise RuntimeError("No sampled rows matched indices")
    tr = np.vstack(xtr)
    te = np.vstack(xte)
    yt = np.concatenate(ytr)
    ye = np.concatenate(yte)
    if len(tr) != len(train_idx) or len(te) != len(test_idx):
        raise RuntimeError("Index or row count mismatch")
    return tr, yt, te, ye, start

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["2018","2019"], default="2018")
    p.add_argument("--results-dir", type=Path, default=Path("research/results/experiment11_2018"))
    p.add_argument("--features-json", type=Path, default=Path("reports/score_diagnostics/metadata.json"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--path", type=Path)
    p.add_argument("--out", type=Path, default=Path("research/results/experiment11_shortcut_audit_2018"))
    p.add_argument("--chunk-size", type=int, default=50000)
    args=p.parse_args()
    base = args.results_dir / f"target_{args.dataset}"
    with (base / "run_metadata.json").open() as f:
        meta=json.load(f)
    if meta.get("dataset") != args.dataset:
        raise RuntimeError("Saved metadata dataset differs from --dataset")
    features = exp11.read_features(args.features_json)
    if features != meta["ordered_features"]:
        raise RuntimeError("Feature order mismatch: STOP")
    path = args.path if args.path else Path(meta["csv_path"])
    if not path.is_file():
        path = exp11.choose_dataset_path(args.raw_root, args.dataset, None)
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping = meta["resolved_mapping_2017_names_to_target_names"]
    with np.load(base/"partitions.npz") as splits:
        tridx = np.asarray(splits["used_train_idx"], dtype=np.int64)
        teidx = np.asarray(splits["used_test_idx"], dtype=np.int64)
    if np.intersect1d(tridx,teidx).size:
        raise RuntimeError("Saved train and test index overlap")
    print(f"Loading saved train and test samples for {args.dataset} without retraining...", flush=True)
    tr, yt, te, ye, n_rows = take_saved_rows(path, args.dataset, mapping, features, tridx, teidx, args.chunk_size)
    if n_rows != exp11.EXPECTED[args.dataset]["rows"]:
        raise RuntimeError(f"Row count mismatch: {n_rows}")
    if tr.shape != (len(tridx),57) or te.shape != (len(teidx),57):
        raise RuntimeError("Unexpected sample shape")
    counts = meta["used_partition_counts"]
    for label, observed, expected in [
        ("train attacks", int(yt.sum()), counts["train"]["attack"]),
        ("test attacks", int(ye.sum()), counts["test"]["attack"])]:
        if observed != expected:
            raise RuntimeError(f"{label}: {observed} vs {expected}. Stop.")
    rows=[]
    for j, name in enumerate(features):
        train_missing = float(np.mean(np.isnan(tr[:, j])))
        test_missing = float(np.mean(np.isnan(te[:, j])))
        valid = tr[:, j][np.isfinite(tr[:,j])]
        if valid.size == 0:
            median=0.0
        else:
            median=float(np.median(valid))
        vtr=np.nan_to_num(tr[:,j],nan=median)
        vte=np.nan_to_num(te[:,j],nan=median)
        # Replace only missing/non-finite (handled above; 'nan_to_num' is safe).
        auc_tr=float(roc_auc_score(yt,vtr))
        sign=1 if auc_tr >= 0.5 else -1
        auc_te=float(roc_auc_score(ye,sign*vte))
        rows.append({
            "feature":name, "input_column":mapping[name],
            "direction_selected_on_train":"higher_is_attack" if sign==1 else "lower_is_attack",
            "train_single_feature_auc":max(auc_tr,1-auc_tr),
            "test_single_feature_auc_fixed_direction":auc_te,
            "train_missing_fraction":train_missing,"test_missing_fraction":test_missing,
            "train_unique_values":int(np.unique(vtr).size),
            "test_unique_values":int(np.unique(vte).size),
            "train_median":median
        })
    df=pd.DataFrame(rows).sort_values("train_single_feature_auc",ascending=False)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out/"single_feature_auc.csv",index=False)
    top = df.head(12)[["feature","input_column","train_single_feature_auc","test_single_feature_auc_fixed_direction","train_unique_values"]]
    print("\nTop 12 features by TRAIN AUC; direction applied unchanged on test:")
    print(top.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
    print("\nSaved:",args.out/"single_feature_auc.csv")
    print("CAUTION: This screen assesses association and possible benchmark shortcuts.")
    print("It does NOT prove label leakage, connection independence, or semantic correctness.")
if __name__ == "__main__":
    main()
