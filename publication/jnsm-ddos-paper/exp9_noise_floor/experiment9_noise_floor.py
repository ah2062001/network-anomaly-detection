#!/usr/bin/env python3
"""Experiment 9: fixed-partition RF noise floor and class-matched placebo.

IMPORTANT: The original Experiment 6B script reportedly used the 57 features in
reports/score_diagnostics/metadata.json order.  The mandatory reference check
prevents labeling mismatched experiments as replications. The split code is
reconstructed and still requires reference verification.

Runs no experiments when imported. Never edits existing manuscript or data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

EXPECTED_HASHES = {
    "source": "6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee",
    "target": "97af3b16e45f5309a6492cc16d947a74fdc135267f03756b8cea2eb13f4b2379",
}
REFERENCE = {"original": 0.904330, "deduplicated": 0.827471}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalize_columns(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame.columns = frame.columns.map(lambda x: str(x).strip())
    if not frame.columns.is_unique:
        raise ValueError(f"Duplicate column names after stripping in {label}")
    return frame


def parse_seeds(expr: str) -> list[int]:
    # e.g. 42 or 42,43,44 or 42-61
    result = []
    for segment in expr.split(","):
        segment = segment.strip()
        if "-" in segment:
            lo, hi = map(int, segment.split("-", 1))
            if hi < lo:
                raise ValueError("Seed range must be ascending")
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(segment))
    if not result or len(result) != len(set(result)):
        raise ValueError("Provide nonempty distinct seeds")
    return result


def get_labels(src: pd.DataFrame, tgt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "Label" not in src or "Class" not in tgt:
        raise ValueError("Expected source 'Label' and target 'Class' columns; inspect the CSV schemas")
    src_values = src["Label"].astype(str).str.strip().str.upper()
    target_values = tgt["Class"].astype(str).str.strip().str.upper()
    if not src_values.isin(["BENIGN", "DDOS"]).all():
        raise ValueError(f"Unexpected source labels: {src_values.value_counts().to_dict()}")
    if not target_values.isin(["BENIGN", "ATTACK"]).all():
        raise ValueError(f"Unexpected target classes: {target_values.value_counts().to_dict()}")
    return (src_values.eq("DDOS").astype("int8").to_numpy(),
            target_values.eq("ATTACK").astype("int8").to_numpy())


def choose_features(src: pd.DataFrame, tgt: pd.DataFrame,
                    metadata_path: Path) -> list[str]:
    """Use exact ordered 57-feature metadata from the original Experiment 1/6B.

    RF random-feature selection is sensitive to column order; finding 57 names
    in the raw CSV order does NOT reproduce the original ordered design matrix.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Required original baseline feature metadata not found: {metadata_path}. "
            "Locate reports/score_diagnostics/metadata.json from Experiment 1/6B; "
            "do not guess or regenerate a feature list.")
    with metadata_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    features = meta.get('features')
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError(f"Expected JSON array of feature names at 'features' in {metadata_path}")
    if len(set(features)) != len(features):
        raise ValueError("Original feature metadata contains duplicate feature names")
    source_missing = [c for c in features if c not in src.columns]
    target_missing = [c for c in features if c not in tgt.columns]
    if source_missing or target_missing:
        raise ValueError(f"Features absent from source: {source_missing}; target: {target_missing}")
    if len(features) != 57:
        raise ValueError(f"Expected 57 ordered features, found {len(features)}")
    return features


def to_numeric_matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # Float64 retains medians before sklearn's float32 tree conversion.
    values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    return values


def get_splits(y: np.ndarray, split_npz: Path | None, split_seed: int):
    if split_npz is not None:
        with np.load(split_npz, allow_pickle=False) as arr:
            keys = set(arr.files)
            if {"train_idx", "cal_idx", "test_idx"} - keys:
                raise ValueError("NPZ needs train_idx, cal_idx, test_idx")
            train = np.asarray(arr["train_idx"], dtype=int)
            cal = np.asarray(arr["cal_idx"], dtype=int)
            test = np.asarray(arr["test_idx"], dtype=int)
        provenance = f"exact user-supplied indices from {split_npz}"
    else:
        # Reconstructed 70/10/20 stratified split; original partition provenance
        # has not been independently confirmed.
        idx = np.arange(len(y))
        development, test = train_test_split(idx, test_size=0.20,
                                             stratify=y, random_state=split_seed)
        train, cal = train_test_split(development, test_size=0.125,
                                     stratify=y[development], random_state=split_seed)
        provenance = (f"UNVERIFIED reconstructed 80/20 then 87.5/12.5 split, "
                      f"stratified, seed={split_seed}")
    train, cal, test = map(lambda x: np.asarray(x, dtype=int), (train, cal, test))
    joined = np.concatenate([train, cal, test])
    if len(joined) != len(y) or len(np.unique(joined)) != len(y):
        raise ValueError("Split indices must cover every source row exactly once")
    if joined.min() < 0 or joined.max() >= len(y):
        raise ValueError("Split indices are out of range")
    if len(set(np.unique(y[train]))) < 2 or len(set(np.unique(y[cal]))) < 2 or len(set(np.unique(y[test]))) < 2:
        raise ValueError("Every split needs both labels")
    return train, cal, test, provenance


def class_counts(y: np.ndarray, idx: np.ndarray) -> dict[str, int]:
    return {"benign": int(np.sum(y[idx] == 0)), "attack": int(np.sum(y[idx] == 1))}


def score_model(seed: int, arm: str, selected_idx: np.ndarray,
                X_source: np.ndarray, y_source: np.ndarray,
                X_target: np.ndarray, y_target: np.ndarray,
                cal_idx: np.ndarray, test_idx: np.ndarray, n_jobs: int,
                placebo_rep: int | None = None) -> dict:
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_source[selected_idx])
    X_cal = imputer.transform(X_source[cal_idx])
    X_test = imputer.transform(X_source[test_idx])
    X_ext = imputer.transform(X_target)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=seed, n_jobs=n_jobs)
    clf.fit(X_train, y_source[selected_idx])
    positive_index = int(np.where(clf.classes_ == 1)[0][0])
    cal_scores = clf.predict_proba(X_cal)[:, positive_index]
    source_scores = clf.predict_proba(X_test)[:, positive_index]
    ext_scores = clf.predict_proba(X_ext)[:, positive_index]
    threshold = float(np.quantile(cal_scores[y_source[cal_idx] == 0], 0.99))
    source_pred = source_scores > threshold  # Use the strict score > threshold rule reported in the manuscript.
    ext_pred = ext_scores > threshold
    out = {
        "seed": seed, "arm": arm, "placebo_rep": placebo_rep if placebo_rep is not None else "",
        "train_rows": len(selected_idx), "train_benign": int(np.sum(y_source[selected_idx] == 0)),
        "train_attack": int(np.sum(y_source[selected_idx] == 1)),
        "threshold": threshold,
        "source_auc": float(roc_auc_score(y_source[test_idx], source_scores)),
        "source_recall": float(np.mean(source_pred[y_source[test_idx] == 1])),
        "source_fpr": float(np.mean(source_pred[y_source[test_idx] == 0])),
        "cal_fpr": float(np.mean(cal_scores[y_source[cal_idx] == 0] > threshold)),
        "external_auc": float(roc_auc_score(y_target, ext_scores)),
        "external_recall": float(np.mean(ext_pred[y_target == 1])),
        "external_fpr": float(np.mean(ext_pred[y_target == 0])),
        "external_tp": int(np.sum(ext_pred & (y_target == 1))),
        "external_fp": int(np.sum(ext_pred & (y_target == 0))),
    }
    del clf, X_train, X_cal, X_test, X_ext
    return out


def append_csv(path: Path, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize(results: pd.DataFrame, outdir: Path):
    if results.empty:
        return
    metrics = ["external_auc", "source_auc", "external_recall", "external_fpr"]
    output = results.groupby("arm", sort=True)[metrics].agg(["count", "mean", "std", "min", "max"])
    output.to_csv(outdir / "arm_summary.csv")
    # Use the average placebo AUC across placebo draws within each forest seed.
    pivot = (results.groupby(["seed", "arm"])["external_auc"]
             .mean().unstack("arm"))
    if "original" in pivot and "deduplicated" in pivot:
        pivot["dedup_minus_original"] = pivot["deduplicated"] - pivot["original"]
    if "original" in pivot and "placebo" in pivot:
        pivot["placebo_minus_original"] = pivot["placebo"] - pivot["original"]
    if "deduplicated" in pivot and "placebo" in pivot:
        pivot["dedup_minus_placebo"] = pivot["deduplicated"] - pivot["placebo"]
    pivot.to_csv(outdir / "paired_by_seed.csv")
    for col in ("dedup_minus_original", "placebo_minus_original", "dedup_minus_placebo"):
        if col in pivot:
            vals = pivot[col].dropna().to_numpy()
            if len(vals):
                print(f"{col}: n={len(vals)} mean={vals.mean():+.6f}; "
                      f"seed-range=[{vals.min():+.6f}, {vals.max():+.6f}]; "
                      f"seed-quantiles(2.5%,97.5%)="
                      f"[{np.percentile(vals,2.5):+.6f}, {np.percentile(vals,97.5):+.6f}]")
    print("NOTE: These are seed-variation summaries on fixed files, NOT confidence "
          "intervals for independent network deployments.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/ddos.csv"))
    p.add_argument("--target", type=Path, default=Path("data/raw/cicddos2019/cicddos2019_dataset.csv"))
    p.add_argument("--out", type=Path, default=Path("research/results/experiment9_noise_floor"))
    p.add_argument("--seeds", default="42", help="42 or 42,43,44 or 42-61")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--features-json", type=Path, default=Path("reports/score_diagnostics/metadata.json"),
                   help="Original ordered 57-feature metadata from Experiment 1/6B")
    p.add_argument("--split-npz", type=Path, default=None,
                   help="EXACT Experiment 6B train_idx/cal_idx/test_idx saved with numpy.savez")
    p.add_argument("--placebo-reps", type=int, default=1,
                   help="independently sampled fixed placebo subsets (e.g. 3)")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--no-hash-check", action="store_true", help="ONLY for diagnostics on alternative CSVs")
    p.add_argument("--skip-reference-check", action="store_true",
                   help="ONLY for diagnostics; output cannot claim 6B reproduction")
    p.add_argument("--expected-train-rows", type=int, default=158021)
    p.add_argument("--expected-duplicates", type=int, default=1532)
    p.add_argument("--expected-features", type=int, default=57)
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)
    if args.placebo_reps < 1:
        p.error("--placebo-reps must be >= 1")
    if not args.skip_reference_check and 42 not in seeds:
        p.error("Include 42 in --seeds for mandatory baseline reproduction, or use --skip-reference-check for diagnostics only")
    if not args.source.exists() or not args.target.exists():
        p.error("CSV missing: run from repository root and check --source/--target paths")
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "per_run.csv"
    if result_path.exists():
        p.error(f"Output {result_path} already exists. Use a NEW --out directory to prevent mixed protocols.")

    hashes = {"source": sha256(args.source), "target": sha256(args.target)}
    if not args.no_hash_check:
        for dataset, expected in EXPECTED_HASHES.items():
            if hashes[dataset] != expected:
                raise RuntimeError(f"{dataset} SHA-256 does not match manuscript. "
                                   f"Expected {expected}, found {hashes[dataset]}")
    print("Reading CSVs and building 57-feature design matrix...", flush=True)
    src = normalize_columns(pd.read_csv(args.source, low_memory=False), "source")
    target = normalize_columns(pd.read_csv(args.target, low_memory=False), "target")
    ysrc, ytgt = get_labels(src, target)
    features = choose_features(src, target, args.features_json)
    print(f"Using original ordered features from {args.features_json} (first five: {features[:5]})", flush=True)
    if len(features) != args.expected_features:
        raise RuntimeError(f"Found {len(features)} features; expected {args.expected_features}. "
                           "Must obtain the exact Experiment 6B feature list before proceeding. "
                           f"Candidate list: {features}")
    train, cal, holdout, split_method = get_splits(ysrc, args.split_npz, args.split_seed)
    if len(train) != args.expected_train_rows:
        raise RuntimeError(f"Training partition has {len(train)} rows, expected {args.expected_train_rows}. "
                           "Use exact Experiment 6B indices (--split-npz).")
    # Full original source row equivalence, not just model-feature equivalence.
    train_df = src.iloc[train]
    duplicates = train_df.duplicated(keep="first").to_numpy()
    duplicate_count = int(duplicates.sum())
    if duplicate_count != args.expected_duplicates:
        raise RuntimeError(f"Found {duplicate_count} duplicate extras in training, "
                           f"expected {args.expected_duplicates}. "
                           "Check exact split and the original 6B duplicate definition.")
    dedup_idx = train[~duplicates]
    # The placebo removes *only* singleton (never duplicated) full rows.
    singleton_idx = train[~train_df.duplicated(keep=False).to_numpy()]
    removed_idx = train[duplicates]
    remove_counts = class_counts(ysrc, removed_idx)
    print(f"Source {len(src):,}, target {len(target):,}, features {len(features)}")
    print(f"Train {len(train):,}; calibration {len(cal):,}; holdout {len(holdout):,}")
    print(f"Training duplicates removed {duplicate_count:,}; class mix {remove_counts}")
    print("SPLIT METHOD:", split_method)

    # Predefine and preserve placebo sets independently of fitted-forest seeds.
    rng = np.random.default_rng(92026)
    placebos = []
    for rep in range(args.placebo_reps):
        removed = []
        for lab, count in [(0, remove_counts["benign"]), (1, remove_counts["attack"])]:
            available = singleton_idx[ysrc[singleton_idx] == lab]
            if len(available) < count:
                raise RuntimeError(f"Not enough singleton rows of class {lab} for placebo")
            removed.extend(rng.choice(available, size=count, replace=False).tolist())
        selected = train[~np.isin(train, np.asarray(removed, dtype=int))]
        if len(selected) != len(dedup_idx) or class_counts(ysrc, selected) != class_counts(ysrc, dedup_idx):
            raise AssertionError("Placebo row counts or class balance differ from deduplicated arm")
        placebos.append((rep, selected, np.asarray(removed, dtype=int)))

    manifest = {
        "purpose": "fixed-split 6B reproduction + paired RF seed variation + class-matched placebo",
        "important_warning": "Default split is an UNVERIFIED reconstruction unless exact --split-npz supplied; baseline must reproduce.",
        "source_path": str(args.source), "target_path": str(args.target),
        "data_sha256": hashes, "features": features,
        "features_json_path": str(args.features_json),
        "features_sha256": hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest(),
        "split_method": split_method, "split_seed": args.split_seed,
        "seeds": seeds, "n_jobs": args.n_jobs, "placebo_reps": args.placebo_reps,
        "train_rows_original": len(train), "train_rows_dedup": len(dedup_idx),
        "removed_by_class": remove_counts,
        "source_split_class_counts": {
            "train": class_counts(ysrc, train),
            "calibration": class_counts(ysrc, cal),
            "holdout": class_counts(ysrc, holdout)},
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "model": {"class": "RandomForestClassifier", "n_estimators": 100,
                  "class_weight": "balanced", "random_state": "varied", "n_jobs": args.n_jobs},
        "threshold": "strict score > source benign calibration 99th percentile",
        "reference_check_required": not args.skip_reference_check,
    }
    # Save row indices to identify the fixed samples across runs.
    np.savez_compressed(args.out / "partitions.npz", train_idx=train, cal_idx=cal, test_idx=holdout)
    for rep, _, removed in placebos:
        np.save(args.out / f"placebo_removed_indices_rep{rep}.npy", removed)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Xsrc = to_numeric_matrix(src, features)
    Xtgt = to_numeric_matrix(target, features)
    del src, target, train_df
    done = []
    for seed in seeds:
        runs = [("original", None, train), ("deduplicated", None, dedup_idx)]
        runs += [("placebo", rep, selected) for rep, selected, _ in placebos]
        for arm, rep, idx in runs:
            print(f"Fitting seed={seed} arm={arm} placebo_rep={rep}", flush=True)
            result = score_model(seed, arm, idx, Xsrc, ysrc, Xtgt, ytgt,
                                 cal, holdout, args.n_jobs, placebo_rep=rep)
            if not args.skip_reference_check and seed == 42 and arm in REFERENCE:
                expected = REFERENCE[arm]
                if abs(result["external_auc"] - expected) > 0.000005:
                    raise RuntimeError(
                        f"REFERENCE CHECK FAILED for seed 42 {arm}: "
                        f"expected external AUC {expected}, got {result['external_auc']:.6f}. "
                        "STOP. Do not interpret new runs as Experiment 6B replication; "
                        "compare with original 6B script and supply exact indices/feature list.")
                print(f"REFERENCE CHECK PASS: {arm} external AUC {result['external_auc']:.6f}")
            append_csv(result_path, result)
            done.append(result)
            print(f"  source_AUC={result['source_auc']:.6f}; "
                  f"external_AUC={result['external_auc']:.6f}; "
                  f"external_recall={result['external_recall']:.4%}; "
                  f"external_FPR={result['external_fpr']:.4%}", flush=True)
        summarize(pd.DataFrame(done), args.out)
    print("COMPLETE. Outputs in", args.out)


if __name__ == "__main__":
    main()
