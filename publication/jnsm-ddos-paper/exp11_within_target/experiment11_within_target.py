#!/usr/bin/env python3
"""Experiment 11: Leakage-conscious *within-target* DDoS classification sanity checks.

Build one independent RandomForest model for each target (redistributed CIC-DDoS2019
and CSE-CIC-IDS2018 21-Feb CSV) using identical source-established ordered 57 features.
No CIC-IDS2017 data are used to fit these target-local models.

GroupShuffleSplit keeps identical feature vectors disjoint across train/cal/test.
This prevents exact modeled-vector overlap, NOT connection/host/time overlap.
Train/cal/test subsampling occurs *after* grouping; it does not cross partitions.
The 100-tree Random Forest is capacity-limited to protect laptop memory; do NOT
interpret target-local scores as the same model as Experiment 6B.

This script does not modify source data, earlier outputs, or manuscripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

EXPECTED = {
    "2019": {"rows": 431371,
             "sha256": "97af3b16e45f5309a6492cc16d947a74fdc135267f03756b8cea2eb13f4b2379"},
    "2018": {"rows": 1048575,
             "sha256": "a5f4a1c2689e0aa6566c03a58466de9c407c0be0cbd3cc69306544026611be04"},
}
FILENAME_2018 = "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"

# Fixed, explicit dictionary of candidate CICFlowMeter2018 aliases. No fuzzy matches.
# This is a *schema alignment* only, NOT verification of cross-dataset semantics.
CANDIDATE_2018_ALIASES = {
    "Avg Bwd Segment Size": "Bwd Seg Size Avg",
    "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Bwd Header Length": "Bwd Header Len",
    "Bwd IAT Total": "Bwd IAT Tot",
    "Bwd Packet Length Max": "Bwd Pkt Len Max",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean",
    "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Bwd Packets/s": "Bwd Pkts/s",
    "ECE Flag Count": "ECE Flag Cnt",
    "FIN Flag Count": "FIN Flag Cnt",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Fwd Header Length": "Fwd Header Len",
    "Fwd IAT Total": "Fwd IAT Tot",
    "Fwd Packet Length Max": "Fwd Pkt Len Max",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean",
    "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Fwd Packets/s": "Fwd Pkts/s",
    "Packet Length Mean": "Pkt Len Mean",
    "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "PSH Flag Count": "PSH Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "SYN Flag Count": "SYN Flag Cnt",
    "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts",
    "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Fwd Packets": "Subflow Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "URG Flag Count": "URG Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
}


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def columns_of(path: Path):
    cols = [str(s).strip() for s in pd.read_csv(path, nrows=0).columns]
    if len(set(cols)) != len(cols):
        raise ValueError(f"Ambiguous duplicate header names after stripping: {path}")
    return cols


def read_features(path: Path):
    feats = json.loads(path.read_text(encoding="utf-8")).get("features", None)
    if not isinstance(feats, list) or len(feats) != 57 or len(set(feats)) != 57:
        raise ValueError("Original reports/score_diagnostics/metadata.json needs 57 distinct ordered features")
    if not all(isinstance(f, str) and f.strip() for f in feats):
        raise ValueError("Feature names must be nonblank strings")
    if any(f in {"Class", "Label", "Protocol", "Timestamp", "Dst Port"} for f in feats):
        raise ValueError("Unexpected label or metadata in 57-feature model set")
    return feats


def resolve_columns(features, cols, dataset):
    available = set(cols)
    result, missing = {}, []
    for feat in features:
        if feat in available:
            result[feat] = feat
        elif dataset == "2018" and CANDIDATE_2018_ALIASES.get(feat) in available:
            result[feat] = CANDIDATE_2018_ALIASES[feat]
        else:
            missing.append((feat, CANDIDATE_2018_ALIASES.get(feat, "")))
    if len(set(result.values())) != len(result):
        raise ValueError(f"Multiple features map to the same target field: {result}")
    return result, missing


def parse_y(series, dataset):
    values = series.astype("string").str.strip().str.upper()
    if dataset == "2019":
        allowed = {"ATTACK", "BENIGN"}
        y = values.eq("ATTACK").to_numpy(dtype=np.int8)
    else:
        allowed = {"BENIGN", "DDOS ATTACK-HOIC", "DDOS ATTACK-LOIC-UDP"}
        y = values.ne("BENIGN").to_numpy(dtype=np.int8)
    bad = values[~values.isin(allowed)]
    if len(bad):
        raise ValueError(f"{dataset} unexpected labels: {bad.value_counts(dropna=False).head(12).to_dict()}")
    return y


def load_matrix(path, dataset, features, mapping, chunk_size=80000):
    # Read only required 57 fields + label. Coerce and store float32; this is
    # sklearn tree's numerical representation, reducing the memory footprint.
    lab = "Class" if dataset == "2019" else "Label"
    wanted = set(mapping.values()) | {lab, "Label"}
    arrays, ys, family_parts = [], [], []
    family_ids = {}
    count = 0
    print(f"{dataset}: reading {path.name} in chunks (57 numeric features + {lab})", flush=True)
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunk_size,
                             usecols=lambda name: str(name).strip() in wanted):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if set(chunk.columns) != wanted:
            raise ValueError(f"Read-column mismatch for {dataset}: {wanted - set(chunk.columns)}")
        ordered = [mapping[f] for f in features]
        # pd.to_numeric may coerce strange strings to missing; count checked below.
        x = np.column_stack([pd.to_numeric(chunk[name], errors="coerce").to_numpy(
            dtype=np.float32) for name in ordered])
        x[~np.isfinite(x)] = np.nan
        # Ensure -0.0 and +0.0 have identical group keys and model values.
        x[x == 0] = np.float32(0)
        y = parse_y(chunk[lab], dataset)
        # Family labels are diagnostics only, never model inputs or grouping keys.
        fam = chunk["Label"].astype("string").str.strip().fillna("(missing)")
        for label in pd.unique(fam):
            if str(label) not in family_ids:
                family_ids[str(label)] = len(family_ids)
        family_parts.append(fam.map(family_ids).to_numpy(dtype=np.int16))
        arrays.append(x)
        ys.append(y)
        count += len(y)
        if count % 320000 < chunk_size:
            print(f"  read {count:,} rows", flush=True)
    x = np.concatenate(arrays, axis=0)
    y = np.concatenate(ys, axis=0)
    family_codes = np.concatenate(family_parts, axis=0)
    del arrays, ys, family_parts
    if len(y) != count or x.shape != (count, len(features)):
        raise RuntimeError("Matrix row count mismatch")
    print(f"{dataset}: matrix {x.shape} float32; attacks={int(y.sum()):,}; benign={int((y==0).sum()):,}", flush=True)
    return x, y, family_codes, {v:k for k,v in family_ids.items()}


def hash_groups(x):
    # float32 matches the downstream Random Forest representation. NaNs are
    # normalized to one sentinel for hashing; hash is *not* used as a model input.
    # 64-bit pandas hashes have very low (not mathematically zero) collision risk.
    group_keys = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy(dtype=np.uint64)
    return group_keys


def group_audit(groups, y):
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    p = np.bincount(inverse, weights=y, minlength=len(unique)).astype(np.int64)
    conflicted = (p > 0) & (p < sizes)
    stats = {
        "unique_feature_vector_groups": int(len(unique)),
        "duplicate_group_rows": int(len(groups) - len(unique)),
        "mixed_label_group_count": int(conflicted.sum()),
        "mixed_label_group_rows": int(sizes[conflicted].sum()),
        "largest_group_size": int(sizes.max()),
    }
    return stats


def split_groups(y, groups, seed):
    all_idx = np.arange(len(y), dtype=np.int64)
    outer = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    dev_rel, test_rel = next(outer.split(all_idx, y, groups))
    dev = all_idx[dev_rel]
    test = all_idx[test_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=.125, random_state=seed + 11)
    train_rel, cal_rel = next(inner.split(dev, y[dev], groups[dev]))
    train, cal = dev[train_rel], dev[cal_rel]
    for name, inds in (("train", train), ("cal", cal), ("test", test)):
        if len(np.unique(y[inds])) != 2:
            raise RuntimeError(f"{name} contains only one class after grouped split. Try another --split-seed; do not silently resplit.")
    assert len(np.unique(np.concatenate([train, cal, test]))) == len(y)
    group_sets = [np.unique(groups[inds]) for inds in (train, cal, test)]
    for i in range(3):
        for j in range(i + 1, 3):
            if np.intersect1d(group_sets[i], group_sets[j]).size:
                raise RuntimeError("Exact vector group leaked across splits")
    return train, cal, test


def cap_stratified(indices, y, cap, seed):
    if cap <= 0 or len(indices) <= cap:
        return indices
    # Sampling happens *within* one already-grouped partition. It cannot
    # introduce cross-partition overlap, although duplicate vectors can remain.
    selected, unused = train_test_split(indices, train_size=cap,
                                        stratify=y[indices], random_state=seed)
    return np.asarray(selected, dtype=np.int64)


def metrics(y, s, threshold):
    pred = s > threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "threshold": float(threshold),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "attack_recall": float(tp / (tp + fn)) if tp + fn else None,
        "benign_fpr": float(fp / (fp + tn)) if fp + tn else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "attack_prevalence": float(np.mean(y)),
    }


def choose_dataset_path(raw_root, tag, supplied):
    if supplied is not None:
        path = Path(supplied)
        if not path.is_file():
            raise FileNotFoundError(f"Specified {tag} file does not exist: {path}")
        return path
    if tag == "2019":
        path = raw_root / "cicddos2019" / "cicddos2019_dataset.csv"
        if path.is_file():
            return path
        filename = "cicddos2019_dataset.csv"
    else:
        filename = FILENAME_2018
    matches = list(raw_root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(f"Cannot find {tag} file '{filename}' below {raw_root}; specify --path-{tag}")
    raise RuntimeError(f"Multiple candidate {tag} CSVs found: {matches}; specify --path-{tag}")


def run_one(tag, path, features, mapping, args, outdir):
    started = time.time()
    if not args.skip_hash_check:
        file_hash = digest(path)
        if file_hash != EXPECTED[tag]["sha256"]:
            raise RuntimeError(f"{tag} SHA-256 mismatch; expected {EXPECTED[tag]['sha256']}, observed {file_hash}. STOP.")
        print(f"{tag} SHA-256 verified: {file_hash}", flush=True)
    else:
        file_hash = "SKIPPED FOR TESTING — NOT FOR PUBLICATION"
    x, y, family_codes, family_names = load_matrix(path, tag, features, mapping, chunk_size=args.chunk_size)
    if not args.skip_hash_check and len(y) != EXPECTED[tag]["rows"]:
        raise RuntimeError(f"{tag} row count unexpected: {len(y)} vs {EXPECTED[tag]['rows']}")
    groups = hash_groups(x)
    audit = group_audit(groups, y)
    print(f"{tag} group audit: {audit}", flush=True)
    tr, ca, te = split_groups(y, groups, args.split_seed)
    partition_counts = {name:{"rows":len(idx), "benign":int((y[idx]==0).sum()),
                              "attack":int(y[idx].sum()), "unique_groups":int(np.unique(groups[idx]).size)}
                        for name,idx in (("train",tr),("cal",ca),("test",te))}
    print(f"{tag} grouped partitions: {partition_counts}", flush=True)
    tr_used = cap_stratified(tr,y,args.max_train,args.split_seed+101)
    ca_used = cap_stratified(ca,y,args.max_cal,args.split_seed+102)
    te_used = cap_stratified(te,y,args.max_test,args.split_seed+103)
    used_counts = {name:{"rows":len(idx),"benign":int((y[idx]==0).sum()),"attack":int(y[idx].sum()),
                         "unique_groups":int(np.unique(groups[idx]).size)}
                   for name,idx in (("train",tr_used),("cal",ca_used),("test",te_used))}
    print(f"{tag} evaluated subsets after caps: {used_counts}", flush=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(x[tr_used])
    x_cal = imputer.transform(x[ca_used])
    x_test = imputer.transform(x[te_used])
    del x
    if x_train.shape[1] != 57 or np.any(np.isnan(x_train)) or np.any(np.isnan(x_cal)) or np.any(np.isnan(x_test)):
        raise RuntimeError("Imputation produced missing values or a different feature count")
    rf = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced",
                                min_samples_leaf=args.min_leaf, max_leaf_nodes=args.max_leaf_nodes,
                                random_state=args.model_seed, n_jobs=args.n_jobs)
    print(f"{tag} fitting {args.trees} trees on {len(tr_used):,} rows (max_leaf_nodes={args.max_leaf_nodes})...", flush=True)
    rf.fit(x_train, y[tr_used]); del x_train
    cal_scores = rf.predict_proba(x_cal)[:,1]
    test_scores = rf.predict_proba(x_test)[:,1]
    threshold = float(np.quantile(cal_scores[y[ca_used]==0], 0.99))
    result = metrics(y[te_used], test_scores, threshold)
    result["calibration_fpr"] = float((cal_scores[y[ca_used] == 0] > threshold).mean())
    # A second threshold (0.5) is a descriptive reference, not tuned on test.
    default = metrics(y[te_used], test_scores, 0.5)
    result["recall_at_0_5"] = default["attack_recall"]
    result["fpr_at_0_5"] = default["benign_fpr"]
    # Group-weighted AUC treats each observed modeled-vector group as one unit
    # of total weight, instead of counting every duplicate row equally.
    te_groups = groups[te_used]
    uniq, inv, counts = np.unique(te_groups, return_inverse=True, return_counts=True)
    test_weights = 1.0 / counts[inv]
    result["group_equal_weight_auc"] = float(roc_auc_score(y[te_used], test_scores, sample_weight=test_weights))
    result["unique_test_vector_groups"] = int(len(uniq))
    # Report family-level holdout results separately for HOIC and LOIC-UDP.
    family_rows = []
    for fam_code in np.unique(family_codes[te_used]):
        subset = family_codes[te_used] == fam_code
        ys_family = y[te_used][subset]
        sf = test_scores[subset]
        family_rows.append({"dataset":tag, "family":family_names[int(fam_code)],
                            "test_rows":int(subset.sum()),"attack_rows":int(ys_family.sum()),
                            "detected_attack":int(np.sum((sf>threshold) & (ys_family==1))),
                            "attack_recall":float(np.mean(sf[ys_family==1]>threshold)) if np.any(ys_family==1) else None})
    result["dataset"] = tag
    result["train_rows_used"] = int(len(tr_used))
    result["cal_rows_used"] = int(len(ca_used))
    result["test_rows_used"] = int(len(te_used))
    print(f"{tag} RESULT: AUC={result['roc_auc']:.6f}; AP={result['average_precision']:.6f}; "
          f"recall={result['attack_recall']:.3%}; FPR={result['benign_fpr']:.3%}; "
          f"group-weighted AUC={result['group_equal_weight_auc']:.6f}",flush=True)
    # Per-family support in test; important if the 2018 LOIC-UDP class is tiny.
    # The feature and label columns were streamed; individual family labels are not retained
    # to control memory usage, so only binary statistics are reported here.
    metadata = {
        "purpose":"target-only within-file sanity check; not evidence of cross-dataset semantic compatibility",
        "dataset":tag,"csv_path":str(path),"sha256":file_hash,
        "rows":int(len(y)),"attack_rows":int(y.sum()),"benign_rows":int((y==0).sum()),
        "ordered_features":features,
        "resolved_mapping_2017_names_to_target_names":mapping,
        "feature_hash_grouping":"pandas 64-bit hash of normalized float32 57-feature vectors (NaN retained); labels excluded",
        "group_audit":audit,"full_partition_counts":partition_counts,"used_partition_counts":used_counts,
        "group_overlap_check":"PASS; zero group key overlap among train/cal/test",
        "seed_split":args.split_seed,"seed_model":args.model_seed,
        "model":{"class":"RandomForestClassifier", "n_estimators":args.trees, "class_weight":"balanced",
                 "min_samples_leaf":args.min_leaf,"max_leaf_nodes":args.max_leaf_nodes,"n_jobs":args.n_jobs},
        "imputer":"training-only median, keep_empty_features=True; nonfinite->NaN",
        "threshold_rule":"strict score > empirical 99th percentile of within-target benign calibration scores",
        "sampling_rule":"group-disjoint full partition; optional stratified row sample only within partition; all fixed by seed",
        "python":sys.version,"pandas":pd.__version__,"numpy":np.__version__,
        "sklearn":sklearn.__version__,"platform":platform.platform(),
        "run_time_seconds":time.time()-started,
        "warnings":["No connection, host, time, or capture disjointness established",
                    "Preexisting labels and flow extraction not independently verified",
                    "High AUC here only implies within-file patterns under this split are learnable",
                    "This memory-limited RF differs from uncapped original Experiment 6B RF"],
        "metrics":result,
    }
    sub = outdir / f"target_{tag}"
    sub.mkdir(parents=True,exist_ok=True)
    with (sub/"run_metadata.json").open("w",encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    pd.DataFrame([result]).to_csv(sub/"metrics.csv",index=False)
    pd.DataFrame(family_rows).to_csv(sub/"family_test_metrics.csv",index=False)
    # Index bundles allow reproducible splitting from immutable input file order.
    np.savez_compressed(sub/"partitions.npz",train_idx=tr,cal_idx=ca,test_idx=te,
                        used_train_idx=tr_used,used_cal_idx=ca_used,used_test_idx=te_used)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",choices=["2019","2018","both"],default="both")
    ap.add_argument("--audit-only",action="store_true",help="Only check CSV locations and 57-field schema; do not train")
    ap.add_argument("--path-2019",type=Path,default=None)
    ap.add_argument("--path-2018",type=Path,default=None)
    ap.add_argument("--raw-root",type=Path,default=Path("data/raw"))
    ap.add_argument("--features-json",type=Path,default=Path("reports/score_diagnostics/metadata.json"))
    ap.add_argument("--out",type=Path,default=Path("research/results/experiment11_within_target"))
    ap.add_argument("--split-seed",type=int,default=42)
    ap.add_argument("--model-seed",type=int,default=42)
    ap.add_argument("--n-jobs",type=int,default=2)
    ap.add_argument("--trees",type=int,default=100)
    ap.add_argument("--min-leaf",type=int,default=2)
    ap.add_argument("--max-leaf-nodes",type=int,default=2048)
    ap.add_argument("--max-train",type=int,default=120000)
    ap.add_argument("--max-cal",type=int,default=20000)
    ap.add_argument("--max-test",type=int,default=100000)
    ap.add_argument("--chunk-size",type=int,default=80000)
    ap.add_argument("--skip-hash-check",action="store_true",help="SMOKE TEST ONLY; results NOT for publication")
    args=ap.parse_args()
    if args.trees < 1 or args.max_leaf_nodes < 2 or args.min_leaf < 1:
        raise ValueError("Invalid tree parameters")
    args.out.mkdir(parents=True,exist_ok=True)
    features=read_features(args.features_json)
    tags=["2019","2018"] if args.dataset=="both" else [args.dataset]
    configs=[]
    for tag in tags:
        path=choose_dataset_path(args.raw_root,tag, getattr(args,f"path_{tag}"))
        cols=columns_of(path)
        label="Class" if tag=="2019" else "Label"
        if label not in cols:
            raise ValueError(f"Missing {label} label column in {tag} target file")
        mapping,missing=resolve_columns(features,cols,tag)
        print(f"{tag}: {path}",flush=True)
        print(f"  mapped {len(mapping)}/57 features; renamed aliases={sum(k!=v for k,v in mapping.items())}",flush=True)
        if missing:
            print(f"  MISSING (source name -> candidate alias): {missing}",flush=True)
        configs.append((tag,path,mapping,missing))
    # Write the schema audit even when mapping is incomplete.
    audit={tag:{"path":str(path),"mapped":len(mapping),"mapping":mapping,"missing":missing}
           for tag,path,mapping,missing in configs}
    (args.out/"schema_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(f"Schema audit written: {args.out / 'schema_audit.json'}",flush=True)
    if args.audit_only:
        print("AUDIT COMPLETE (no data read, no model fitted).",flush=True)
        return
    if any(missing for tag,path,mapping,missing in configs):
        raise RuntimeError("STOP: 57-feature target mapping incomplete. Please share schema_audit.json; do not guess names.")
    rows=[]
    for tag,path,mapping,missing in configs:
        rows.append(run_one(tag,path,features,mapping,args,args.out))
    pd.DataFrame(rows).to_csv(args.out/"within_target_summary.csv",index=False)
    print(f"\nCOMPLETE. {args.out/'within_target_summary.csv'}",flush=True)
    print("CAUTION: grouped exact-vector evaluation is not host-, connection-, or time-disjoint,",flush=True)
    print("and high within-target AUC does not validate cross-dataset feature semantics.",flush=True)

if __name__=="__main__":
    main()
