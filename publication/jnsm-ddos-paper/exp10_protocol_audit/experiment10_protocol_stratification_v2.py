#!/usr/bin/env python3
"""Experiment 10: protocol/transport-signature stratification of the fixed Experiment 6B baseline.

This is a diagnostic evaluation only: there is no target-driven model selection,
threshold tuning, or model modification. Refits the documented original seed-42
Random Forest to regenerate flow-level scores, then checks reference metrics.

IMPORTANT: `Init_Win_bytes_forward == -1` is a TCP-window *signature*, NOT a
verified IP protocol label. If the redistributed CSV lacks a Protocol field,
this tool will NEVER label -1-window flows as UDP or non--1-window as TCP.

Run from the network-anomaly-detection project root. Does not change manuscripts.
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

EXPECTED_SOURCE_SHA256 = "6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee"
EXPECTED_TARGET_SHA256 = "97af3b16e45f5309a6492cc16d947a74fdc135267f03756b8cea2eb13f4b2379"
REFERENCE = {"source_auc": 0.999934, "external_auc": 0.904330,
             "external_recall": 0.581139, "external_fpr": 0.106520}
PROTOCOL_COLUMN_CANDIDATES = ["Protocol", "Protocol ID", "IP Protocol", "ip_proto", "proto"]
WINDOW_COLUMN = "Init_Win_bytes_forward"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path):
    columns = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate column names after stripping in {path}")
    return columns


def protocol_column(cols):
    candidates = {s.casefold(): s for s in cols}
    for key in PROTOCOL_COLUMN_CANDIDATES:
        if key.casefold() in candidates:
            return candidates[key.casefold()]
    return None


def parse_protocol_column(series: pd.Series) -> pd.Series:
    """Recognize only explicit TCP(6)/UDP(17); all else remains Other/Unknown."""
    src = series.astype("string").str.strip().str.upper()
    cleaned = src.str.replace(r"\.0+$", "", regex=True)
    result = pd.Series("Other/unknown protocol", index=series.index, dtype="string")
    result.loc[cleaned.isin(["6", "TCP"])] = "TCP (explicit)"
    result.loc[cleaned.isin(["17", "UDP"])] = "UDP (explicit)"
    return result


def window_signature(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    out = pd.Series("Window missing/invalid", index=series.index, dtype="string")
    out.loc[nums.notna()] = "Window != -1 (not protocol)"
    out.loc[nums.eq(-1)] = "Window = -1 (not protocol)"
    return out


def labels_for(src: pd.DataFrame, tgt: pd.DataFrame):
    if "Label" not in src.columns or "Class" not in tgt.columns:
        raise RuntimeError("Missing Label (source) or Class (target): inspect CSV")
    sy = src["Label"].astype("string").str.strip().str.upper()
    ty = tgt["Class"].astype("string").str.strip().str.upper()
    if not sy.isin(["DDOS", "BENIGN"]).all():
        raise RuntimeError(f"Unexpected source labels: {sy.value_counts(dropna=False).to_dict()}")
    if not ty.isin(["ATTACK", "BENIGN"]).all():
        raise RuntimeError(f"Unexpected target classes: {ty.value_counts(dropna=False).to_dict()}")
    return sy.eq("DDOS").to_numpy(dtype=np.int8), ty.eq("ATTACK").to_numpy(dtype=np.int8)


def counts_table(frame: pd.DataFrame, groups: pd.Series, y, dataset: str, definition: str):
    temp = pd.DataFrame({"group": groups.to_numpy(), "is_attack": y})
    return pd.DataFrame([
        {"dataset":dataset, "definition":definition, "group":str(name),
         "total":int(len(sub)), "attack":int(sub.is_attack.sum()),
         "benign":int(len(sub)-sub.is_attack.sum())}
        for name,sub in temp.groupby("group", dropna=False, sort=True)
    ])


def stat_rows(groups: pd.Series, labels: np.ndarray, scores: np.ndarray,
              threshold: float, definition: str, dataset: str):
    if len(groups) != len(scores):
        raise ValueError("Group rows and model score count differ")
    rows = []
    for group, index in groups.groupby(groups, sort=True).groups.items():
        ii = np.asarray(list(index), dtype=np.int64)
        y = labels[ii].astype(bool)
        p = scores[ii]
        positive = p > threshold
        n_attack = int(y.sum())
        n_benign = int((~y).sum())
        row = {
            "dataset": dataset, "definition": definition, "group": str(group),
            "n": len(ii), "attack_n": n_attack, "benign_n": n_benign,
            "tp": int(np.sum(positive & y)), "fp": int(np.sum(positive & ~y)),
            "recall": float(np.mean(positive[y])) if n_attack else np.nan,
            "fpr": float(np.mean(positive[~y])) if n_benign else np.nan,
            "auc_within_group": float(roc_auc_score(y, p)) if n_attack and n_benign else np.nan,
            "mean_score_attack": float(p[y].mean()) if n_attack else np.nan,
            "median_score_attack": float(np.median(p[y])) if n_attack else np.nan,
            "median_score_benign": float(np.median(p[~y])) if n_benign else np.nan,
            "threshold": threshold,
        }
        rows.append(row)
    return rows


def as_numeric_matrix(frame: pd.DataFrame, names: list[str]):
    x = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/ddos.csv"))
    p.add_argument("--target", type=Path, default=Path("data/raw/cicddos2019/cicddos2019_dataset.csv"))
    p.add_argument("--features-json", type=Path, default=Path("reports/score_diagnostics/metadata.json"))
    p.add_argument("--out", type=Path, default=Path("research/results/experiment10_protocol"))
    p.add_argument("--audit-only", action="store_true", help="Inspect headers and class/protocol counts only; no model fitting")
    p.add_argument("--n-jobs", type=int, default=2)
    args = p.parse_args()
    for path in [args.source, args.target]:
        if not path.is_file():
            p.error(f"Missing input: {path}. Run from project root.")
    if not args.audit_only and not args.features_json.is_file():
        p.error(f"Missing original ordered feature list: {args.features_json}")
    args.out.mkdir(parents=True, exist_ok=True)
    source_cols = read_header(args.source)
    target_cols = read_header(args.target)
    src_proto, tgt_proto = protocol_column(source_cols), protocol_column(target_cols)
    print("SOURCE protocol column:", src_proto or "ABSENT", flush=True)
    print("TARGET protocol column:", tgt_proto or "ABSENT", flush=True)
    print("SOURCE TCP-window signature field:", WINDOW_COLUMN in source_cols, flush=True)
    print("TARGET TCP-window signature field:", WINDOW_COLUMN in target_cols, flush=True)
    print("NOTE: -1 window size is NOT a validated UDP label; family names are NOT protocol labels.", flush=True)

    sha_source, sha_target = digest(args.source), digest(args.target)
    if sha_source != EXPECTED_SOURCE_SHA256 or sha_target != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Dataset hashes do not match verified experiment files; stopping. "
                           f"Source={sha_source}, target={sha_target}")

    source = pd.read_csv(args.source, low_memory=False)
    target = pd.read_csv(args.target, low_memory=False)
    source.columns = source_cols
    target.columns = target_cols
    y_source, y_target = labels_for(source, target)
    print(f"Source: {len(source):,} rows, {int(y_source.sum()):,} attack; "
          f"target: {len(target):,} rows, {int(y_target.sum()):,} attack", flush=True)

    count_tables = []
    contexts = []
    for dataset, frame, y, pcol in [
        ("source_2017", source, y_source, src_proto),
        ("target_2019", target, y_target, tgt_proto),
    ]:
        if pcol:
            groups = parse_protocol_column(frame[pcol])
            count_tables.append(counts_table(frame, groups, y, dataset, "explicit_protocol"))
            contexts.append((dataset, "explicit_protocol", groups))
        if WINDOW_COLUMN in frame.columns:
            groups = window_signature(frame[WINDOW_COLUMN])
            count_tables.append(counts_table(frame, groups, y, dataset, "tcp_window_signature_NOT_protocol"))
            contexts.append((dataset, "tcp_window_signature_NOT_protocol", groups))
    if not count_tables:
        raise RuntimeError("Neither explicit Protocol fields nor forward window signature exist. "
                           "Cannot stratify from these CSVs: obtain original PCAP or verified metadata.")

    counts = pd.concat(count_tables, ignore_index=True)
    counts.to_csv(args.out / "protocol_and_window_counts.csv", index=False)
    print("\nCounts from input CSVs (all rows):")
    print(counts.to_string(index=False))
    if tgt_proto is None:
        print("\nWARNING: Target lacks an explicit Protocol field. "
              "The analysis can report only window-signature strata, NOT TCP/UDP strata.", flush=True)
    if args.audit_only:
        print(f"\nAUDIT COMPLETE. {args.out / 'protocol_and_window_counts.csv'}")
        return

    with open(args.features_json, encoding="utf-8") as f:
        features = json.load(f)["features"]
    if len(features) != 57 or len(set(features)) != 57:
        raise RuntimeError("Expected exactly 57 distinct ordered features in baseline metadata")
    if not set(features).issubset(source.columns) or not set(features).issubset(target.columns):
        raise RuntimeError("Original baseline feature list not fully present in both input CSVs")
    idx = np.arange(len(source))
    dev_idx, test_idx = train_test_split(idx, test_size=.20, stratify=y_source, random_state=42)
    train_idx, cal_idx = train_test_split(dev_idx, test_size=.125, stratify=y_source[dev_idx], random_state=42)
    if (len(train_idx), len(cal_idx), len(test_idx)) != (158021,22575,45149):
        raise RuntimeError("Partition sizes do not match original Experiment 6B")
    print("\nRefitting original seed-42 baseline for per-flow predictions (no tuning)...", flush=True)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(as_numeric_matrix(source.iloc[train_idx], features))
    x_cal = imputer.transform(as_numeric_matrix(source.iloc[cal_idx], features))
    x_test = imputer.transform(as_numeric_matrix(source.iloc[test_idx], features))
    x_target = imputer.transform(as_numeric_matrix(target, features))
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight="balanced", n_jobs=args.n_jobs)
    rf.fit(x_train, y_source[train_idx])
    scores_cal = rf.predict_proba(x_cal)[:, 1]
    scores_source = rf.predict_proba(x_test)[:, 1]
    scores_target = rf.predict_proba(x_target)[:, 1]
    threshold = float(np.quantile(scores_cal[y_source[cal_idx] == 0], 0.99))
    observed = {"source_auc":float(roc_auc_score(y_source[test_idx], scores_source)),
                "external_auc":float(roc_auc_score(y_target, scores_target)),
                "external_recall":float(np.mean(scores_target[y_target == 1] > threshold)),
                "external_fpr":float(np.mean(scores_target[y_target == 0] > threshold))}
    print("Original Experiment 6B reference:", {k:round(v, 6) for k,v in observed.items()},
          "threshold=", round(threshold, 6), flush=True)
    for k, expectation in REFERENCE.items():
        if abs(observed[k] - expectation) > 3e-6:
            raise RuntimeError(f"REFERENCE FAIL {k}: expected ~{expectation}, got {observed[k]}. "
                               "STOP; do not interpret saved group metrics.")
    print("REFERENCE CHECK PASS", flush=True)

    rows = []
    for dataset, definition, groups in contexts:
        if dataset == "source_2017":
            # Score only the fixed source holdout, not training/calibration records.
            group_sub = groups.iloc[test_idx].reset_index(drop=True)
            rows.extend(stat_rows(group_sub, y_source[test_idx], scores_source,
                                  threshold, definition, "source_2017_holdout"))
        else:
            rows.extend(stat_rows(groups.reset_index(drop=True), y_target, scores_target,
                                  threshold, definition, "target_2019"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "stratified_metrics.csv", index=False)
    print("\nStratified metrics:")
    print(metrics[["dataset","definition","group","n","attack_n","benign_n","recall","fpr","auc_within_group"]].to_string(index=False))

    # Families are retained as descriptive identifiers; never classify them as TCP/UDP.
    if "Label" in target.columns:
        families = target["Label"].astype("string").str.strip()
        family_rows = []
        for family, ii in families[y_target == 1].groupby(families[y_target == 1]).groups.items():
            ii = np.asarray(list(ii), dtype=int)
            family_rows.append({"family":str(family), "attack_rows":len(ii),
                                "tp":int((scores_target[ii] > threshold).sum()),
                                "recall":float((scores_target[ii] > threshold).mean()),
                                "mean_score":float(scores_target[ii].mean())})
        fam = pd.DataFrame(family_rows).sort_values("attack_rows", ascending=False)
        fam.to_csv(args.out / "family_recall.csv", index=False)
        print("\nTop attack families:")
        print(fam.head(16).to_string(index=False))
        for dataset, definition, groups in contexts:
            if dataset != "target_2019":
                continue
            ct = pd.crosstab(families[y_target == 1], groups[y_target == 1], dropna=False)
            ct.to_csv(args.out / f"family_by_{definition}.csv")

    # Follow-up: attack-family x explicit protocol, with observed decisions per group.
    # Unlike family_by_explicit_protocol.csv (counts only), this reports recall.
    if tgt_proto is not None and "Label" in target.columns:
        proto_group_all = parse_protocol_column(target[tgt_proto]).to_numpy(dtype=str)
        protocol_raw_all = target[tgt_proto].astype("string").fillna("(missing)").to_numpy(dtype=str)
        family_all = target["Label"].astype("string").str.strip().fillna("(missing)").to_numpy(dtype=str)
        attack_mask = (y_target == 1)
        predicted = (scores_target > threshold)
        fam_proto = pd.DataFrame({
            "family": family_all[attack_mask],
            "protocol_group": proto_group_all[attack_mask],
            "protocol_value": protocol_raw_all[attack_mask],
            "detected": predicted[attack_mask].astype(np.int8),
            "score": scores_target[attack_mask],
            "score_tied_at_threshold": np.isclose(scores_target[attack_mask], threshold, atol=1e-12, rtol=0).astype(np.int8),
        })
        grouped = (fam_proto.groupby(["family", "protocol_group", "protocol_value"], dropna=False)
                   .agg(attack_rows=("detected", "size"),
                        tp=("detected", "sum"),
                        tied_at_threshold=("score_tied_at_threshold", "sum"),
                        mean_score=("score", "mean"),
                        median_score=("score", "median"))
                   .reset_index())
        grouped["fn"] = grouped["attack_rows"] - grouped["tp"]
        grouped["recall"] = grouped["tp"] / grouped["attack_rows"]
        grouped["threshold"] = threshold
        grouped["low_sample_count_flag"] = grouped["attack_rows"] < 100
        grouped = grouped.sort_values(["family", "attack_rows"], ascending=[True, False])
        assert int(grouped["attack_rows"].sum()) == int(attack_mask.sum())
        assert int(grouped["tp"].sum()) == int((predicted & attack_mask).sum())
        grouped.to_csv(args.out / "family_protocol_performance.csv", index=False)

        # Save raw flow-level scores by immutable target row order. These are *scores*,
        # not underlying flow features. Dataset hash and threshold are in run_metadata.json.
        np.savez_compressed(
            args.out / "target_scores_seed42.npz",
            score=scores_target.astype(np.float64),
            label=y_target.astype(np.int8),
            protocol=protocol_raw_all.astype("U32"),
            family=family_all.astype("U64"),
            threshold=np.array([threshold], dtype=np.float64),
        )

        # Comparison within families represented substantially in both TCP and UDP.
        eligible = grouped[(grouped["protocol_group"].isin(["TCP (explicit)", "UDP (explicit)"]))
                           & (grouped["attack_rows"] >= 100)].copy()
        pvt = eligible.pivot(index="family", columns="protocol_group", values="attack_rows")
        fam_both = pvt.dropna().index
        print("\nWithin-family protocol recall (>=100 attack rows in EACH group):", flush=True)
        if len(fam_both):
            disp = eligible[eligible["family"].isin(fam_both)]
            print(disp[["family", "protocol_group", "attack_rows", "tp", "recall", "median_score"]].to_string(index=False), flush=True)
        else:
            print("No families have >=100 attack records in both TCP and UDP.", flush=True)
        print("Saved family_protocol_performance.csv and target_scores_seed42.npz", flush=True)

    info = {
        "source_sha256":sha_source, "target_sha256":sha_target,
        "features_json":str(args.features_json), "features_ordered":features,
        "split_seed":42, "forest_seed":42,
        "source_partition_rows":{"train":len(train_idx),"calibration":len(cal_idx),"holdout":len(test_idx)},
        "threshold":threshold, "reference":observed,
        "source_protocol_column":src_proto,"target_protocol_column":tgt_proto,
        "window_signature_is_not_protocol":True,
        "python":sys.version,"sklearn":sklearn.__version__,"pandas":pd.__version__,
        "platform":platform.platform(),
        "interpretation_warning":"Inferred -1 window groups are not validated TCP/UDP; origin and preprocessing lineage remain limited."
    }
    with open(args.out / "run_metadata.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCOMPLETE. Outputs in {args.out}", flush=True)

if __name__ == "__main__":
    main()
