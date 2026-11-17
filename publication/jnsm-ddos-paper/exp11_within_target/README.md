# Experiment 11 — Within-target sanity checks

**Status:** Script verified with synthetic test CSVs; **no real 2019 or 2018 experiment results have been generated here**. Run on your own Mac using the byte-identified input CSVs.

## Purpose

Train and evaluate a *new, separate* Random Forest within each of:

- Redistributed Mendeley CIC-DDoS2019 (`431,371` rows; local hash `97af3b16…f4b2379`), and
- CSE-CIC-IDS2018 file `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` (`1,048,575` rows; local hash `a5f4a1c2…26611be04`).

The exact **ordered 57 features** from `reports/score_diagnostics/metadata.json` are used. For the 2018 CSV, the script checks a hard-coded list of explicit candidate CICFlowMeter column aliases. These aliases are a schema mapping, not a guarantee of identical feature semantics across datasets.

This is a **within-target learnability and pipeline sanity check**, NOT a proof that the earlier 2017→2018/2019 transfer failure was caused solely by network shift, or that the original labels/extraction are correct. No CIC-IDS2017 data are used for training here.

## Install

From `~/Projects/network-anomaly-detection` with the project `.venv` activated:

```bash
unzip -o "$HOME/Downloads/experiment11_within_target_package.zip"
```

## Step A — Audit schema and locate datasets (fast; NO training)

```bash
python3 exp11_within_target/experiment11_within_target.py \
  --audit-only \
  --out research/results/experiment11_within_target
```

The script automatically searches `data/raw/` for the 2018 filename. If it is not found, supply its real full path using `--path-2018 "/path/to/Wed...csv"`. If any of the 57 fields is missing, inspect the terminal output and `schema_audit.json` before proceeding. Preserve the documented feature mapping.

## Step B — Run only 2019 first

```bash
python3 exp11_within_target/experiment11_within_target.py \
  --dataset 2019 \
  --n-jobs 2 \
  --out research/results/experiment11_within_target
```

## Step C — Run 2018 independently

```bash
python3 exp11_within_target/experiment11_within_target.py \
  --dataset 2018 \
  --n-jobs 2 \
  --out research/results/experiment11_within_target
```

The default model is `RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_leaf=2, max_leaf_nodes=2048, random_state=42, n_jobs=2)`. **It is deliberately capacity-limited and trained with capped samples for laptop memory; do not call it a direct reproduction of the unbounded original source model.** Defaults use at most 120,000 training, 20,000 calibration and 100,000 test rows, selected *within* the grouped partitions using fixed seeds. The full target file is read for group audits.

To use the original unbounded forest on a machine with sufficient resources, the script would need a code change to support `max_leaf_nodes=None`, and the larger-model results should be distinguished from this pilot. Do not change the model based on external test results.

## Leakage precautions and caveats

1. Deterministically hash each float32 model-input vector across the same 57 features, excluding labels. Identical modeled vectors are grouped before splitting. Check zero group-hash overlap across train/calibration/test.
2. Select grouped development/holdout (approx 80/20 **of groups**) and grouped train/calibration (approx 87.5/12.5 **of remaining groups**) once with fixed split seed 42. **Group sizes differ**, so actual row counts and prevalence may deviate from 70/10/20; inspect output metadata.
3. Apply optional stratified row caps *within each existing partition*, with fixed seeds. Fit median imputer only on selected training data; calibrate decision threshold using only benign calibration scores (`score > empirical 99th percentile`). Evaluate once on held-out test rows.
4. Save both row-weighted AUC and test-vector-equal-weight AUC, class precision/recall/FPR, confusion matrix, AP, and per-family test counts. High performance is not proof of clean labels, host/session/time independence, or cross-dataset semantic compatibility.
5. The grouping uses a 64-bit hash of normalized float32 vectors. Hash collision risk is extremely small but not theoretically zero. Identical input vectors with contradictory labels are reported. Different vectors may still represent the same connection or capture condition.
6. Dataset SHA-256 checks are mandatory for publication runs (do not use `--skip-hash-check`, which is for synthetic testing). The script does **not** overwrite your earlier Experiment 6B–10 results or manuscript.

## Outputs

`research/results/experiment11_within_target/`

- `schema_audit.json`: dataset paths, feature map, missing aliases.
- `target_2019/metrics.csv`, `run_metadata.json`, `family_test_metrics.csv`, `partitions.npz`.
- `target_2018/metrics.csv`, `run_metadata.json`, `family_test_metrics.csv`, `partitions.npz`.
- `within_target_summary.csv` is created when `--dataset both` is used in one invocation. For separate runs, the two per-target `metrics.csv` files hold the results.

The full grouped partitions and the sampled indices are saved. The measured variability is conditional on the selected dataset file and split, and these are **descriptive sanity checks, not independent deployments**.

## First share

Paste the `--audit-only` terminal output. If all 57 fields map in both files, run the 2019 test first and send its final `RESULT` line and `metrics.csv`.
