# Experiment 9: paired Random Forest noise floor and placebo

**Purpose:** Test whether Experiment 6B's fixed-split external AUC difference (0.904330 to 0.827471) persists across forest seeds and differs from a class-matched, same-size placebo removal.

**Important:** Unzip the package in the `network-anomaly-detection` project root. The original Experiment 6B Python source was not supplied. The default stratified split is a *hypothesis*, not confirmed. The script **stops** if its feature count, train size, duplicate count or seed-42 reference AUC does not match. Do not use `--skip-reference-check` to create publication evidence.

## On the Mac, from the project root

Run the one-seed smoke/reproduction check first:

```bash
python3 exp9_noise_floor/experiment9_noise_floor.py --seeds 42 --n-jobs 2 --out research/results/experiment9_pilot
```

After the reference checks pass, run a paired 20-seed evaluation with 2 independently selected class-matched placebo subsets:

```bash
python3 exp9_noise_floor/experiment9_noise_floor.py --seeds 42-61 --placebo-reps 2 --n-jobs 2 --out research/results/experiment9_20seeds
```

That is 20 seeds × (1 original + 1 deduplicated + 2 placebo replicates) = 80 forest fits. Set `--placebo-reps 1` for 60 fits. Each placebo subset is fixed *across forest seeds*; use different placebo replicates to capture subset-selection variability.

If reconstruction does **not** match, do not force through. Inspect the existing Experiment 6B code. Ideally export the original exact indices and rerun:

```python
np.savez_compressed('research/experiments/experiment6B_indices.npz',
                    train_idx=train_idx, cal_idx=cal_idx, test_idx=test_idx)
```

The variable names in this snippet are examples: use the actual index variables from your code. Then:

```bash
python3 exp9_noise_floor/experiment9_noise_floor.py --split-npz research/experiments/experiment6B_indices.npz --seeds 42 --out research/results/experiment9_pilot_exact
```

If its feature list or preprocessing also differs, integrate the exact original 6B code before running the multiseed experiment. The result must reproduce the reported baseline first.

## Output

- `manifest.json`: file hashes, features, versions, split provenance, class counts, seed settings
- `partitions.npz`: index arrays for audit
- `placebo_removed_indices_rep*.npy`: independently chosen placebo rows
- `per_run.csv`: external/source AUC, recall, FPR, threshold, TP/FP per model
- `paired_by_seed.csv`: paired AUC differences by seed (placebos averaged by seed)
- `arm_summary.csv`: descriptive distributions over seeds

Interpret seed ranges/quantiles as **algorithmic variability on a fixed dataset**, not confidence intervals for independent networks. The placebo samples are matched on *class*, not all underlying feature distributions. No manuscript changes are made.


## v2 correction (2026-11-09)
The v1 runner generated the 57 features in source CSV order, even though Experiment 1/6B used a saved ordered list at `reports/score_diagnostics/metadata.json` under the `features` key. Random Forest results can differ when a feature matrix is reordered, because the random feature subset at each split uses feature indices. v2 **requires this original metadata**; it will not silently fall back to guessed CSV ordering.

Run from repository root:

```bash
python3 exp9_noise_floor/experiment9_noise_floor.py --seeds 42 --n-jobs 2 --out research/results/experiment9_pilot_v2
```

If the reference still fails, compare the exact original 6B script: `research/experiments/experiment_06b_fixed_split_dedup.py`; inspect feature transforms, labels, split, and any RF parameters. Do not use `--skip-reference-check` to claim successful reproduction.
