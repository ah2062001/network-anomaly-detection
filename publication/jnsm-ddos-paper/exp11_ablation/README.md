# Experiment 11: controlled within-2018 feature ablations

From your repository root, with the existing Experiment 11 files still present:

```bash
unzip -o "$HOME/Downloads/experiment11_ablation_package.zip"
python3 exp11_ablation/experiment11_template_ablation.py \
  --results-dir research/results/experiment11_2018 \
  --n-jobs 2 \
  --out research/results/experiment11_ablation_2018
```

Models: unchanged reference 57 features; minus the precise three template fields;
minus a broader packet size/count family (including known redundancies). Exactly
same saved train/cal/test row indices and source-derived 2018 study model settings.
Each arm recalibrates its own threshold using the existing 2018 calibration
partition and the same 99th-percentile rule. The original reference must match
saved 2018 AUC, TP, FP, TN, FN, and threshold before any ablation scores are accepted.

Audit limitation: after dropping columns, previously distinct full 57-feature
vectors may become identical. This script checks new collisions of projected
feature vectors and additionally compares ablated and full baseline results on
the very same subset of test vectors that are absent in train/cal under the
ablated representation. See `projected_vector_overlap.csv`.

Outputs: `ablation_metrics.csv`, `projected_vector_overlap.csv`,
`family_recall.csv`, `manifest.json`.

Interpretation: exploratory, post-hoc feature-family ablation; not evidence of
causality, correct labels, compatible flow extraction, or independence at the
host, connection, capture, or time level.
