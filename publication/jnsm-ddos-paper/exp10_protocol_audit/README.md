# Experiment 10 — Protocol / TCP-window stratification (diagnostic)

Run from the root of `network-anomaly-detection` with the existing `.venv` activated.
No manuscript changes and no dataset changes are made.

## Step 1: Fast audit (no model fit)

```bash
python3 exp10_protocol_audit/experiment10_protocol_stratification.py --audit-only \
  --out research/results/experiment10_protocol_audit
```

Check whether the **actual redistributed target CSV** contains `Protocol` (or a supported equivalent) and view the `protocol_and_window_counts.csv` table. A `-1` forward TCP-window feature is **not proof of UDP**; an attack-family name containing UDP is also not a verified per-flow protocol field.

## Step 2: Reference-checked stratified scoring

```bash
python3 exp10_protocol_audit/experiment10_protocol_stratification.py \
  --n-jobs 2 \
  --out research/results/experiment10_protocol_scoring
```

Uses the identical source and target CSV SHA-256s and the ordered 57-feature metadata from original Experiment 6B, fixed 42 source split and RF seed, training-only median imputation, 100 trees, balanced class weights, and a strict `score > 99th percentile(source benign calibration)` decision rule. Reproduces original external AUC 0.904330 and refuses to report group metrics if reference metrics do not match.

Files:
* `protocol_and_window_counts.csv`: class counts for true `Protocol` (if present) and always separately for forward-window `== -1` / `!= -1` signatures (if available), in source and target.
* `stratified_metrics.csv`: per-group attack recall, benign FPR, and ROC-AUC *only where both classes occur*. Source metrics refer to its fixed holdout; target metrics to whole evaluated CSV.
* `family_recall.csv`: attack recall for existing `Label` families, without pretending family name is a protocol measurement.
* `family_by_explicit_protocol.csv` or `family_by_tcp_window_signature_NOT_protocol.csv`: attack-family composition of strata.
* `run_metadata.json`: source and target hashes, feature list, thresholds, split sizes, software versions, references.

If the CSV lacks `Protocol`, **do not report TCP-only / UDP-only AUC**. Report a window-signature stratification and say that definitive transport-protocol confirmation would require verified original flow/PCAP metadata or a file that includes per-flow protocol. Do not relabel window signatures as protocols.

Analytical caveat: If true protocol is available, within-protocol AUC is evaluated against benign flows of that protocol. Family-level AUC is not calculated here. Identical benchmark CSVs do not establish independent deployment performance or verified original extraction/labels.
