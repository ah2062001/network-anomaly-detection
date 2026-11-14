# Experiment 01 — Prediction-score and threshold transfer

**Status:** Designed, not yet executed.  
**Date designed:** 2026-10-30  
**Associated manuscript:** *Attack-Family Blind Spots in Cross-Dataset DDoS Detection* (working title)

## Research question
Does the observed CIC-IDS2017 → CIC-DDoS2019 classification failure reflect loss of discriminative ranking, or predominantly a non-transferable decision threshold?

## Data and provenance
- Source: `data/raw/ddos.csv` (CIC-IDS2017 Friday DDoS subset; earlier audit: 225,745 flows).
- Target: `data/raw/cicddos2019/cicddos2019_dataset.csv` (redistributed CIC-DDoS2019 subset; earlier label audit: 431,371 flows).
- **Important:** Original target dataset provenance, sampling, feature extraction software version and semantic feature equivalence remain to be verified.
- Source label mapping: `BENIGN=0`, `DDoS=1`. Target class mapping: `Benign=0`, `Attack=1`.
- Target attack-type `Label` is used for analysis after inference, not selection or training.

## Pre-registered diagnostic expectations
- **H1 (ranking failure):** Target ROC-AUC and average precision may approach chance/reference levels despite high source-test values.
- **H2 (threshold failure):** Target score ranking may remain useful despite poor recall at the fixed 0.5 threshold.
- **H3 (family dependence):** Detection rates/scores differ by target attack family, especially SYN vs others.

These are alternative/non-exclusive hypotheses; their truth is unknown before Experiment 01.

## Protocol
1. Stratified source split: 70% model training, 10% threshold calibration, 20% held-out internal test (seed 42).
2. Choose compatible numeric column *names* from source and target; select nonconstant features using source training rows only. (Shared names do NOT establish semantic compatibility.)
3. Fit median imputation + Random Forest (100 trees, balanced weights, seed 42) **only** on source training rows.
4. Fit no learned transformation using target data.
5. Obtain raw attack probabilities on source calibration, source holdout, and target dataset using the **same fitted model**.
6. Predefine operational thresholds: default 0.5 and source calibration benign-score 99th percentile with strict greater-than comparison (nominal 1% FPR, actual achieved FPR reported because of ties).
7. Evaluate external labels only **after** all scores and thresholds have been calculated. Save overall metrics, confusion matrices, score quantiles, attack-type breakdown, and per-flow scores.

## Primary outcomes
- ROC-AUC (0.5 denotes chance ranking; below 0.5 suggests reversed ranking but not proof of its cause).
- Average precision (random reference approximately target attack prevalence; depends on tie behavior).
- Attack recall, precision, F1, balanced accuracy, FPR at both thresholds.
- Class-conditional score distributions, per-attack-family score and detection distributions.

## Interpretation rules and caveats
- No threshold or feature subset is chosen using external target labels.
- Source calibration is not a completely independent report of source FPR; report held-out source test FPR separately.
- We have already inspected CIC-DDoS2019 labels and errors in earlier exploratory tests. It is an *exploratory target*, not an untouched final confirmation set.
- Differences in source and target traffic family, capture, dataset composition, and feature semantics remain confounded.
- A poor ROC-AUC does not establish that the underlying traffic is inherently undetectable; it only pertains to the trained model and supplied representation.
- Report attack prevalence alongside average precision and accuracy.

## Output artifacts
`reports/score_diagnostics/summary.csv`, `score_quantiles.csv`, `by_traffic_type.csv`, `source_feature_importance.csv`, `external_per_flow_scores.csv.gz`, `source_holdout_scores.csv.gz`, `metadata.json`.

## Next actions after results
1. Inspect overall ROC-AUC, average precision and attack/benign score quantiles.
2. Decide whether to prioritize ranking/feature diagnostics or source-only threshold/calibration strategies.
3. Verify target provenance and feature extraction equivalence before causal attribution.
4. Repeat final method with held-out independent confirmation dataset, if warranted.
