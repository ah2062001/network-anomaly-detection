# Research Log — Experiment 01: Score and Threshold Transfer

**Date:** 30 October 2026  
**Status:** Completed exploratory experiment; no mitigation validated.  
**Project:** Flow-based DDoS cross-dataset transfer, CIC-IDS2017 -> redistributed CIC-DDoS2019 subset.

## Hypothesis and design

**Question:** Does poor external recall at a fixed decision threshold reflect complete loss of discriminative score ranking or primarily failure of score/threshold transfer?

**Model and source-only protocol:** RandomForestClassifier, 100 trees, class_weight='balanced', random_state=42, median imputation. 57 shared numeric features that vary in **source training**. Stratified source split: training 158,021, calibration 22,575, holdout 45,149; target 431,371. The **same fitted classifier** scores both source holdout and target. Threshold 0.50 uses `score >= 0.50`. A nominal 1% benign FPR threshold is chosen using source calibration only: score `> 0.010000` (strict greater-than to handle ties); actual source calibration benign FPR 0.6549%. Target labels are used **only after scoring** for metrics and post-hoc interpretation. Output files saved in `reports/score_diagnostics/`.

## Primary quantitative results

| Evaluation | Threshold | Accuracy | Balanced accuracy | Attack precision | Attack recall | Attack F1 | Benign FPR | ROC-AUC | AP | AP minus prevalence | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source internal holdout | >=0.50 | 0.99969 | 0.99969 | 0.99977 | 0.99969 | 0.99973 | 0.00031 | 0.99993 | 0.99993 | 0.43280 | 19,538 | 6 | 8 | 25,597 |
| Source internal holdout | >0.01 | 0.99714 | 0.99672 | 0.99514 | 0.99984 | 0.99749 | 0.00640 | 0.99993 | 0.99993 | 0.43280 | 19,419 | 125 | 4 | 25,601 |
| External target | >=0.50 | 0.24837 | 0.51349 | 0.98641 | 0.02830 | 0.05502 | 0.00133 | 0.90433 | 0.95342 | 0.18021 | 97,701 | 130 | 324,101 | 9,439 |
| External target | >0.01 | 0.65197 | 0.73731 | 0.94898 | 0.58114 | 0.72084 | 0.10652 | 0.90433 | 0.95342 | 0.18021 | 87,410 | 10,421 | 139,707 | 193,833 |

**External prevalence:** 333,540 / 431,371 ≈ 77.32% attack flows. External AP baseline for random ordering is attack prevalence ≈ 0.7732, so AP lift is ≈ 0.1802.

**Score distributions:** Source benign median 0, attack median 1.0; external benign median 0, attack median 0.03, indicating a major score-scale change. External benign mean 0.00888; external attack mean 0.07424.

**Attack-family recall at source threshold `>0.01`:** Syn 98.550%, TFTP 90.926%, UDP-lag 70.548%, DrDoS_NTP 39.312%, UDP 1.675%, DrDoS_UDP 1.631%, with significant differences among additional types. At threshold 0.50, only Syn has substantial recall (19.118%); others listed show 0%.

## Interpretation supported by evidence

1. Target ROC-AUC 0.90433 establishes useful **aggregate target ranking** even as default-threshold recall collapses. Thus 'no transferable signal' is **not** supported.
2. The threshold selected from the source benign calibration distribution does **not** meet the source nominal operating point on external traffic: external benign FPR 10.652% versus source calibration 0.6549% and source holdout 0.640%.
3. Lowering the threshold raises external attack recall from 2.830% to 58.114%, at the expense of 10,291 additional false positive flows (130 -> 10,421). This is **not** an operationally validated remedy.
4. Target precision at `>0.01` (94.898%) is influenced by the unusually high attack prevalence in this redistributed subset and should not be generalized to deployment prevalence.
5. Family-specific results imply heterogeneous ranking/score transfer; per-family ROC-AUC against a shared external benign reference is the next diagnostic.

## Important non-conclusions and threats to validity

- High aggregate ROC-AUC may mask weak ranking for particular attack families. It does not prove class-wise robust generalization.
- The thresholds are on raw Random Forest `predict_proba` outputs; their values are **not guaranteed calibrated real-world attack probabilities**.
- The script uses strict `>` for source threshold but `>=` for 0.50. Many scores cluster at 0.01; record tied-score counts.
- Target CSV provenance, sampling, attack-label definitions, CICFlowMeter version and feature equivalence require independent verification.
- Flow records could be correlated within capture periods or attack runs; no capture-group identifiers are currently established.
- The external test set has already been explored extensively, so a remedy designed in response to its outcomes needs a **different untouched confirmation dataset**.
- The target set's attack prevalence ≈77.32% means precision cannot be extrapolated to an operational network without prevalence assumptions.
- This experiment establishes diagnostic observations only. Neither a new method nor causal explanation has yet been demonstrated.

## Next experiment

Run `experiment_02_family_ranking.py` on the frozen Experiment 01 per-flow score files. For each target attack family, report ROC-AUC against the same external benign reference, score quantiles, ties at 0.01 and attack recalls at both previously selected thresholds. Then compare feature semantics and the source/target collection pipeline before designing any intervention.
