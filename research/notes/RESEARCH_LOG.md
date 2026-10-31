# Network Anomaly Detection — Research Log

**Project:** Cross-dataset DDoS detection and network anomaly diagnosis  
**Record created:** 2026-10-30  
**Status:** Exploratory research; no publication claims established.

## Research questions
1. How well does a flow-based DDoS detector trained on CIC-IDS2017 transfer to a redistributed CIC-DDoS2019 subset?
2. Are external errors associated with attack family, feature-distribution shift, or feature-extraction differences?
3. Can a validated approach improve detection of unseen attack families without tuning on the final external test set?

## Data provenance and caveats
- **Training:** `data/raw/ddos.csv`, CIC-IDS2017 Friday DDoS CSV; 225,745 rows, 128,027 DDoS and 97,718 benign (previously reported).
- **External:** `data/raw/cicddos2019/cicddos2019_dataset.csv`, redistributed CIC-DDoS2019-derived subset. Source and transformation pipeline **not yet verified**.
- External subset: 431,371 rows; 333,540 Attack and 97,831 Benign; `Class` is binary label, `Label` identifies traffic type.
- Column-name comparison: 67 shared names before training-only constant-feature removal; 57 usable features in external evaluation. Identical names do not prove identical semantics.
- Neither dataset's raw capture independence nor potential record overlap has yet been verified.

## Experiment history (reported terminal outputs)

### E01 — Initial Random Forest internal baseline
- Script: `src/train.py`
- First 100,000 rows; stratified 80/20 random split; RF 100 trees; median imputation; class_weight balanced.
- Accuracy 99.985%; confusion matrix [[7761, 0], [3, 12236]].
- Limitation: random split of same capture, possible duplicates and capture-specific shortcuts.

### E02 — Feature importance / ablation
- Scripts: `src/feature_importance.py`, `src/ablation.py`
- High-importance features previously reported: Init_Win_bytes_forward, Bwd Packet Length Max, Fwd Packet Length Mean, Subflow Fwd Bytes, Avg Fwd Segment Size.
- Removing top 1/5/15 features retained very high internal accuracy; verify exact saved outputs before citing.

### E03 — Block holdout
- Script: `src/block_holdout.py`
- Train first 175,000 rows, test remaining 50,745; accuracy ~99.74%; 88 FP, 44 FN; ROC-AUC ~0.99933.
- Limitation: row order is not verified chronological order.

### E04 — Full dataset audit
- Script: `research/experiments/dataset_audit.py`
- 225,745 rows; 2,633 exact duplicate rows; 64 infinite numeric values; 4 missing values in Flow Bytes/s; 10 constant columns; no explicit timestamp or flow identifier columns.
- Interpretation: data quality requires attention; no proof of leakage from duplicates alone.

### E05 — Deduplication diagnostic
- Script: `research/experiments/deduplication_test.py`
- Original first 100,000 rows: 68 features, accuracy 0.99985, F1 0.9998774259448416; confusion matrix [[7761,0],[3,12236]].
- Deduplicated: 99,271 rows (729 removed), 68 features, accuracy 0.9997985394107278, F1 0.9998365746036935; confusion matrix [[7615,1],[3,12236]].
- Limitation: changed train/test composition; diagnostic, not a controlled estimate of duplicate effect.

### E06 — External validation
- Script: `research/experiments/cross_dataset_test.py`
- Trained on 225,745 CIC-IDS2017 rows using 57 common usable features; tested on 431,371 CIC-DDoS2019-subset rows without retraining.
- Accuracy 0.25300031759204955; attack precision 0.9885067404078811; attack recall 0.034295736643281165; attack F1 0.06629152275203412.
- Confusion matrix [[97698,133],[322101,11439]] (rows actual benign/attack; columns predicted benign/attack).
- SYN: 11,439 / 49,373 detected (23.1685%); all other attack categories combined: 0 / 284,167 detected. Benign correct: 97,698 / 97,831.
- Caution: internal baseline used a different feature set; matched-feature internal baseline pending. Different attack types, capture conditions, and possible preprocessing differences may explain the failure.

### E07 — Feature-shift audit
- Script: `research/experiments/feature_shift_audit.py`
- CIC-IDS2017 vs CIC-DDoS2019 attack medians: Total Backward Packets 4 vs 0; Bwd Packet Length Mean 1934.5 vs 0; Fwd Packet Length Mean 7 vs 440; Flow Bytes/s ~160.1 vs ~13215.7.
- Attack zero-backward-packet fraction 36.3423% vs 87.2357%.
- Interpretation: major distribution differences; do not assert causation or identical feature definitions.

### E08 — External error stratification
- Added diagnostic to `research/experiments/cross_dataset_test.py`.
- With >=1 backward packet: 42,574 attacks, 2,142 detected, recall 5.0312%.
- With zero backward packets: 290,966 attacks, 9,297 detected, recall 3.1952%.
- SYN: 49,373 attacks, 11,439 detected, recall 23.1685%.
- Other attacks: 284,167 attacks, 0 detected.
- Interpretation: attack family is strongly associated with failure; backward-packet count alone does not explain it. No causal attribution yet.

## Hypothesis tracker
| ID | Hypothesis | Current evidence | Status |
|---|---|---|---|
| H1 | Random-split accuracy overstates cross-dataset transfer | 99.985% internal vs 25.30% external, but feature sets differ | Supported provisionally; controlled baseline needed |
| H2 | Attack-family shift drives poor external recall | SYN 23.17%; non-SYN 0% | Strong association, cause unverified |
| H3 | Zero backward packets explain missed attacks | 3.20% vs 5.03% recall | Insufficient as sole explanation |
| H4 | Feature extraction / dataset processing is incompatible | 67 shared names; strong feature shifts | Open; requires provenance and semantic checks |

## Publication bucket list
1. **Cross-dataset DDoS generalization and attack-family/feature shift.** Feasible experiments underway; novelty and dataset compatibility not yet established.
2. **Attack-versus-operational-fault diagnosis with synchronized telemetry.** Requires genuine synchronized data or controlled testbed; do not imply synthetic independent data are real observations.
3. **SDN/DRL fault management for QKD-enabled optical networks.** Separate thesis-related research direction.

## Priority next steps
- [ ] Run matched-57-feature internal holdout, retaining exact split seed and metrics.
- [ ] Verify redistributed CIC-DDoS2019 CSV provenance, transformations, feature definitions and licensing.
- [ ] Check train/test feature distributions and prediction scores; compare attack families without tuning on external labels.
- [ ] Design a separate validation dataset / untouched holdout for model-selection experiments.
- [ ] Review closest cross-dataset DDoS papers to establish a genuinely novel contribution.
- [ ] Record package versions, git commit hash, hardware, run time, random seeds, data hashes and exact commands for every future experiment.

## Reproducibility template (copy for each run)
**Experiment ID / date:**  
**Research question / hypothesis:**  
**Script and command:**  
**Git commit / environment:**  
**Dataset source, version, hash, rows and labels:**  
**Preprocessing and feature list:**  
**Split protocol / random seed / leakage checks:**  
**Model and hyperparameters:**  
**Metrics and saved output files:**  
**Interpretation:**  
**Limitations / alternative explanations:**  
**Next decision:**  


### Experiment: Matched-Feature Internal Baseline and External Validation

**Date:** 30 October 2026

**Objective:** Determine whether the large external performance degradation persists when internal and external evaluations use the same feature set and model configuration.

**Datasets:**
- Training/internal evaluation: CIC-IDS2017 Friday DDoS subset, 225,745 flows.
- External evaluation: redistributed CIC-DDoS2019 CSV, 431,371 flows.

**Method:**
- Random Forest, 100 trees, balanced class weights, random state 42.
- 57 common usable numeric features selected using CIC-IDS2017.
- Median imputation.
- Internal evaluation: stratified 80/20 random split.
- External evaluation: separately fitted model trained on all CIC-IDS2017 flows and evaluated on CIC-DDoS2019 without retraining on external labels.

**Results:**

| Metric | Internal | External |
|---|---:|---:|
| Accuracy | 99.9646% | 25.3000% |
| Attack precision | 99.9727% | 98.8507% |
| Attack recall | 99.9649% | 3.4296% |
| Attack F1 | 99.9688% | 6.6292% |

**Internal confusion matrix:** TN = 19,537; FP = 7; FN = 9; TP = 25,596.

**External confusion matrix:** TN = 97,698; FP = 133; FN = 322,101; TP = 11,439.

**Attack-family findings:**
- SYN: 11,439 of 49,373 detected (23.17%).
- All other attack categories combined: 0 of 284,167 detected.

**Backward-traffic findings:**
- At least one backward packet: 5.03% attack detection.
- Zero backward packets: 3.20% attack detection.

**Interpretation:** The failure persists with a matched feature set. The external model detects very few attacks and recognizes only a subset of SYN traffic. This is consistent with poor cross-dataset transfer, but the cause remains unresolved.

**Limitations:** Different fitted models and training sizes; potential dataset provenance and feature extraction differences; internal random split may overestimate generalization; no independent replication or statistical uncertainty analysis yet.

**Next actions:**
1. Verify CIC-DDoS2019 CSV provenance and feature definitions.
2. Check feature distributions and prediction scores.
3. Compare with existing cross-dataset DDoS research.
4. Evaluate additional baselines and controlled feature subsets.
5. Preserve exact experiment configuration and results.
---

## Experiment 03 — Operating-Point Feasibility and Score-Tie Analysis

**Objective:** Evaluate external attack recall under constrained false-positive rates.

**Method:** Analyze saved prediction scores from Experiment 01 without retraining. Compare strict and inclusive thresholds and calculate target-label-informed ROC operating points.

### Results

| Metric | Result |
|---|---:|
| External ROC-AUC | 0.90433 |
| Recall at score > 0.01 | 58.114% |
| FPR at score > 0.01 | 10.652% |
| Recall at score >= 0.01 | 96.138% |
| FPR at score >= 0.01 | 14.480% |
| Recall at score >= 0.50 | 2.830% |
| FPR at score >= 0.50 | 0.133% |
| External attacks with score exactly 0.01 | 38.02% |
| External benign flows with score exactly 0.01 | 3.83% |

### Target Oracle Operating Points

| Maximum FPR | Achieved FPR | Attack Recall |
|---|---:|---:|
| 1% | 0.595% | 7.023% |
| 5% | 4.201% | 18.555% |
| 10% | 8.872% | 53.711% |

### Interpretation

The model retains substantial aggregate ranking discrimination (ROC-AUC = 0.90433), but detection performance at low external false-positive rates remains inadequate.

Large concentrations of prediction scores at 0.01 produce abrupt changes in attack recall and false-positive rates when the threshold comparison changes from > to >=.

### Limitations

- Oracle operating points were selected using target labels and are not deployable thresholds.
- External dataset provenance and feature equivalence remain unverified.
- These results do not establish the causal mechanism behind score concentration.
- The external dataset has a high attack prevalence, so its observed precision should not be generalized to operational traffic.

### Next Experiment

Experiment 04: Semantic feature compatibility and failure-attribution audit.

