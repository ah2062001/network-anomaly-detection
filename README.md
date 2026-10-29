# Network Anomaly Detection & Telecom Fault Diagnosis

**Machine Learning | Network Security | Telecom Operations | Explainable AI**

An experimental machine learning project investigating network anomaly detection, DDoS traffic classification, feature importance, model robustness, and context-aware telecom fault diagnosis.

The project combines network security research with practical network operations concepts, including traffic monitoring, fault detection, and root-cause investigation.

**Project status:** Active development — security classification experiments completed; hybrid telecom diagnosis under development.

---

## 1. Project Overview

Modern networks generate large volumes of traffic and operational telemetry. Detecting anomalies is important, but distinguishing malicious activity from operational failures remains a challenging problem.

This project investigates two related tasks:

1. **Network security anomaly detection:** Identify potentially malicious traffic, initially focusing on DDoS classification.
2. **Telecom fault diagnosis:** Investigate network degradation associated with congestion, packet loss, interface errors, routing instability, and link failures.

The longer-term objective is to evaluate whether combining traffic-flow features with operational network telemetry improves anomaly interpretation and fault diagnosis.

## 2. Research Objectives

- Build reproducible machine learning baselines for network traffic classification.
- Evaluate the importance and redundancy of network-flow features.
- Investigate how evaluation strategies influence reported performance.
- Assess model generalization using independent network datasets.
- Develop a separate telecom telemetry classification pipeline.
- Explore context-aware diagnosis using both security and operational indicators.

## 3. Dataset

### CIC-IDS2017

The initial security experiments use the Friday DDoS subset of CIC-IDS2017.

**Dataset:** [Canadian Institute for Cybersecurity — CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

The CSV contains:

| Property | Value |
|---|---|
| Total observations | 225,745 |
| Benign flows | 97,718 |
| DDoS flows | 128,027 |
| Initial model features | 78 |
| Classification task | Binary: BENIGN / DDoS |

The original dataset is not included in this repository. Users must obtain it separately and place the relevant CSV at `data/raw/ddos.csv`.

## 4. Methodology

### Baseline classification

The initial model uses a Random Forest classifier implemented with scikit-learn.

The pipeline includes:

- Numerical feature selection
- Infinite-value handling
- Missing-value imputation
- Stratified train/test splitting
- Random Forest classification
- Classification metrics and confusion matrix
- ROC-AUC evaluation

The initial baseline uses the first 100,000 observations and a stratified random 80/20 train/test split.

### Feature importance

Random Forest impurity-based feature importance is used to investigate which flow characteristics contribute to the classifier's predictions.

Prominent features include:

- `Init_Win_bytes_forward`
- `Bwd Packet Length Max`
- `Fwd Packet Length Mean`
- `Subflow Fwd Bytes`
- `Avg Fwd Segment Size`

Feature importance is not interpreted as proof of causality. Correlated features can also affect impurity-based importance rankings.

### Feature ablation

An ablation study removes groups of high-importance features and retrains the classifier.

The objective is to investigate whether performance depends strongly on a small subset of features.

### Block-holdout evaluation

A separate experiment trains on the first 175,000 observations and evaluates on the remaining 50,745 observations.

This tests performance on a different contiguous section of the same CSV.

**Important:** The CSV does not contain an explicit timestamp column. Therefore, this is a block-holdout experiment, not a verified chronological evaluation.

## 5. Experimental Results

### Experiment 1 — Random Forest baseline

| Metric | Result |
|---|---:|
| Test observations | 20,000 |
| Accuracy | 99.985% |
| ROC-AUC | 0.99999998 |
| False positives | 0 |
| False negatives | 3 |

These results are from a random split of the first 100,000 observations, not an independent external dataset.

### Experiment 2 — Feature ablation

| Experiment | Features | Accuracy | F1-score |
|---|---:|---:|---:|
| All features | 78 | 99.985% | 0.999877 |
| Remove top 1 | 77 | 99.980% | 0.999837 |
| Remove top 5 | 73 | 99.985% | 0.999877 |
| Remove top 15 | 63 | 99.985% | 0.999877 |

Removing the 15 highest-ranked features did not materially reduce performance under this evaluation setup.

This is consistent with substantial feature redundancy or relatively easy separation within the evaluated dataset. It does not establish generalization to unseen network environments.

### Experiment 3 — Block-holdout evaluation

| Metric | Result |
|---|---:|
| Training observations | 175,000 |
| Test observations | 50,745 |
| Accuracy | 99.740% |
| ROC-AUC | 0.99933 |
| False positives | 88 |
| False negatives | 44 |
| DDoS precision | 99.38% |
| DDoS recall | 99.69% |

**Confusion matrix**

| Actual / Predicted | BENIGN | DDoS |
|---|---:|---:|
| BENIGN | 36,609 | 88 |
| DDoS | 44 | 14,004 |

Performance remains high, although the block-holdout experiment produces more errors than the initial random-split baseline.

The experiments use different training and testing populations, so the difference cannot be attributed solely to the splitting strategy.

## 6. Current Limitations

Several limitations must be considered when interpreting these results:

1. **Single-capture evaluation:** The completed security experiments use observations from the same CSV.
2. **Potential dataset-specific patterns:** Very high classification accuracy may reflect characteristics of the capture rather than broad network generalization.
3. **Feature correlation:** Flow-level variables may contain redundant information.
4. **No verified chronological split:** Row order is not confirmed to represent chronological order.
5. **No independent external validation yet:** Generalization to different networks and attack scenarios remains untested.

These limitations motivate the next research phase.

## 7. Hybrid Telecom Fault Diagnosis — Work in Progress

The project is being extended to investigate operational network anomalies.

The planned telemetry features include:

| Feature | Operational relevance |
|---|---|
| Link utilization | Potential congestion |
| Latency | Network delay and degradation |
| Packet loss | Connectivity and service quality |
| Interface errors | Potential physical or link-layer problems |
| Route changes | Routing instability |

An initial synthetic telemetry generator is planned to validate the software pipeline.

The operational scenarios are:

- Normal operation
- Congestion
- Link degradation
- Routing instability
- Link failure

**Synthetic telemetry is intended for development and testing only.** It will not be treated as real-world validation.

Future experiments will use measured network telemetry and controlled fault scenarios.

## 8. Planned Evaluation

The next research stages include:

- Independent DDoS evaluation using compatible features from CIC-DDoS2019.
- Grouped evaluation across separate network captures or experimental runs.
- Telecom fault classification using measured telemetry.
- Comparison of flow-only, telemetry-only, and hybrid models.
- Evaluation of false-positive rates, per-class recall, macro F1-score, and detection delay.
- Investigation of interpretable evidence for incident diagnosis.

The final hybrid evaluation will require synchronized observations from the same network environment.

## 9. Repository Structure

```text
network-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train.py
│   ├── feature_importance.py
│   ├── ablation.py
│   ├── block_holdout.py
│   └── telecom/
│       ├── generate.py
│       └── train.py
├── reports/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── ablation_results.csv
├── data/
│   ├── raw/
│   └── simulated/
└── models/
```

Some files and directories belong to ongoing development and may not yet be available in the public repository.

## 10. Installation

Clone the repository:

```bash
git clone https://github.com/ah2062001/network-anomaly-detection.git
cd network-anomaly-detection
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the CIC-IDS2017 Friday DDoS CSV at:

```text
data/raw/ddos.csv
```

Run the baseline experiment:

```bash
python src/train.py
```

Run feature importance analysis after training:

```bash
python src/feature_importance.py
```

Run the ablation study:

```bash
python src/ablation.py
```

Run the block-holdout experiment:

```bash
python src/block_holdout.py
```

For the telecom module, once available:

```bash
python src/telecom/generate.py
python src/telecom/train.py
```

## 11. Research Direction

**Working research theme:**

*Context-Aware Network Anomaly Detection and Fault Diagnosis Using Flow-Level and Operational Telemetry*

The central hypothesis is that combining network-flow indicators with operational telemetry may improve fault diagnosis and reduce false alarms compared with using either source independently.

This hypothesis remains to be experimentally tested.

## 12. Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Random Forest · Network Traffic Analysis · Machine Learning · Telecom Fault Management

## 13. Author

**Ameer Hamza Hassan**

Electrical / Telecommunications Engineering background, with professional experience in telecom network operations, monitoring, and incident investigation.

Current research interests include computer networks, network security, intelligent fault management, and machine learning for network operations.

GitHub: [ah2062001](https://github.com/ah2062001)

---

*This repository is an evolving research and engineering project. Experimental findings are preliminary and should not be interpreted as evidence of deployment-ready detection performance.*