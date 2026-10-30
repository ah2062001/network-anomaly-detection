# Attack-Family Blind Spots in Cross-Dataset DDoS Detection: A Reproducible Audit of Feature and Decision-Score Shift

**Working manuscript — version 0.1, 30 October 2026**  
**Author:** Ameer Hamza Hassan  
**Affiliation:** Universiti Putra Malaysia (to verify for submission)  
**Status:** Preliminary empirical manuscript. This document is **not submission-ready**. Statements marked *planned* are hypotheses or analysis protocols, not completed findings.

## Abstract

Machine-learning detectors for distributed denial-of-service (DDoS) traffic frequently achieve near-perfect scores when training and evaluation samples originate from the same capture. Whether those results remain meaningful when attack mechanisms, traffic directionality, and flow-export pipelines change is less certain. This work presents an initial audit of a Random Forest detector trained on a 225,745-flow CIC-IDS2017 Friday DDoS subset and evaluated on a separate, redistributed 431,371-flow CIC-DDoS2019 CSV. Fifty-seven identically named, nonconstant numeric flow features were used with source-fitted median imputation. With the matched 57-feature representation, stratified internal evaluation on CIC-IDS2017 produced 99.9646% accuracy and 99.9649% attack recall. A separately fitted model trained on all 2017 flows achieved 25.3000% accuracy and only 3.4296% attack recall on the 2019 CSV, with 322,101 missed attacks and 133 false alarms. Crucially, all 11,439 detected attacks belonged to the SYN category: recall was 23.17% for SYN and 0% across 284,167 flows from the remaining reported attack categories. Large source–target differences also appeared in backward-packet availability and forward-packet lengths. These findings document a preliminary attack-family-specific transfer failure, but do **not** identify its cause: differences in feature generation, sampling, capture context, and decision-score calibration remain unverified. We therefore formulate a controlled research protocol separating feature compatibility, within-domain attack-family holdout, probability ranking, and source-only threshold transfer. This audit aims to identify which failure modes are measurable and which require a new held-out dataset before claims of improved generalization are justified.

**Keywords:** distributed denial of service; intrusion detection; cross-dataset evaluation; attack-family generalization; distribution shift; flow features; semantic harmonization; reproducibility.

## 1. Introduction

Flow-based network intrusion detection systems (NIDSs) classify traffic using aggregated quantities such as packet counts, packet lengths, inter-arrival times, flag counters, and forward/backward statistics. They are attractive because their inputs can often be collected without examining encrypted payloads. In an operational setting, however, the distribution encountered by a detector may differ from its development benchmark: transport protocols, attack tools, clients, capture placement, exporters, and background services may all change. A high score on a randomized within-dataset split is consequently not evidence that a trained classifier will remain useful on an independently acquired capture.

This is not a new criticism. D'hooge et al. [1] showed that supervised methods with strong performance on CIC-IDS2017 and CSE-CIC-IDS2018 often fail to generalize across the two datasets, including for related attack classes. Their later study [2] extended the assessment to CIC-DDoS2019 and found both conspicuous failures and pockets of successful generalization, especially among network-centric attacks. Therefore, the simple claim that 'within-dataset accuracy does not guarantee cross-dataset accuracy' cannot constitute the novelty of this work.

More recent work further restricts the potential contribution. Elangovan et al. [3] studied temporal and semantic generalization across enterprise and IoT benchmarks; Jin and Lee [4] distinguished ranking, calibration, threshold transfer, and score-orientation failures; Berhili et al. [5] quantified the influence of evaluation protocols and applied semantic alignment; Zekiouk et al. [6] used behavior-driven hierarchical classification and multi-dataset enrichment including CIC-DDoS2019; and Patil and Deore [7] constructed a 15-semantic-feature benchmark with SHAP and directional-transfer analysis. Hence, merely proposing feature harmonization, hierarchy, SHAP, or a calibrated threshold would substantially overlap with published work.

Our initial investigation began as an attempt to transfer an ordinary Random Forest detector from CIC-IDS2017 to CIC-DDoS2019. The marked loss of attack recall suggested that a finer-grained characterization was warranted. The strongest descriptive observation is not overall accuracy collapse, but a highly concentrated *attack-family blind spot*: some SYN flows were recognized, whereas every flow from the other reported attack categories was predicted benign in the examined target CSV.

The overarching question of this work is: **Under a documented feature interface and source-only decision policy, which measurable aspects of cross-dataset change coincide with the failure to recognize unseen DDoS mechanisms?** A secondary, prospective question is whether source-selected representations or source-only alert policies can reduce the failure on a genuinely unused confirmation dataset. The present version answers the first question only descriptively and proposes experiments required to investigate the rest.

### 1.1 Research questions

**RQ1 (observed):** How large is the internal-to-external degradation when the same 57-feature representation and Random Forest configuration are used?

**RQ2 (observed descriptively):** How does attack recall vary by target attack label and backward-traffic characteristics?

**RQ3 (planned):** Is the target failure primarily a loss of ranking ability, a failure of source-derived threshold transfer, or both?

**RQ4 (planned):** How much of the gap can be attributed to identifiable feature-definition and feature-distribution incompatibilities, rather than unseen attack family alone?

**RQ5 (planned, conditional):** Can a prespecified source-only intervention improve per-family recall under a fixed false-alarm budget on an independent, previously unused evaluation dataset?

### 1.2 Scope and contribution boundary

The presently supported contribution is an **auditable case study**: a matched-feature internal/external Random Forest evaluation, class-level error analysis, and descriptive feature-shift audit for one specific source/target CSV pair. It does not establish causal attribution, general superiority of a model, or novelty of the transfer-failure phenomenon. A stronger final contribution would require reproducible failure-mode decomposition and independent confirmation; such results are not yet available.

## 2. Related Work

### 2.1 Cross-dataset transfer and attack-family generalization

D'hooge et al. (2020) [1] evaluated 12 supervised learners on CIC-IDS2017 and CSE-CIC-IDS2018, then examined transfer for overlapping DoS and botnet-related attack classes. Their results undermined the premise that robust intradataset results imply reliable interdataset performance, while also showing dependence on feature scaling and classifier choice. The present study does not replicate their full method comparison or their bidirectional transfer design.

D'hooge et al. (2023) [2] is the closest conceptual predecessor. Models trained on CSE-CIC-IDS2018 were tested on related classes from CIC-IDS2017, CIC-DoS2017 and CIC-DDoS2019. Their CIC-DDoS2019 section analyzes reflection- and exploitation-oriented attack variants, reporting both highly variable tree-model behavior and rare strongly generalizing models. As the authors emphasize, choosing only the top performers conceals instability across models and data reductions. Our 2017→2019 training direction and current redistribution differ from theirs, but neither the direction nor the target benchmark alone guarantees novelty. A final paper must explicitly compare corresponding attack families and preprocessing to this study.

### 2.2 Evaluation-protocol bias

Berhili et al. [5] systematically studied accuracy inflation due to validation-tail selection, random seeds, and preprocessing procedures on UNSW-NB15, NSL-KDD, and CIC-IDS2017. They reported large validation-protocol effects on two benchmarks and semantic feature alignment in cross-benchmark transfer. This motivates reporting exact training indices, software versions, preprocessing fit boundaries, and uncertainty rather than comparing headline accuracy values from incompatible protocols. Importantly, their conclusions concern measured mechanisms under their protocol, not a claim that every near-perfect CIC-IDS2017 score results from leakage.

### 2.3 Decision scores, thresholds and operational alerting

Jin and Lee [4] evaluate strict source-only transfer among HIKARI-2021, CICIDS2017, and a CICIoT2023 subset using Rich-64 and Minimal-13 feature sets. They distinguish precision–recall ranking from performance at source-derived thresholds and examine orientation inversion diagnostically. For HIKARI→CICIDS2017, their strict Rich-64 tests report zero recall at the transferred source FPR threshold for several models; such outcomes need not have the same underlying explanation. We must therefore compute target ROC-AUC, average precision, score quantiles and false alarms at thresholds fixed without target labels before attributing our low recall to absent discriminative signal.

### 2.4 Temporal/domain variation and semantic stability

Elangovan et al. [3] evaluate Random Forest, gradient boosting, MLP, autoencoder, and 1D-CNN across datasets spanning 2017–2024. Their analysis associates more stable generalization with certain flow-intensity and temporal feature groups, but transfer conditions, sample composition and attack taxonomies are not directly identical to ours. A new 'stable feature subset' claim would therefore require clear separation from their semantic-feature comparison.

### 2.5 Multi-dataset training and behavioral groupings

Zekiouk et al. [6] propose BHM-IDS, which trains using CIC-IDS2017 supplemented by CIC-DDoS2019 and evaluates on CSE-CIC-IDS2018. The system combines binary detection, behavior-based grouping, and specialized attack classification. It includes ablations isolating multi-dataset and hierarchy components. This demonstrates that a generic 'combine 2017 and 2019 and group attack types' approach has already been investigated. Our planned within-family holdout would target diagnostic attribution rather than claiming that grouping itself is new.

### 2.6 Semantic mapping and missing-feature artifacts

Patil and Deore [7] align CICIoT2023, CIC-DDoS2019, and TON-IoT into 15 canonical traffic semantics and evaluate ML, DL, and graph approaches with SHAP inspection. They explicitly distinguish legacy target-feature imputation from strict source-training-only preprocessing, note that structural zero filling can create artificial domain cues, and show instances where ROC-AUC remains high even though F1 falls sharply. Their GNN comparisons use behavioral kNN graphs rather than measured network topology. This study makes it especially important that our feature mapping be verified semantically—not just matched by column name—and that the final manuscript distinguish operational and diagnostic uses of target data.

### 2.7 Synthesis of the literature gap

The seven reviewed papers establish that transfer can fail, that some network-centric attacks *can* transfer, that evaluation choices and schema mismatches matter, and that threshold and ranking failure must be distinguished. The *possible* narrower opportunity is a **prespecified, mechanism-level failure attribution protocol for the 2017→2019 case**, supported by verified feature definitions, score diagnostics, attack-family held-out tests, and confirmation on independent data. This is a candidate positioning, not a verified unoccupied gap. Additional focused review of recent methods and datasets is required before a novelty claim can be made.

## 3. Materials and Methods

### 3.1 Study design and reporting status

We separate completed analyses from planned follow-up experiments. The initial analysis uses a single transferred binary classifier, descriptive attack-family stratification, and marginal distribution summaries. The subsequent protocol (Section 3.7) is prospective and must not be represented as already executed. Results from a target dataset whose labels have been examined can generate hypotheses, but cannot be treated as an untouched final confirmatory evaluation if later models or decisions are informed by those observations.

### 3.2 Data sources and provenance

**Source (CIC-IDS2017 subset):** `data/raw/ddos.csv`, 225,745 rows and 79 columns in the local copy, with 128,027 labeled `DDoS` and 97,718 `BENIGN`. It is the previously downloaded Friday DDoS CSV. The exact source download URL, checksum, processing history, original exporter version and relationship to official released data must be documented before submission.

**Target (redistributed CIC-DDoS2019):** `data/raw/cicddos2019/cicddos2019_dataset.csv`, 431,371 rows and 80 columns in the local copy, labeled `Attack` (333,540) and `Benign` (97,831) through the `Class` field. Its `Label` field contains finer traffic types, including SYN, TFTP, UDP, amplification/reflector labels, and benign. This redistribution is **not established to be the full official CIC-DDoS2019 release**. Sampling, feature-generation details, cleaning, and provenance require independent verification. A `Protocol` field and an `Unnamed: 0` field were present in this copy; neither was used as a predictor.

**Proposed confirmation:** CSE-CIC-IDS2018 is a possible independent validation dataset, but it must be evaluated only after a fixed model/policy and analysis plan are established. Its class definitions and DDoS scenarios should be mapped and documented. Merely adding a third dataset is not sufficient unless its intended role is explicit.

### 3.3 Data-quality audit

The complete 2017 CSV contained 2,633 exact duplicate rows, four NaN entries in `Flow Bytes/s`, 64 infinite numeric values, and ten constant columns in a first pass. No explicit timestamp, Flow ID, source IP or destination IP was identified by the audit script. A diagnostic using the first 100,000 rows found 729 exact duplicates; after dropping them the random-split accuracy changed from 99.985% to approximately 99.9799%, under nonidentical partitions. This comparison is descriptive only and does not isolate the causal effect of duplicates. None of these facts proves or rules out train/test leakage in the reported baseline.

Before final evaluation, a frozen data-cleaning manifest must describe exact duplicate handling, numerical coercion, infinite-value replacement, constant-feature treatment, label mapping, and how any duplicate or near-duplicate flows crossing train/test boundaries are addressed. Raw files should remain immutable and their cryptographic hashes recorded.

### 3.4 Preliminary feature interface

An initial header comparison identified 67 identically named numeric-flow feature candidates across the two CSVs, without establishing equivalence of measurement semantics. The experiment excluded label fields, identifiers, `Protocol`, and `Unnamed: 0`. It then retained 57 nonconstant numeric columns using variation observed in the 2017 source. This 57-column interface was applied unchanged in both evaluations. Infinities were replaced with missing values and a median imputer was fitted on the model's training data only.

In particular, the label `Class` supplied the binary target in the 2019 CSV, and `Label` supplied the attack subtype; neither entered feature matrices. The output `Common usable features: 57` records implementation behavior, not independent validation of the 57 measurement definitions. Canonical feature alignment, exporter version checks, unit tests on representative packet captures, and a per-feature equivalence table remain necessary. No time-based split can be claimed from the current CSV because it lacks verified timestamp information.

### 3.5 Classifier and split configuration

A scikit-learn `RandomForestClassifier` used 100 trees, `class_weight='balanced'`, `random_state=42`, and `n_jobs=-1`, preceded by median imputation. The internal experiment performed a stratified, randomized 80/20 split of CIC-IDS2017 (`random_state=42`). Its internal model was fitted on 80% of the 2017 rows. The external experiment fitted a **separate instance** of the same classifier on all 225,745 2017 rows and predicted the entire 431,371-row 2019 CSV without fitting to 2019 labels or altering the learned model.

Thus, the feature interface and hyperparameters matched, but the internal and external numbers were produced by different fitted models and training sizes. A future matched-model protocol must use one source-trained model and evaluate it both on its source-held-out partition and a fixed external dataset; only after that comparison may the source training set be expanded for a separate final fitted model.

### 3.6 Measures and definitions

Let attack be the positive class (1), with true positives TP, false negatives FN, false positives FP, and true negatives TN. Attack recall = TP/(TP+FN); precision = TP/(TP+FP); F1 = 2PR/(P+R); specificity = TN/(TN+FP); balanced accuracy = (recall + specificity)/2. We also report per-attack-family recall and absolute counts, because overall accuracy can conceal failure on a heavily attack-weighted target. Preliminary tests used the classifier's default decision policy; target ROC-AUC, average precision, calibration and fixed-budget alert metrics remain uncomputed.

Where possible, evaluation should later report confidence intervals from repeated source seeds and suitable capture/run-level resampling. Treating adjacent flows as iid units may produce deceptively narrow intervals; the current datasets do not expose verified run identifiers for cluster bootstrap.

### 3.7 Prespecified follow-up diagnostic protocol (not yet executed)

**Stage A: Freeze provenance and matched-model baseline.** Save SHA-256 checksums; dataset source and transformation records; a semantic crosswalk of retained features; code and package versions; source train/validation/holdout indices; source-only cleaning and imputation rules; one frozen fitted model; and per-record predictions. Test that the 2017 held-out and 2019 target predictions are generated by the identical model. Separately audit duplicates and identifier-like shortcuts.

**Stage B: Separate ranking failure from threshold failure.** Save `predict_proba` outputs. Compute internal and external ROC-AUC, average precision, Brier score and reliability summaries; draw positive/negative score histograms and source/target score quantiles. Choose every operational threshold on a source validation partition only, including a threshold targeting a prespecified source false-positive rate. Apply it without adjustment to the target. Thresholds optimized on target labels, as well as flipping score orientation using target labels, may be shown *only* as clearly marked nondeployable diagnostic upper bounds.

**Stage C: Audit measurement semantics and feature shift.** Verify feature names against definitions, direction conventions, units, and extractor versions; explicitly distinguish true zeros from absent or undefined measurements. Quantify shifts using train-fixed bins/quantiles, zero-fraction differences, robust effect sizes, and feature-level support overlap. Do not interpret a large marginal median difference as proof of a causal explanation of model errors. Link source-trained feature importance or SHAP summaries to error cohorts, while noting that attribution is not causation.

**Stage D: Isolate family shift where feasible.** Construct within-domain leave-one-attack-family-out evaluations using a dataset with several attack families and comparable benign captures. Keep each held-out family out of training, preprocessing selection, validation and threshold optimization; ensure the relevant benign reference group is also out of training when its capture/run would otherwise leak. Evaluate both per-family attack recall and benign false-alarm burden. This is a different experiment from the original 2017→2019 transfer and must be reported as such.

**Stage E: Test one source-only intervention against frozen baselines.** Candidate interventions include a prespecified traffic-behavior feature subset, an invariant-feature selection method that uses only source sub-environments, or a source-only threshold policy. Select only one after benchmarking simpler controls. Compare with full-feature Random Forest, logistic regression, and a boosting baseline under identical splits. Report absolute attack-recall gains together with precision/FPR and per-family trade-offs. No improvement claims can be included until results are measured.

**Stage F: Untouched confirmation.** Because the 2019 subtype labels and errors have already influenced project decisions, treat that CSV as *diagnostic/development evidence* for future iterations. Freeze the proposed method first, then evaluate it on a separate held-out capture/dataset with compatible labels and feature definitions. The confirmation stage must not use target labels for selection, fitting, calibration, or thresholds.

## 4. Preliminary Results

### 4.1 Internal and external performance

**Table 1. Random Forest outcomes using a common 57-feature interface.**

| Metric | Source internal (80/20 stratified split) | External target (2019 CSV) |
|---|---:|---:|
| Evaluated flows | 45,149 | 431,371 |
| Accuracy | 99.9646% | 25.3000% |
| Attack precision | 99.9727% | 98.8507% |
| Attack recall | 99.9649% | 3.4296% |
| Attack F1 | 99.9688% | 6.6292% |
| TN | 19,537 | 97,698 |
| FP | 7 | 133 |
| FN | 9 | 322,101 |
| TP | 25,596 | 11,439 |

The external false-positive rate was 133/97,831 = 0.1359%, and attack specificity was 99.8641%. Balanced accuracy was approximately 51.6468%. External attack prevalence was 77.32%; a degenerate model predicting every target flow as attack would achieve about 77.32% accuracy but would generate 97,831 benign false alarms. Conversely, predicting everything as benign would achieve 22.68% accuracy and zero attack recall. These comparisons reinforce why accuracy and precision alone are insufficient.

The internal-to-external attack recall gap is approximately 96.5353 percentage points. This is a descriptive comparison across independently fitted models; it is not an estimate of a single model's causal degradation due to any one shift factor.

### 4.2 Concentration of detections in SYN traffic

**Table 2. External attack detection by broad type, from the 2019 `Label` field.**

| Target group | Attack flows | Detected | Attack recall |
|---|---:|---:|---:|
| SYN | 49,373 | 11,439 | 23.1685% |
| All other reported attack types | 284,167 | 0 | 0.0000% |
| Total attacks | 333,540 | 11,439 | 3.4296% |

The zero-recall group includes TFTP (98,917), DrDoS_NTP (121,368), UDP (18,090), and all the other non-SYN attack labels listed in the input audit. These are **flow-level findings for this particular redistributed CSV**, not evidence that the model can never detect such attack mechanisms elsewhere. The observation motivates family-conditioned diagnosis, but the family labels are also correlated with protocol and source dataset construction.

### 4.3 Backward-flow characteristics

**Table 3. External attack detection stratified by backward-packet count.**

| Backward-packet group | Attack flows | Detected | Attack recall |
|---|---:|---:|---:|
| Zero backward packets | 290,966 | 9,297 | 3.1952% |
| At least one backward packet | 42,574 | 2,142 | 5.0312% |

The 1.84-percentage-point difference suggests some association with traffic directionality, but it is much smaller than the SYN-versus-other-family distinction. A backward-packet count cannot be isolated from attack type, protocol, extraction conventions, and capture geometry using these marginals. A two-way attack-family × backward-count analysis with prediction scores is required before interpreting the mechanism.

### 4.4 Marginal feature shifts

**Table 4. Selected attack-class median and zero-fraction shifts.**

| Feature | 2017 attack median | 2019 attack median | Additional observation |
|---|---:|---:|---|
| `Total Backward Packets` | 4 | 0 | Zero fraction: 36.34% vs. 87.24% |
| `Bwd Packet Length Mean` | 1,934.5 | 0 | Zero fraction: 36.34% vs. 88.18% |
| `Bwd Header Length` | 92 | 0 | Zero fraction: 36.34% vs. 87.25% |
| `Fwd Packet Length Mean` | 7 | 440 | Zero fraction: 0% vs. 2.40% |
| `Flow Bytes/s` | 160.10 | 13,215.697 | Large median separation |

These comparisons support the presence of substantial **observed feature distribution differences**. They do not establish whether the differences arise from attack mechanisms, background traffic, clipping and extraction logic, or dataset sampling. In particular, shared feature names do not yet guarantee shared measurement semantics.

### 4.5 Data-quality sensitivity

The first 100,000-row source subset contained 729 exact duplicate records. The initial random-split test yielded 99.985% accuracy; the deduplicated and resplit diagnostic yielded 99.9799%. The tiny numerical difference is insufficient to conclude that duplicate records had no impact, because the train and test partitions changed simultaneously. A truly controlled experiment should maintain a common held-out test set and prohibit exact feature-vector matches across the training/test boundary.

## 5. Discussion

### 5.1 Interpretation and alternatives

The findings indicate that the specific source-trained Random Forest transfers poorly to the redistributed target and that recognized attack flows are exclusively from the target's SYN category. One plausible interpretation is attack-family mismatch: the source DDoS scenario may not span the diverse flooding and amplification patterns in the target CSV. The observed backward-traffic and packet-length shifts are compatible with that interpretation, but they do not distinguish it from changed exporters or capture conditions. In particular, reflection/amplification traffic can have directional flow signatures quite different from a bidirectional application exchange; whether that explains the observed false negatives requires matched feature semantics and a family-controlled evaluation.

A second interpretation is score-threshold instability. The detector may rank attack flows above benign flows but assign most predicted attack probabilities below its default threshold. This possibility is explicitly raised by the calibration and transfer results in [4,5,7]. Until the model's prediction scores are inspected, declaring a total absence of transferable signal would be unjustified.

A third interpretation is data artifact or shortcut learning. Strong within-capture performance may be supported by a small collection of dataset-specific packet-length, flag, or flow-duration patterns. Our initial feature-ablation tests showing high source accuracy after removing top features are not evidence of domain-invariant representation: correlated substitutes may remain. Conversely, feature importance calculated on source labels does not show which features explain the target failures.

### 5.2 Comparison with published evidence

The findings are consistent with the broad warning from D'hooge et al. [1,2] but do not reproduce their exact training direction, target sampling, class balance, or model-selection procedure. Zekiouk et al. [6] already address attack coverage through multi-dataset training and behavioral hierarchy; consequently, combining 2017 and 2019 labels into one training dataset cannot be claimed as a new solution. Patil and Deore [7] already demonstrate semantic feature alignment and SHAP on CIC-DDoS2019, while Jin and Lee [4] explicitly separate ranking and operational threshold failure. The manuscript's potential added value must therefore come from **carefully controlled error attribution and an independently validated targeted intervention**, not from the use of these components separately.

### 5.3 Operational interpretation

A classifier with external precision of 98.85% and FPR of 0.136% could superficially appear attractive for an alerting pipeline, but it misses more than 322,000 attacks in this evaluation. This underscores the need to report attack recall and missed-attack counts alongside false alarms, particularly under extreme class imbalance. Conversely, lowering the decision threshold without controlling alert volume could flood security operations teams with false positives. A source-fitted fixed-budget analysis would permit deployment-relevant comparisons without tuning on hidden target labels.

### 5.4 Threats to validity

**Construct validity:** the source CSV represents one Friday DDoS scenario; the target redistribution contains multiple attack families with incompletely verified collection and processing provenance. The two positive-class distributions are not equivalent in mechanism coverage.

**Measurement validity:** 57 matching column names are not proof of consistent flow direction, time units, handling of malformed flows, or CICFlowMeter semantics. An `Unnamed` index was excluded but other artifact-like fields must still be reviewed.

**Internal validity:** the internal and external results use different fitted instances and different source training sizes. The internal split is randomized, not a verified temporal or capture-disjoint split; exact and near-duplicate leakage has not been fully controlled. Marginal feature comparisons are associative and confounded.

**Target contamination and multiple comparisons:** subtype errors and feature shifts from the 2019 target have now been inspected. Subsequent interventions selected based on these labels cannot use that same CSV as an unbiased final generalization test. A reserved confirmation capture is necessary.

**Statistical validity:** a single seed, one model family, correlated network flows, and point estimates without run-level intervals constrain reliability. Per-flow bootstrap intervals would understate uncertainty if flows are correlated.

**External validity:** conclusions are limited to the examined CSV versions and the chosen DDoS/benign label mapping. The study does not demonstrate performance in an operational production network or under novel live attacks.

**Publication novelty:** multiple prior studies already document cross-dataset failure, feature shift, score calibration, and mitigation strategies. A paper consisting only of the present baseline tables would be an incremental replication unless accompanied by a distinctive, independently supported analysis or intervention.

## 6. Planned Confirmatory Experiments and Decision Criteria

**Experiment E1 — Matched-model performance and score audit.** A single source-trained classifier predicts both the held-out 2017 partition and 2019 CSV; all model scores and row IDs are saved. Report ROC-AUC, average precision, precision–recall curves, confidence diagnostics, and source-selected threshold transfer. **Success criterion:** determine whether ranking signal exists independently of the 0.5 threshold. No performance improvement is presupposed.

**Experiment E2 — Feature-semantic crosswalk.** Each selected feature is annotated with source/target name, definition, units, direction convention, missingness interpretation, exporter evidence, and an equivalence status. Exclude or quarantine unverified features and rerun the fixed model protocol. **Success criterion:** a defensible comparable feature interface, not a target-tuned accuracy gain.

**Experiment E3 — Family-versus-environment separation.** Implement leave-one-family-out evaluation on a multi-family source dataset, with capture/run isolation where metadata allows; compare it with fixed-family cross-environment transfer. **Success criterion:** provide evidence as to which changes accompany which failures, without making unsupported causal claims.

**Experiment E4 — One intervention plus negative controls.** Choose a proposed source-only solution after E1–E3: either a robust feature group, a prespecified source-validation alert threshold, or another narrowly formulated remedy. Benchmark against unmodified RF, logistic regression, and boosting under the same conditions. Control for simpler explanations by evaluating equal-sized random feature subsets and source-only tuning alternatives. **Success criterion:** replicate a family-level recall improvement while maintaining a predefined external false-positive budget, with results reported per group.

**Experiment E5 — Independent confirmation.** Freeze all choices and test on untouched, provenance-verified CSE-CIC-IDS2018 or another suitable capture. **Success criterion:** independently confirm the main claimed benefit with uncertainty, error counts, and a documented failure analysis. If this is not achieved, the manuscript should remain a descriptive reproducibility case study instead of claiming a transferable method.

### 6.1 Mandatory reproducibility artifacts

Retain dataset checksums, raw/cleaned transformation manifests, model versions and configuration, train/validation/test indices, canonical feature crosswalk, fitted preprocessing/model objects, per-flow scores and predictions, confusion matrices by label, figures (including score distributions), an experiment register, and versioned code. The external datasets should not be committed wholesale to the public Git repository unless licenses allow it; prefer scripts, checksums, provenance records, and retrieval instructions.

### 6.2 Planned tables and figures

**Figure 1 (to create):** Experimental pipeline showing source-only training and the strict separation between diagnostic 2019 data and confirmation data.

**Figure 2 (to create):** Internal/target score distributions by true class; fixed source threshold shown without target optimization.

**Figure 3 (to create):** Per-attack-family recall and sample counts for target; emphasize zero-recall families.

**Figure 4 (to create):** Forward/backward packet and rate distributions by source/target and attack group, with units and export version validation.

**Table 5 (to create):** Semantic crosswalk for all retained features.

**Table 6 (to create):** Source-only threshold transfer and ranking metrics.

**Table 7 (to create):** Multi-seed model comparison and intervention ablation on reserved confirmation data.

## 7. Conclusion

In this preliminary study, a Random Forest using 57 identically named flow features retained high randomized within-source performance but failed to identify most attacks in a redistributed CIC-DDoS2019 evaluation. The failure was strongly associated with attack label: all detected attacks belonged to the SYN category, and other target attack categories exhibited zero recall. Several flow features also differed markedly in distribution across the two datasets. The evidence does not yet establish whether the main cause is unseen attack mechanism, feature-extraction incompatibility, dataset construction, or transferred decision thresholds. Prior work already establishes that cross-dataset degradation is common and that semantic alignment, calibration analysis, and multi-dataset training are important considerations. Accordingly, the main prospective scientific task is to separate these explanations in a reproducible, source-only protocol and test any proposed remedy on independently reserved data. Until that work is complete, the findings should be presented as an exploratory case study rather than a proven new generalizable DDoS detection method.

## References

[1] L. D'hooge, T. Wauters, B. Volckaert, and F. De Turck, “Inter-dataset generalization strength of supervised machine learning methods for intrusion detection,” *Journal of Information Security and Applications*, vol. 54, 102564, 2020. DOI: 10.1016/j.jisa.2020.102564.

[2] L. D'hooge, M. Verkerken, T. Wauters, F. De Turck, and B. Volckaert, “Investigating Generalized Performance of Data-Constrained Supervised Machine Learning Models on Novel, Related Samples in Intrusion Detection,” *Sensors*, vol. 23, no. 4, 1846, 2023. DOI: 10.3390/s23041846.

[3] R. Elangovan et al., “Cross-Dataset Temporal and Semantic Generalization of Intrusion Detection Models for the Future Internet,” *Future Internet*, vol. 18, no. 4, 194, 2026. DOI: 10.3390/fi18040194.

[4] H. Jin and H. Lee, “Benchmark-Shift-Aware Intrusion Detection for Evolving Network Traffic: Cross-Dataset Generalization, Calibrated Alerting, and Score-Orientation Diagnostics,” *Electronics*, vol. 15, no. 17, 3761, 2026. DOI: 10.3390/electronics15173761.

[5] M. Berhili, O. Chaieb, and M. Benabdellah, “How evaluation protocol choices inflate intrusion detection accuracy: A three-dataset systematic study,” *Computers & Security*, vol. 173 (2027), 105218; published online 24 October 2026. DOI: 10.1016/j.cose.2026.105218.

[6] M. Zekiouk et al., “BHM-IDS: Behavior-Driven Hierarchy and Multi-Dataset Training for Cross-Dataset Generalization,” *Applied Sciences*, vol. 16, no. 16, 7885, 2026. DOI: 10.3390/app16167885.

[7] V. T. Patil and S. S. Deore, “A cross-dataset benchmark for IoT DDoS detection with semantic feature alignment and SHAP,” *Computers and Electrical Engineering*, vol. 141 (2027), 111606; published online 12 October 2026. DOI: 10.1016/j.compeleceng.2026.111606.

---

## Editorial and experiment integrity checklist (not part of submission)

- [ ] Verify the source URL, version, SHA-256 and acquisition date for both CSVs.
- [ ] Resolve whether CIC-IDS2017 features are produced by the same CICFlowMeter implementation and unit definitions as the redistributed CIC-DDoS2019 features.
- [ ] Preserve the exact list of 57 columns in source-code version control.
- [ ] Check labels, duplicates, any conflicting feature-identical observations, and train/test overlaps.
- [ ] Repeat internal/external evaluation with a single fitted model, frozen source-only preprocessing and kept source split.
- [ ] Save per-row external probability scores and compute ROC-AUC, AP, rank lift, and source-only threshold transfer.
- [ ] Explicitly account for 2019 being used for exploratory hypothesis generation; reserve a separate final confirmation test.
- [ ] Avoid using 'causal', 'novel', 'robust', 'deployment-ready', 'state of the art', and 'significant improvement' until justified by new controlled data.
- [ ] Confirm all bibliographic metadata, citation style, author contribution and affiliation before journal submission.
- [ ] Add uncertainty, multi-seed results, and a clear experiment registration.
- [ ] Complete and independently verify all planned results/figures; remove or clearly identify placeholders.
