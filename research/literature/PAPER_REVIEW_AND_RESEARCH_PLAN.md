# Critical Literature Review and Research Plan
## Cross-dataset DDoS detection: CIC-IDS2017 to CIC-DDoS2019

**Prepared:** 30 October 2026  
**Scope:** Critical reading of four user-provided papers (72 PDF pages in total), comparison to project experiments supplied in the conversation, and a separately marked brief search for closely overlapping research.  
**Status:** Research planning and evidence audit. No novel method or successful mitigation has yet been established. Do not use this document verbatim as a submitted literature review without checking original tables, citations, and current literature.

---

## Executive conclusions

1. **The observed generalization failure is real for the evaluated pipeline, but the phenomenon itself is not novel.** D'hooge et al. (2020), Berhili et al. (2026 online / 2027 issue), Jin & Lee (2026), and Elangovan et al. (2026) already examine poor cross-dataset generalization and evaluation shortcuts.
2. **Our attack-family observation is more specific:** our CIC-IDS2017 DDoS-trained Random Forest recognizes 11,439/49,373 SYN attack samples in a redistributed CIC-DDoS2019 subset and 0/284,167 other attack samples. This is a useful diagnostic, *not proof of attack-family shift as the cause*.
3. **The first critical missing experiment is ranking-versus-threshold diagnosis.** External attack precision is 98.85% but recall is 3.43%. We must measure ROC-AUC, average precision and prevalence-relative AP, score histograms by class/attack family, and source-validated threshold behavior. Jin & Lee have already established why this separation matters; it would be a reproduction/diagnosis, not a standalone novelty claim.
4. **Feature-schema equality is not feature-semantic equality.** Source and target share 67 names, of which 57 survive source-side nonconstant filtering. Both papers and CICFlowMeter debugging documentation warn about flow-direction, labeling, bug fixes, and data-processing artifacts. Retain provenance and a per-feature compatibility ledger before interpreting causes.
5. **Search beyond the four PDFs revealed a direct 2023 predecessor** (D'hooge et al., Sensors 2023, DOI 10.3390/s23041846) testing transfer involving CIC-DDoS2019; and a 2026 BHM-IDS paper using CIC-IDS2017/CIC-DDoS2019. Their full texts must be read before asserting novelty. Another 2026/2027 benchmark explicitly includes CIC-DDoS2019 and semantic alignment.
6. **Best near-term research option:** rigorously isolate *what drives attack-family transfer failure* and test a pre-specified, operationally feasible remedy under fully held-out data. If these validations fail or yield only known observations, publish an honest focused reproducibility/benchmark note only if the venue accepts that contribution, or switch to a different original research question.

---

## A. The four provided papers — detailed evidence and critique

### P01 — D'hooge, Wauters, Volckaert & De Turck (2020)
**Title:** *Inter-dataset generalization strength of supervised machine learning methods for intrusion detection*.  
**Venue:** Journal of Information Security and Applications 54, article 102564.  
**DOI:** https://doi.org/10.1016/j.jisa.2020.102564  
**Priority:** Foundational; very close to our problem.

**Research question:** Does excellent performance on CIC-IDS2017 or CSE-CIC-IDS2018 remain good when a pretrained binary classifier is evaluated on related attacks from the other dataset without retraining?

**Methods (pp. 3–5, 9–12):** 12 supervised learners spanning seven tree-based methods, nearest-neighbor methods, SVMs and logistic regression; experiment subsets by attack scenario, plus a merged 2017 dataset. Data were cleaned, labels binarized, and model parameters tuned with stratified 5-fold CV on training folds (two-thirds of relevant day data for cross-validated parameter selection). They compared no scaling, min–max scaling and standardization/normalization. Cross-dataset investigation concentrated on DoS/SSL and botnet attacks in both directions; held-out target observations were not used for retraining.

**Findings:** Tree-based learners often achieve near-perfect in-dataset detection, but transfer is fragile. On 2018→2017 DoS subset 1, stronger methods reached balanced accuracy roughly 80–85% in some settings, with tree-based attack recall around 50%; many configurations were near chance or had zero recall. On some 2017→2018 tests, no-scaling methods had F1 near zero. Botnet transfer was asymmetric: one direction remained unreliable despite both datasets including ARES traffic, while a linear SVM in the other direction reached about 90% balanced accuracy (F1 ~82%) under favorable scaling. See Figures 7–12 (pp. 9–11) and conclusion (p. 12). **Do not summarize this paper as proving that every algorithm always fails; some combinations performed reasonably well.**

**Explicit limitations:** Tested a small set of overlapping attack families; heterogeneous collection and benign behaviors could matter; requested further work distinguishing data problems from model problems. Also recognizes CICFlowMeter duplicate forward-header-length and literal infinity in CIC-IDS2017 (p. 3).

**Overlap with us:** Directly establishes in-domain ≠ out-of-domain accuracy; testing another pair of CIC datasets with Random Forest and observing collapse is insufficient novelty.

**Potential distinction:** Explain attack-family-specific failures in CIC-DDoS2019 using controlled feature semantics and test a remediation with independent capture/family holdouts. But do not call attack-family analysis unprecedented until reading the 2023 follow-up.

**Detailed accuracy caution:** The printed Table 1 (p. 3) labels a row with 97,718 benign + 128,027 attack as “Nmap port scanning,” and a row with 127,537 benign + 158,930 attack as “Layer 4 DDoS.” These counts match the common public Friday DDoS and PortScan CSV distributions in the opposite order. This appears to be a row-label swap or dataset-description error; verify against original CSV filenames/labels before reusing Table 1 categorization. It does not undermine the paper's central generalization finding by itself.

### P02 — Berhili, Chaieb & Benabdellah (online Oct 2026; 2027 issue)
**Title:** *How evaluation protocol choices inflate intrusion detection accuracy: A three-dataset systematic study*.  
**Venue:** Computers & Security, volume 173 (2027), article 105218; accepted 17 Oct 2026, online 24 Oct 2026.  
**DOI:** https://doi.org/10.1016/j.cose.2026.105218  
**Code:** https://github.com/mberhili/ids-evaluation-protocol-bias  
**Priority:** Highest methodological standard to emulate; highly overlapping narrative.

**Research questions (pp. 2–4):** Effect of validation protocol, random seed, and preprocessing leakage; genuine transfer under semantic alignment; contribution of source-calibrated versus test-label-oracle threshold/ensemble optimization.

**Methods:** UNSW-NB15 (official split), NSL-KDD (official split), CIC-IDS2017 (500k stratified flows, 80/20 random split). Bias measurements use XGBoost, with SMOTE on training only. AWE-IDS combines XGBoost/LightGBM; tuning on a source-held-out partition is distinguished from **oracle tuning on target test labels** (not deployable). Cross-dataset semantic alignment uses UNSW-NB15 as a pivot, 14 features for UNSW↔CIC2017 and 10 for UNSW↔NSL; NSL↔CIC2017 deemed too semantically incompatible (only three strictly equivalent features). Experiments use multi-seed comparisons and ablations (pp. 4–6, 10–11).

**Principal findings:** Table 11 (p. 11), mean ± SD across 3 seeds: deployable AWE-IDS UNSW→NSL 42.94±5.88% accuracy; NSL→UNSW 69.81±12.11%; UNSW→CIC2017 35.59±8.14%; CIC2017→UNSW 44.94±0.00%, the last equivalent to **all-negative predictions on UNSW**. Their semantic-alignment sensitivity reaches ~34 percentage points in one transfer direction. Deployable AWE-IDS reaches 87.51%, 79.89%, 99.89% in-domain on UNSW, NSL and CIC2017, respectively; oracle increases are explicitly NOT deployable. Table 6 (p. 8) includes RF 99.78% on CIC2017 in-domain; validation and training choices matter more than tiny model improvements.

**Bias-result interpretation caution (Table 3, p. 7):** The widely highlighted 10.46 and 19.52 percentage-point figures are the **biased-validation minus official-test accuracy gaps** on UNSW and NSL, *not necessarily the incremental effect of choosing the biased validation strategy*. The corresponding proper-validation gaps are 8.15 and 19.17 points; the **differences between those gaps** are about 2.31 and 0.35 points. Therefore be precise about the estimand and avoid attributing the entire 10.46/19.52 to the Keras tail-validation shortcut. The paper itself provides both rows.

**Limitations (pp. 11–12):** Manual, approximate semantics; incompatible category vocabularies; source calibration may not reflect target; experiments binary only; no principled domain adaptation; lab traffic is not a substitute for deployment traffic.

**Overlap with us:** Almost completely covers the simplistic claim “random split gives 99%, external set falls to near chance.” A new ensemble or threshold-tuning-on-test-set idea is not persuasive. Their reproduction package is a useful template for saving exact scripts, seeds, split indices and CSV outputs.

**Opportunity remaining:** Attack-family stratification with independently validated feature semantics and operationally feasible source-only mitigation, evaluated more stringently than the pilot.

### P03 — Jin & Lee (2026)
**Title:** *Benchmark-Shift-Aware Intrusion Detection for Evolving Network Traffic: Cross-Dataset Generalization, Calibrated Alerting, and Score-Orientation Diagnostics*.  
**Venue:** Electronics 15(17), 3761; published 22 Aug 2026.  
**DOI:** https://doi.org/10.3390/electronics15173761  
**Priority:** Most important for interpreting our 3.43% recall result.

**Research question:** After benchmark shift, does failure occur in discrimination/ranking, probability calibration, threshold transfer, analyst alert capacity, or inverted score orientation? Can Transformer/SSL models solve it?

**Methods (pp. 4–8):** Datasets HIKARI-2021 (555,278 flows), CICIDS2017 (2,827,876), and a deterministic CICIoT2023 subset (2,158,141; 97.6520% attack). Rich-64 semantic flow features for HIKARI/CIC2017; Minimal-13 for 3-way comparison. XGBoost, supervised Transformer, masked-feature self-supervised Transformer. Source-only preprocessor, sigmoid probability calibrator, source validation threshold targeting 5% FPR. Separates target-label-free score quantile alert budgets from **oracle-only** score inversion. CIC2017 chronological evaluation uses Monday–Wednesday train, Thursday validation, Friday test. The authors state their Rich-64 HIKARI file required an integrity correction after a column scale audit.

**Main numbers:** In-domain XGBoost FPR 4.94–5.40%; F1 0.507–0.995 (Table 3, p. 8). Strict HIKARI→CIC2017 Rich-64 recall at source-derived threshold was **zero for all three model types** (Table 4, p. 9). Its XGBoost ROC-AUC was 0.214126 with oracle-diagnosed flipped orientation (not a deployable fix), while Transformer had ROC-AUC 0.588790 but zero recall at the chosen source threshold. Reverse CIC2017→HIKARI XGBoost target FPR rose to 18.9196%, recall only 10.7839% at transferred threshold. A 5% score-budget rule in that direction yielded F1 0.112238, precision 0.129418 and recall 0.099085 (Table 5, p. 10). Minimal-13 LOSO was also weak: zero recall for two targets and 0.031595 for HIKARI (Table 6, p. 11). Chronological CIC2017: Transformer PR-AUC 0.790 vs XGBoost 0.669, but thresholded recall 0.101 vs 0.362 (Table 7, p. 12).

**Limitations (p. 14):** CICIoT2023 is not the whole dataset; chronology only for CIC2017; fixed single-seed experiments; source validation split shared by calibration and threshold selection; no real-time throughput evaluation; SSL objective not broad enough to disprove SSL generally; significant GPU costs.

**Overlap with us:** Directly invalidates an unqualified “simply adjust threshold and claim a new method” strategy. It already analyzes calibration failure, score orientation, alert budgets, and semantic alignment. We *must measure* ROC-AUC, PR-AUC/AP-lift and scores before saying our external classifier has fundamentally lost its ability to rank attacks.

**Potential distinction:** Test an explicitly attack-family-conditioned, semantics-audited mitigation on CIC-DDoS2019 while separating what is implementable without target labels. Any mitigation must beat source-only thresholds, robust feature subsets, and non-neural baselines on strictly unseen captures.

### P04 — Elangovan et al. (2026)
**Title:** *Cross-Dataset Temporal and Semantic Generalization of Intrusion Detection Models for the Future Internet*.  
**Venue:** Future Internet 18(4), 194; published 2 April 2026.  
**DOI:** https://doi.org/10.3390/fi18040194  
**Priority:** Useful broad comparison; evaluate methodological claims critically.

**Research question:** How do classical/deep IDS methods change as datasets move forward in collection year and from enterprise to IoT conditions? Which semantic feature groups remain stable?

**Methods (pp. 6–14):** Four datasets: CIC-IDS2017 (2017), CSE-CIC-IDS2018 (2018), CICIoT2023 (2023), CIC-DIAD2024 (2024). Five models RF, Gradient Boosting, MLP, Autoencoder, 1D-CNN. Semantic groups: flow intensity, packet size, temporal activity, directionality, transport flags, protocol. Training-only preprocessing, labels harmonized, hyperparameters held fixed, forward chronological ordering **by dataset year**, and reporting mean±SD over five runs. ROC-AUC, FPR, class-wise recall, Macro-F1 and balanced accuracy considered.

**Key results:** RF within-dataset Macro-F1 on CIC2017 0.941±0.003, CSE-CIC2018 0.927±0.004, CICIoT2023 0.891±0.005, CIC-DIAD2024 0.885±0.005 (Table 9, p. 15). RF trained on CIC2017 scored Macro-F1 0.882 on 2018, 0.741 on 2023 and 0.713 on 2024 (Table 10, p. 16); other models also degraded. Reported increases in benign FPR to roughly 0.18–0.31 under enterprise-to-IoT transfer. Authors report comparatively greater stability of flow-intensity/temporal features than protocol-dependent indicators (section 5, pp. 25–27).

**Limitations and methodological cautions:** Their “temporal” trend is estimated primarily from **different datasets from different years**, often with different network domains and attack mixtures. Thus time, network domain, capture instrumentation and attack family are entangled; it cannot isolate a pure time/aging effect. The displayed Table 3 provides category-level representative features, not by itself a complete machine-readable per-column alignment map. Table 8 (p. 15) lists CIC2017 as **83.3% benign**, while the much-used 2.827M-flow representation in the Electronics paper counts 2,271,320/2,827,876 ≈ **80.3%** benign; verify data variants before borrowing the 83.3% statistic. The paper reports five-run variability, but differences in dataset composition are not controlled by repeated seeds alone. No adaptation intervention is evaluated; the authors explicitly present this as future work (pp. 26–27).

**Overlap with us:** Already argues that some feature groups are more stable than others. Selecting flow intensity/temporal variables alone is insufficient novelty.

**Potential distinction:** Produce testable, family-conditioned and capture-aware evidence that distinguishes environment shift from attack-mechanism shift—if sufficient metadata exist—and test a source-only remedy.

---

## B. Critical prior work identified in an additional web search (NOT among the four uploaded PDFs)

**External-source distinction:** The following observations come from publication abstracts or publisher/dataset pages found during supplemental web research, not a full reading of their manuscripts. Do not quote a detailed method or numeric result from them until obtaining and reading the full text.

1. **D'hooge et al., Sensors (2023), “Investigating Generalized Performance of Data-Constrained Supervised Machine Learning Models on Novel, Related Samples in Intrusion Detection.”** DOI https://doi.org/10.3390/s23041846. The abstract explicitly states that it trains on CSE-CIC-IDS2018 and tests on CIC-IDS2017, CIC-DoS2017 **and CIC-DDoS2019**, reporting both serious generalization problems and some very strong cross-dataset models on network-centric attack classes. **Priority #1 to acquire.** This is the closest novelty threat to our current dataset pair and supersedes any assumption that CIC-DDoS2019 transfer has never been studied.
2. **“BHM-IDS: Behavior-Driven Hierarchy and Multi-Dataset Training for Cross-Dataset Generalization,” Applied Sciences (2026).** DOI https://doi.org/10.3390/app16167885. Publisher summary says it trains using CIC-IDS2017 enriched with CIC-DDoS2019 and evaluates on CSE-CIC-IDS2018; explores behavior-based hierarchy. **Priority #2 to acquire** if considering hierarchical or multi-dataset training.
3. **“A cross-dataset benchmark for IoT DDoS detection with semantic feature alignment and SHAP,” Computers and Electrical Engineering, volume 141 (2027 issue; DOI contains 2026),** DOI https://doi.org/10.1016/j.compeleceng.2026.111606. Abstract states that it maps CICIoT2023, CIC-DDoS2019 and TON-IoT into 15 canonical traffic semantics, with ML/DL/GNN baselines and SHAP. **Priority #3** if considering semantic feature alignment, SHAP or explainable models. This is extremely overlapping at the idea level.
4. **Engelen et al. (2021), “Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study.”** DOI https://doi.org/10.1109/SPW53761.2021.00009, plus extended documentation: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html . Documents label/flow-direction and CICFlowMeter defects; essential for interpretation and provenance. Documentation warns flaws may affect 2018/2019 CICFlowMeter-based datasets too.
5. **Catillo et al. (2021/2022):** “Demystifying the Role of Public Intrusion Datasets: A Replication Study of DoS Network Traffic Data,” DOI https://doi.org/10.1016/j.cose.2021.102341; “Transferability of Machine Learning Models Learned from Public Intrusion Detection Datasets: The CICIDS2017 Case Study,” DOI https://doi.org/10.1007/s11219-022-09587-0. Cited in Berhili et al.; read before making transferability novelty claims.

**Dataset provenance:** Current 431,371-flow CSV was obtained from **Talukder & Uddin (2023), Mendeley Data v1**, DOI https://doi.org/10.17632/ssnc74xm6r.1, file `cicddos2019_dataset.csv` (listed as ~147 MB). Mendeley describes/repackages CIC-DDoS2019, but this **does not yet prove which official PCAP/CSV days, filters, sampling, feature-export revisions or label mappings generated its 431,371 records**. Official CIC page: https://www.unb.ca/cic/datasets/ddos-2019.html . It documents separate first- and second-day attacks, CICFlowMeter-V3 flow extraction, and PCAPs/logs. Audit and record the redistributor's exact processing before calling it a representative official benchmark.

---

## C. Reconstructing our actual experiment (do not conflate with published findings)

### User-supplied experimental history

- Source CIC-IDS2017 Friday DDoS CSV: **225,745** flow rows; **128,027 DDoS** and **97,718 benign**.
- Full source audit: **2,633 exact duplicate rows**, **64 infinite values**, **4 missing Flow Bytes/s**, **10 constant columns**, no explicit timestamp or flow ID/IP fields in the exported file. Caution: duplicate count is for the full 225,745 rows; **729 duplicates** were found in the first-100k experiment, not the full set.
- Original baseline: first 100k CIC2017 rows, RF 100 trees, accuracy **99.985%**. Dedup of first 100k: **99.97985%**, not a controlled same-test-set causal comparison.
- Matched-feature internal evaluation: **57 shared usable numeric features**, source stratified 80/20, accuracy **99.96456%**, attack recall **99.96485%**, F1 **99.96875%**; confusion matrix `[[19537, 7], [9, 25596]]`.
- External evaluation: separately fitted 100-tree balanced RF on **all 225,745** source rows, same 57 features, held-out **431,371-row** redistributed CIC2019 CSV; **333,540 attacks, 97,831 benign**; accuracy **25.3000%**, attack precision **98.8507%**, attack recall **3.42957%**, F1 **6.62915%**; confusion matrix `[[97698, 133], [322101, 11439]]`.
- Additional derived statistics: external **specificity 99.8641%**, **FPR 0.13595%**, **balanced accuracy 51.6468%**; predicting every row as an attack would achieve **77.3209% accuracy** but would misclassify all benign flows. Therefore overall accuracy alone is especially misleading.
- Family breakdown: only `Syn` detected (11,439/49,373 = 23.1685% recall). **Zero detected among 284,167 attacks** from other family labels in this evaluated subset. These are sample-level recall figures, not guarantees about the full original CIC2019 collection.
- Directionality diagnostic: attack detection **5.03%** among flows with at least one backward packet, **3.20%** among zero-backward-packet flows. Difference 1.83 percentage points does **not** establish that backward traffic causes failure.
- Feature shift: CIC2017 attack median Bwd Packet Length Mean **1,934.5** vs CIC2019 **0**; Fwd Packet Length Mean **7** vs **440**; Flow Bytes/s **160.1** vs **13,215.7**. These observations are descriptive, strongly confounded by attack family and capture/extraction environment.

### Methodology limitations to correct before publication

1. **Not identical fitted model:** source internal experiment trains on 80%; external experiment trains on 100%. Train one model on a fixed source training fold; evaluate that **exact fitted model** on a fixed source holdout and untouched external test. Repeat seeds or better group/session-based splits where metadata allows.
2. **No reliable source chronology:** a block-holdout based on CSV row order is not a chronological test unless provenance proves rows were time ordered. We must not call it temporal validation without timestamps.
3. **Unverified feature semantics:** 67 exact-name matches != 67 equivalent definitions; 57 nonconstant features may still have extraction defects or different units. Missing 2017 features like `Init_Win_bytes_forward` (one of the strongest original importances) also change the learned representation; document why and how.
4. **No independent raw capture verification:** the 2019 file is redistributed, not the official unmodified collection. Provenance, true attack types, duplicate/near-duplicate overlap and any class-specific filtering remain unknown.
5. **No ranking/score diagnostics:** low recall may arise partly from threshold transfer, poor ranking, score inversion, or an attack-family distribution shift. Measure ROC-AUC and average precision before choosing remedy; only choose deployable operating thresholds from source validation.
6. **Selection/exploration bias:** the target labels have now been inspected repeatedly to choose hypotheses. Treat CIC2019 as an **exploratory target**, not a pristine untouched confirmatory target for a new method. Reserve a different official capture/day or another dataset for final confirmatory evaluation.
7. **No run uncertainty:** single RF seed, no cross-source variance or interval. Report several seeds; confidence intervals should respect correlated flow/session structure, not only per-row binomial formulas.
8. **Attack-family labels and protocol confounded:** SYN/UDP/NTP/TFTP etc. simultaneously reflect protocol, generation method, bidirectional stats and traffic environment. A family-level failure does not imply a mechanistic explanation.
9. **Class balance and operating point:** report precision, recall, specificity, FPR, balanced accuracy, macro-F1, per-family recall, AP/PR-AUC, ROC-AUC, and prevalence/alert volume. Record class prevalence per *actual split*, not only entire dataset.
10. **No data mutation or cherry-picking:** keep original CSV immutable; save dataset SHA-256 hashes, software versions, commit hash, split identifiers, config, confusion matrix and raw predictions/scores for each experiment.

---

## D. Revised research problem and hypotheses

### High-value framing (tentative)
**Working title:** *When DDoS Detectors Fail Across Captures: Disentangling Attack-Family Shift, Flow-Feature Incompatibility, and Decision-Threshold Transfer*.

This title is **provisional, not a novelty claim**. A paper needs to distinguish at least two of these failure mechanisms with controlled, independently replicated evidence; a pre-specified remedy is preferable.

### Research questions

**RQ1 — Model behavior:** Do failures persist when an identical source-trained model is evaluated internally and on multiple independent external target captures, with traceable semantics and balanced metrics?

**RQ2 — Mechanism diagnosis:** How much of the failure is due to feature-generation incompatibilities, benign capture shift, unseen attack-family behaviors, and source-derived threshold transfer? Can we test each factor in controlled ablations rather than correlational shifts alone?

**RQ3 — Remedy:** Can a source-only feature-stability/attack-mechanism approach improve worst-family recall or macro-F1 at an acceptable false-positive budget **without using target labels** to select features, thresholds or hyperparameters?

**RQ4 — Robustness:** Does any remedy persist across a holdout dataset/day and 3–5 independent seeds, with calibration, alert-volume and uncertainty reported?

### Pre-specified candidate hypotheses

- H1: External attack-score ROC-AUC/AP (and per-family score distributions) will reveal whether low recall is primarily cutoff-related or reflects weak ranking. *Unverified.*
- H2: Feature subsets prioritizing physically interpretable, semantically comparable traffic characteristics outperform the unrestricted 57-feature source-trained baseline on at least one independent unseen attack family under an identical source-only threshold budget. *Unverified; literature already covers stability in broad terms.*
- H3: Attack-family holdouts (within a single common extraction pipeline) show a comparable degradation pattern to between-dataset transfer, supporting a contribution from attack diversity beyond extraction mismatch. *Unverified.*
- H4: A small source-only remedy improves **worst-family recall without unacceptable benign false-positive escalation** on a sealed holdout. *Unverified; this would be a stronger contribution if supported.*

---

## E. Data choice and evaluation design — minimal, defensible sequence

**Dataset 1 (source/exploration):** Existing CIC2017 Friday DDoS file, but obtain official source and cross-check row count and semantics.

**Dataset 2 (exploratory target):** Mendeley CIC-DDoS2019 subset (431,371). Useful to find failure mechanisms but now repeatedly inspected and thus no longer an unbiased final confirmatory benchmark.

**Dataset 3 (confirmatory):** Prefer **official CIC-DDoS2019 first/second-day CSVs and documented scenario/session boundaries**, or another genuinely independent collection such as CSE-CIC-IDS2018 **with comparable DDoS attack labels and features**. The official 2019 files have separate attack scenarios and both days; one can prospectively seal one or more scenarios/held-out days, but independent scenarios on the same dataset are not necessarily independent network environments.

**Optional dataset 4:** Corrected/re-extracted CIC2017 flows based on Engelen et al. to quantify sensitivity to labeling/flow-extraction defects. Do not equate this with independent external validation.

**Why not start with UNSW-NB15 or NSL-KDD?** They have different feature-generation paradigms. The Berhili paper demonstrates that only 10–14 features were alignable via pivot, and NSL↔CIC had only three strict equivalences. This makes them less suitable for *isolating* CICFlowMeter attack-family effects in our first study, although valuable for a separate semantic-alignment benchmark.

### Validation matrix

| Train | Evaluate | Purpose | Target data allowed for model/threshold selection? |
|---|---|---|---|
| CIC2017 source train | CIC2017 fixed holdout | Identical-model internal baseline | No |
| CIC2017 source train | Mendeley CIC2019 | Exploratory external diagnosis | No target labels for training/selection; yes post-hoc analysis |
| CIC2019 source portion | CIC2019 different split/scenario | In-domain target baseline and family availability | Training only within designated source portion |
| CIC2019 source portion | CIC2017 holdout | Reverse transfer asymmetry | No target labels for tuning |
| CIC2019 family subset | Sealed unseen CIC2019 family | Within-extractor leave-one-family-out | No held-out family observations for tuning |
| Source-only model | Sealed official target day/scenario or 2018 capture | Final confirmation | Absolutely no target-informed model selection |

**Study integrity:** Pre-register (in a repository note) dataset roles, metrics, acceptable FPR, feature groups and stopping decisions before examining the final holdout. For datasets with unavailable timestamps/IDs, acknowledge absence and avoid claiming session or chronological independence.

---

## F. Experiment order and diagnostic methods

### E0 — Provenance and schema audit (gate before further causal claims)
- Record URLs, download date, author, license, original CSV filename, byte size and SHA256 hashes. Track Mendeley transformations; contact depositor if necessary.
- Audit field names/types/units, class labels, source/day/protocol distributions if available, exact and near duplicates, contradictory feature duplicates with differing labels, NaN/Inf, constant columns and suspicious identifier-like fields.
- Build a **feature compatibility ledger** with: canonical feature concept, original column name in each dataset, units, extraction rule, potential CICFlowMeter bug/version, distribution, whether safe to transfer. For synonyms (e.g. `Init_Win_bytes_forward` vs `Init Fwd Win Bytes`), document equivalence before mapping. Inspect `Protocol` only when source has a verified compatible measurement; avoid adding metadata selectively to target.

### E1 — Identical-model source/target test
- Fix one source training/validation/test split (duplicate-safe, no target data in selection); fit source-only median imputer and model; save the *fitted model*.
- Predict identical source holdout and untouched exploratory target; record 2×2 matrices and all metrics; use at least 3–5 seeds. Log source training feature list and model SHA/checksum.
- Add trivial majority-class baselines and simple interpretable Logistic Regression/Random Forest/gradient boosting where feasible; this is not a model-architecture contest.

### E2 — Is threshold transfer the problem?
- Save `predict_proba` on source holdout, source validation, and target, by actual class and attack family.
- Compute ROC-AUC, Average Precision and AP minus target prevalence; source-calibrated FPR-budget thresholds (e.g. 0.5%, 1%, 5% on source benign validation); plot source vs target score distributions; report target alert rate, attack recall and benign FPR.
- Compare with **oracle-only** target thresholds in a clearly segregated diagnostic appendix, never as deployable model performance. Test score inversion only as a labeled diagnostic, as in Jin & Lee.
- Distinguish (a) good ranking/bad threshold from (b) lost ranking/label orientation inversion.

### E3 — Attack-family versus extraction shift
- Within same original data/extraction pipeline, hold out distinct attack families and evaluate unseen ones; this helps isolate behavioral novelty from inter-extractor shift. Families may still be entangled with capture timing and device environment; report that limitation.
- Compare per-family feature vectors and prediction score distributions, ideally grouped by packet protocol if protocol is genuinely present/derivable on *both* sides. Test backward-packet effects *within each family* and provide effect sizes with counts.
- Test semantically audited feature groups: size, intensity, time, direction, flags; do not make a shortlist based only on target labels.

### E4 — Small mitigation with strict controls
- Candidate A: source-only stable-feature filtering chosen via source-side perturbation or source subdomain validation.
- Candidate B: source-trained model with predefined mechanism/feature groups and an explicit unknown-family/uncertainty/reject option; rejection is not magically correct detection and must be scored as such.
- Candidate C: source-only class reweighting/regularization/score calibration based on source-validation, not test-set threshold sweeps.
- Compare identical base models, data and metrics; publish negative outcomes as limitations; avoid introducing a Transformer merely to claim novelty.

### E5 — Confirmation and reporting
- Freeze method before sealed target evaluation. Repeat across ≥3 seeds and nonoverlapping groups/captures where available; report uncertainty and effective sample unit.
- Report per-family confusion matrices, worst-family recall, macro-F1, balanced accuracy, AP/ROC-AUC, FPR, alert rate, time/memory where relevant. Include an ablation separating (i) feature harmonization, (ii) threshold calibration, (iii) attack-family diversity, (iv) proposed intervention.
- Release scripts, split manifests, environment, dataset provenance notes, plots and raw result summaries; avoid redistributing licensed raw data unnecessarily.

---

## G. Publication reality check and go/no-go criteria

**Already studied (cannot claim as novel):** in-domain vs external collapse, general dataset shift, mere use of RF/XGBoost, semantic alignment in general, risk of random/biased splits, source-threshold transfer failures, score inversion, broad stability of temporal/flow features, and generic multi-dataset training.

**May offer an original contribution only with evidence:** a controlled causal decomposition of shift causes for DDoS families; well-defined feature-extraction comparability controls; a rigorously specified attack-mechanism-aware method that improves worst-family recall at predeclared false-alarm cost; a reusable, verifiably sourced attack-family transfer benchmark with justified label/schema quality. Search the additional directly overlapping papers first.

**Go** if a clear gap remains after reading the 2023 follow-up and 2026 semantic-alignment/hierarchy papers, a clean independent validation dataset is available, and one controlled intervention shows gains against appropriate source-only baselines.  
**Pivot** if the only result is another demonstration of transfer failure, feature definitions cannot be validated, or proposed remedies require target labels to choose parameters. A reproducibility/negative-results paper is still possible depending on target venue, but acceptance/publication inside two months is not guaranteed.

### Seven-day research sprint
1. Read full 2023 Sensors follow-up (exact CIC-DDoS2019 overlap); extract experiments, attack types, feature procedure and strongest successful transfer results.
2. Obtain/read BHM-IDS 2026 and the semantic 15-feature CIC2019 benchmark (at least abstract/Methods; full text if available); update overlap ledger.
3. Document Mendeley CSV provenance and source-2017 version, official CSV/day boundaries and feature mapping.
4. Run identical-fitted-model source/target metrics plus ROC-AUC, average precision and score distributions; preserve run IDs.
5. Write attack-family-shift versus measurement-shift hypothesis table with controls and data feasibility.
6. Decide 1–2 candidate remediation ideas; define source-only training and final sealed holdout protocol.
7. Go/no-go meeting with research supervisor or coauthor: settle a narrow title, target journal/conference, required tables and authorship expectations.

---

## H. Research-log format and artifact naming

Every run should have a durable ID, e.g. `E02_seed42_2026-10-30`. Save:

```
research/
  literature/
    PAPER_REVIEW_AND_RESEARCH_PLAN.md
    literature_matrix.md
  notes/
    RESEARCH_LOG.md
    dataset_provenance.md
    feature_compatibility.md
  experiments/
    ...scripts...
reports/
  experiments/
    E02_seed42_2026-10-30/
      config.json
      dataset_sha256.txt
      feature_list.txt
      metrics.json
      confusion_matrix.csv
      per_attack_family.csv
      score_diagnostics.png
      notes.md
```

Each notes file should contain: research question; dataset names/versions; exact row counts and filters; split protocol and split IDs; fitted-model parameters; commit hash; metrics/plots; confidence intervals; whether target labels were used *anywhere* in selection; supported inference; competing explanations; decision for the next experiment.

---

## I. Bibliography and verification checklist

### Four uploaded PDFs — fully read as the requested primary sources
- [P01] L. D'hooge, T. Wauters, B. Volckaert, F. De Turck, *Inter-dataset generalization strength of supervised machine learning methods for intrusion detection*, JISA 54 (2020) 102564. DOI: 10.1016/j.jisa.2020.102564.
- [P02] M. Berhili, O. Chaieb, M. Benabdellah, *How evaluation protocol choices inflate intrusion detection accuracy: A three-dataset systematic study*, Computers & Security 173 (2027) 105218, available online 24 Oct 2026. DOI: 10.1016/j.cose.2026.105218.
- [P03] H. Jin, H. Lee, *Benchmark-Shift-Aware Intrusion Detection for Evolving Network Traffic: Cross-Dataset Generalization, Calibrated Alerting, and Score-Orientation Diagnostics*, Electronics 15(17) (2026) 3761. DOI: 10.3390/electronics15173761.
- [P04] R. Elangovan et al., *Cross-Dataset Temporal and Semantic Generalization of Intrusion Detection Models for the Future Internet*, Future Internet 18(4) (2026) 194. DOI: 10.3390/fi18040194.

### Newly identified — full-text read pending
- [P05] D'hooge et al. (2023), *Investigating Generalized Performance of Data-Constrained Supervised Machine Learning Models on Novel, Related Samples in Intrusion Detection*, Sensors 23(4) 1846. https://doi.org/10.3390/s23041846.
- [P06] *BHM-IDS: Behavior-Driven Hierarchy and Multi-Dataset Training for Cross-Dataset Generalization*, Applied Sciences 16(16) (2026) 7885. https://doi.org/10.3390/app16167885.
- [P07] *A cross-dataset benchmark for IoT DDoS detection with semantic feature alignment and SHAP*, Computers and Electrical Engineering 141 (2027), 111606. https://doi.org/10.1016/j.compeleceng.2026.111606.
- [P08] G. Engelen et al. (2021), *Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study*, IEEE SPW. https://doi.org/10.1109/SPW53761.2021.00009.
- [P09] M. Catillo et al. (2021), *Demystifying the Role of Public Intrusion Datasets: A Replication Study of DoS Network Traffic Data*, Computers & Security. https://doi.org/10.1016/j.cose.2021.102341.
- [P10] A. Catillo et al. (2022), *Transferability of Machine Learning Models Learned from Public Intrusion Detection Datasets: The CICIDS2017 Case Study*, Software Quality Journal. https://doi.org/10.1007/s11219-022-09587-0.

### Primary dataset documentation
- Official CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- Official CIC-DDoS2019: https://www.unb.ca/cic/datasets/ddos-2019.html
- Talukder & Uddin Mendeley Data v1 (2023): https://doi.org/10.17632/ssnc74xm6r.1
- Engelen extended documentation: https://intrusion-detection.distrinet-research.be/WTMC2021/extended_doc.html

---

**Research integrity reminder:** All suggested remediation and novelty statements in this memo are hypotheses, not reported achievements. The four-paper reading is complete at the level of full text and primary displayed tables. Supplemental studies were located by metadata/abstract and still require full-text scrutiny; dataset compatibility and actual external validation reproducibility remain unverified. Publication in two months cannot be promised.
