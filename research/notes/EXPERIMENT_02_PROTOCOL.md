# Experiment 02 — Attack-family ranking diagnostics

Date: 30 October 2026
Status: Follow-up diagnostic. Not evidence of a validated mitigation.

## Motivation
Experiment 01 compared **one fitted Random Forest** on a source-held-out CIC-IDS2017 split and an external redistributed CIC-DDoS2019 subset. External ROC-AUC was 0.90433 and average precision was 0.95342 (target attack prevalence approximately 0.77321). Default threshold 0.50 produced 2.830% attack recall and 0.133% benign FPR; a source-calibration-derived strict `score > 0.01` threshold yielded 58.114% attack recall but 10.652% benign FPR. Target threshold selection was **not** performed.

## Question
Does the retained target ranking signal extend across attack families, or does the overall ROC-AUC obscure near-random or inverted ranking for individual attack types?

## Method
1. Read saved `external_per_flow_scores.csv.gz`, `source_holdout_scores.csv.gz`, and `metadata.json` from Experiment 01. **Do not retrain** or read target labels for threshold selection.
2. For each attack family, compare its flow scores against the **same set of all external benign scores**; report ROC-AUC and descriptive score distribution.
3. Report sensitivity at the two **pre-existing** source-defined thresholds (0.5 with `>=`, 0.01 with strict `>`). Report the family sample sizes and tie fraction exactly at the chosen threshold.
4. Report attack-family AP with the paired family-plus-all-benign prevalence; note that these AP values have different baselines and are **not directly comparable** across families.

## Controls and caveats
- The dataset remains a redistributed CIC-DDoS2019 subset whose origin and preprocessing should be verified.
- Scores are produced by a model previously inspected using this target dataset. Family analysis is **exploratory**.
- Shared feature names do not establish semantic or unit equivalence.
- Flow records are not necessarily independent; confidence intervals should ideally account for capture/run grouping, which the source CSV may not retain.
- AUC measures ranking within this target sample, not whether fixed thresholds meet operational alert budgets.
- A family may contain few examples (e.g., WebDDoS = 51), so avoid generalizing from its point estimate.
- No target labels, per-family outcomes, or target-class statistics may be used to tune a claimed deployable method. Reserve a genuinely untouched dataset for confirmation.

## Outputs
`reports/score_diagnostics/family_ranking/per_family_ranking.csv`

## Manuscript interpretation rule
Results may be described as **post-hoc diagnostic evidence** of score/ranking behavior. Do not claim a new model, correction, deployment performance, causal explanation, or successful generalization improvement until demonstrated by independent controlled experiments.
