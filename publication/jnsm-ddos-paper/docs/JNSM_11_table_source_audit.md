# JNSM Revision 10: evidence audit against Mac-side exports

**Audit date:** 12 November 2026  
**Source:** User-uploaded `JNSM_evidence_from_Mac.zip` (97 files; raw traffic CSVs deliberately excluded).  
**Compared manuscript:** Revision 10 `main.tex` (11 tables).

## Result: 11 of 11 tables pass

A standalone source-to-table script checked **170 printed numeric cells** and the expected row labels across all **eleven LaTeX tables**. Every displayed numerical cell matched the actual exported result CSV/JSON values to its printed precision. The script independently recalculates Experiment 9's mean, sample SD, minimum and maximum from the 80-fit `per_run.csv` rather than assuming previously transcribed summary numbers. The results are recorded in `table_audit.json`; the checking script is included.

| LaTeX table | Comparison source | Numeric cells | Status |
|---|---|---:|---|
| 1. Dataset counts | Exp 9 manifest and Exp 11 per-dataset run metadata | 9 | PASS |
| 2. Initial transfer | `reports/score_diagnostics/summary.csv` | 9 | PASS |
| 3. Threshold sensitivity | `reports/score_diagnostics/operating_point_audit.csv` | 12 | PASS |
| 4. Feature ablation | `reports/experiment_05/metrics.csv`, `design.json` | 15 | PASS |
| 5. Training-only deduplication | `reports/experiment_06/fixed_split_dedup_comparison.csv` | 10 | PASS |
| 6. Three-seed source-grouped 2019 | `reports/experiment_07/metrics_by_seed.csv` | 30 | PASS |
| 7. Three-seed source-grouped 2018 | `reports/experiment_08/final_metrics.csv` | 30 | PASS |
| 8. Twenty-seed original arm | `research/results/experiment9_20seeds/per_run.csv` | 12 | PASS |
| 9. 2019 protocol strata | `research/results/experiment10_protocol_scoring/stratified_metrics.csv` | 15 | PASS |
| 10. Attack-family × protocol | `research/results/experiment10_protocol_followup/family_protocol_performance.csv` | 16 | PASS |
| 11. Within-target 2018 and 2019 | Two `within_target_summary.csv` exports | 12 | PASS |
| **Total** | | **170** | **11/11 PASS** |

## Code and metadata continuity

- All **97** files match the SHA-256 hashes recorded in the uploaded collection manifest. This establishes **archive integrity**, not raw dataset provenance.
- All included Python scripts pass compilation/syntax checking. This is **not** an execution audit of all historical experiments.
- The ordered **57-feature A** list matches Experiment 1's feature metadata exactly. The **37-feature B** list is a strict subset, with its original order retained.
- Experiment 8's alias dictionary has 49 definitions, of which **34** apply to the 57 features used by A (matching the schema audit).
- Exp 8's six source-holdout AUCs and six calibration thresholds match Exp 7's saved values; hence the 2018 evaluation was recorded with the same source-model protocol as the earlier 2019 evaluation.
- Exp 10's three explicit protocol groups reproduce the original global target-class counts and the seed-42 TP/FP confusion totals.
- The two Exp 11 output confusion matrices, sample sizes, recall and benign FPR are arithmetically consistent.
- Audit software was tested against an intentionally corrupted LaTeX table AUC and returned failure; this is a fail-closed comparison for the displayed numeric table cells.

## Original three-seed feature comparison (uploaded CSVs)

| Target | Seed | A (57) AUC | B (37) AUC | Difference B−A |
|---|---:|---:|---:|---:|
| Redistributed 2019 | 42 | 0.848663 | 0.853043 | +0.004380 |
| Redistributed 2019 | 43 | 0.806813 | 0.817333 | +0.010520 |
| Redistributed 2019 | 44 | 0.786427 | 0.900897 | +0.114470 |
| CSE-CIC-IDS2018 | 42 | 0.526722 | 0.450275 | −0.076447 |
| CSE-CIC-IDS2018 | 43 | 0.539547 | 0.163671 | −0.375877 |
| CSE-CIC-IDS2018 | 44 | 0.517250 | 0.521950 | +0.004700 |

The mean paired AUC differences are +0.043123 on 2019 and −0.149208 on 2018. These three correlated runs alone cannot justify a robust difference in generalization. Feature B was chosen after examining 2019, which remains exploratory.

## What the audit does NOT establish

1. **No independent model rerun on the raw CSVs** was possible here because the shared ZIP excludes them. The raw CSV SHA-256 values stored in run metadata were not independently rehashed in this audit.
2. The 2018 daily file's exact-row count and provenance concerns remain qualified; an exact 2018 row count of 1,048,575 in previous literature does not prove upstream completeness. A 34-name schema match does not demonstrate measurement equivalence.
3. **Prose numbers not printed in the eleven tables** were not comprehensively checked by this script. Selected protocol and confusion arithmetic was additionally checked, but this is not a sentence-by-sentence fact audit.
4. **DOI and citation-to-claim verification** and author/funding/AI-use declarations are separate final gates.
5. An extended 20-seed A-versus-B comparison is still essential before characterizing representation superiority or reversal as stable. The attached runner is tested for syntax, split disjointness, the legacy-reference comparison function and numerical target-cache/stream equivalence on synthetic data. It has **not** been executed against the author's raw CSVs.

## Next action

Use `paired_AB_20seeds.py` on the Mac: first reproduce the 12 archived seed–target–arm combinations, then run the full 20-seed extension. The source splitting protocol, imputation, RF hyperparameters, target aliases and score > threshold decision rule are preserved. Once the Mac output arrives, recompute paired A/B results and revise the claim strength in the LaTeX manuscript accordingly. Do not interpret the three-seed pattern as a robust ranking in advance.
