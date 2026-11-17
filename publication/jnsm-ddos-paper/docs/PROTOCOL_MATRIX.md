# Research protocols and reproducibility status

| Protocol | Partition type | Core configuration | Evidence | Status |
|---|---|---|---|---|
| Exp 1: source-to-2019 baseline | source 2-stage stratified rows, seed 42 | 57 features; RF 100 balanced; source benign 99th percentile, strict `>` | `research/experiments/experiment_01_score_transfer.py`, score CSVs | Local inputs required |
| Exp 6B: training-only dedup | fixed source split | original/dedup at seed 42 | `experiment_06b_fixed_split_dedup.py` | saved comparison verified; original partition arrays not archived |
| Exp 9: fixed split and placebo | fixed 2017 train/cal/holdout, 20 forest states | original, dedup, two class-matched fixed placebo subsets | `exp9_noise_floor/`, `research/results/experiment9_20seeds` | 20-seed results archived; arrays `.npz` excluded |
| Exp 7/8 and paired extension | source GroupShuffleSplit by 37-feature hash, seeds 42–61, cal split seed+1000 | A=57, B=37, 100-tree balanced RF, source 99th percentile threshold | `jnsm_audit_v2/paired_AB_20seeds.py`, `research/results/paired_AB_20seeds` | 12 historical reference checks PASS; 20-seed summary archived |
| Exp 10: protocol stratification | unchanged Exp 1 seed-42 model; group by explicit target Protocol | Protocol 0, 6, 17; family groups | `exp10_protocol_audit/`, `research/results/experiment10_protocol_scoring` | metrics archived; source protocol unavailable |
| Exp 11: within-target 2018/2019 | grouped by equal 57-feature vectors, not host/time/session | 100 trees, 2048 max leaf nodes, capped training/cal/test | `exp11_within_target/`, saved CSVs | diagnostics archived; row-index `.npz` excluded |
| Exp 11 ablations | saved 2018 partitions; post-hoc feature removal | each arm recalibrated | `exp11_ablation/`, CSVs | rows' saved partition arrays omitted; exploratory |

**Distinct 20-seed designs must not be conflated:** Exp 9 keeps source partitions
fixed and varies forest state; the paired A/B evaluation changes *both*
source group partitions and forest states by seed.

**Data lineage cautions:** 57/57 name matches and 34 needed aliases on 2018 do
not establish extractor equivalence. The 2019 redistribution matches its
Mendeley public CSV byte-for-byte but has unverified upstream lineage.
