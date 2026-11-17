# Source-trained DDoS detection across traffic datasets

**Scope:** Experiment scripts and archived results underlying the JNSM manuscript.
The code is shared for inspection and follow-up analysis; independent raw-data reruns and original packet-to-flow extraction are not included.

This package accompanies the working manuscript *Evaluating Source-Trained
DDoS Alerts Across Traffic Collections: Calibration, Model Variability, and
Protocol-Associated Errors*. It contains source code, preprocessing definitions,
feature sets, source-reported result CSVs and diagnostic run metadata.
The archived tables were cross-checked against the manuscript, but the
researcher's large raw CSVs were **not** included.

## Quick start

Use Python 3.9 as in the author's recorded 20-seed Mac run; create a venv
and install `requirements-reported.txt` (partial, not a complete lockfile):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-reported.txt
python3 tools/verify_saved_results.py
python3 tools/verify_package_manifest.py
```

These commands validate **archived result consistency and package integrity**;
no raw data is needed for them. For model refitting you must obtain the **exact
input files** described in [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md)
and pass their checksum verification. The 2017 input matches a public mirror of the complete Friday DDoS CSV by SHA-256; a fresh byte comparison with the original provider-hosted file has not been completed.

```bash
python3 tools/verify_input_fingerprints.py --root .
python3 jnsm_audit_v2/paired_AB_20seeds.py --project . \
    --verify-three-only --n-jobs 2 \
    --out research/results/paired_AB_reference_check
# Only after all 12 reference checks pass:
python3 jnsm_audit_v2/paired_AB_20seeds.py --project . \
    --n-jobs 2 --out research/results/paired_AB_20seeds_NEW
```

Do not reuse the archived `paired_AB_20seeds` output directory: the script
protects its outputs against accidental overwrite. The full study includes
other analyses; see [`docs/PROTOCOL_MATRIX.md`](docs/PROTOCOL_MATRIX.md).

## Directory guide

- `research/experiments/`: original numbered experiment scripts.
- `exp9_noise_floor/`, `exp10_protocol_audit/`, `exp11_*/`: later validated
  analysis/diagnostic scripts and README notes.
- `jnsm_audit_v2/`: full paired 57/37 feature-set 20-seed extension.
- `reports/`, `research/results/`: archived small aggregate CSV/JSON outputs.
- `docs/feature_set_manifest.json`: 57- and 37-feature lists.
- `docs/cicids2018_alias_definitions.json`: original 2018 names/aliases.
- `docs/input_fingerprints.json`: file hashes, counts and expected local paths.

## Important limitations

- Source 2017 source holdout is *not* a connection-, host- or time-disjoint
  independent network test. Grouped-source evaluations keep identical 37-
  feature vectors together, not necessarily their underlying sessions.
- 2019's smaller representation was informed by exploratory analysis of that
  target; positive paired results there are not prospectively validated.
- The 2018 daily target is a highly concentrated HOIC-dominated CSV with
  extensive repeated feature vectors; it is not the full 2018 collection.
- High internal AUC does not establish label validity or equivalent feature
  extraction across files.
- The Experiment 9 seed intervals and A/B experiment intervals use correlated
  repeated fits on fixed traffic files, not independent deployments.

A Python interpreter with the necessary dependencies is required for model refitting;
`requirements.txt` records the direct versions reported by the author's environment,
not a complete environment lock. The raw flow CSVs are excluded. Review
[`docs/CODE_RELEASE_REVIEW.md`](docs/CODE_RELEASE_REVIEW.md) for outstanding
reproducibility limitations and license selection.

## Results and figure/table reproduction

The script `table_reproduction/rebuild_all_ten_tables.py` reconstructs the six
main and four supplementary table bodies from archived numerical evidence.
Run it from any directory; the script reads its adjacent `evidence_for_tables/`.
It does not retrain the models.

The script `figure_reproduction/reproduce_figures.py` redraws source-derived
experimental plots from the included aggregate CSVs. Its figure appearance may
differ from the final typeset manuscript. Run it with the plotting dependencies
listed in `figure_reproduction/README.md`.

**Research attribution and limitations** are documented in the manuscript.
The repository should not be taken as evidence of independently verified traffic
labels or real-world deployment performance.
