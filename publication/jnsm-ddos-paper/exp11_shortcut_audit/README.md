# Experiment 11 follow-up: univariate shortcut audit

No model retraining. Uses saved Experiment 11 train/test row indices and
the original ordered 57 features. Chooses feature direction using the train
subset and evaluates the same direction on the test subset.

Install the ZIP from project root. Run:

    python3 exp11_shortcut_audit/experiment11_shortcut_audit.py \
      --dataset 2018 \
      --results-dir research/results/experiment11_2018 \
      --out research/results/experiment11_shortcut_audit_2018

Optional 2019:

    python3 exp11_shortcut_audit/experiment11_shortcut_audit.py \
      --dataset 2019 \
      --results-dir research/results/experiment11_within_target \
      --out research/results/experiment11_shortcut_audit_2019

Output: single_feature_auc.csv. This audit reads raw CSVs but does not fit
a Random Forest, modify original files, or edit the manuscript.

Interpret results as descriptive feature-label association, not demonstrated leakage.
