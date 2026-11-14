"""Experiment 02: Attack-family ranking diagnostics from frozen Experiment 01 scores.

Does NOT retrain, change thresholds, or select on target labels. Labels are
used only for post-hoc diagnostics on already-scored CIC-DDoS2019 data.
Run: python research/experiments/experiment_02_family_ranking.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'reports' / 'score_diagnostics'
OUTPUT = INPUT / 'family_ranking'
OUTPUT.mkdir(parents=True, exist_ok=True)

required = ['external_per_flow_scores.csv.gz', 'source_holdout_scores.csv.gz', 'metadata.json']
for name in required:
    if not (INPUT / name).is_file():
        raise SystemExit(f'Missing {INPUT/name}. First rerun experiment_01_score_transfer.py.')

external = pd.read_csv(INPUT / 'external_per_flow_scores.csv.gz')
source = pd.read_csv(INPUT / 'source_holdout_scores.csv.gz')
metadata = json.loads((INPUT / 'metadata.json').read_text())

for col in ['attack_type', 'true_attack', 'attack_probability']:
    if col not in external:
        raise ValueError(f'Missing {col} in external saved scores')
for col in ['true_attack', 'attack_probability']:
    if col not in source:
        raise ValueError(f'Missing {col} in internal saved scores')

external['attack_type'] = external['attack_type'].astype(str).str.strip()
source_threshold = float(metadata['source_threshold'])
print('=== EXPERIMENT 02: FROZEN PREDICTIONS ===')
print('Source-derived threshold:', source_threshold, '(strict >)')
print('External rows:', len(external), 'source holdout rows:', len(source))

# Every attack family is compared against the SAME external benign reference.
# AUC for each family measures within-this-dataset discrimination; it must
# not be interpreted as proof of mechanism, novelty, or deployment performance.
benign = external.loc[external['true_attack'].eq(0), 'attack_probability'].to_numpy(float)
if len(benign) == 0:
    raise ValueError('No external benign samples found')
rows = []
for family, grp in external.loc[external['true_attack'].eq(1)].groupby('attack_type'):
    attack = grp['attack_probability'].to_numpy(float)
    ys = np.concatenate([np.zeros(len(benign), dtype=int), np.ones(len(attack), dtype=int)])
    ps = np.concatenate([benign, attack])
    rows.append({
        'attack_family': family,
        'attack_n': len(attack),
        'reference_benign_n': len(benign),
        'roc_auc_vs_same_external_benign': float(roc_auc_score(ys, ps)),
        'average_precision_vs_same_external_benign': float(average_precision_score(ys, ps)),
        'paired_sample_attack_prevalence': float(ys.mean()),
        'attack_score_median': float(np.median(attack)),
        'attack_score_p10': float(np.quantile(attack, .1)),
        'attack_score_p90': float(np.quantile(attack, .9)),
        'fraction_score_zero': float(np.mean(attack == 0)),
        'fraction_score_exactly_source_threshold': float(np.mean(np.isclose(attack, source_threshold, atol=1e-12))),
        'recall_fixed_0p5': float(np.mean(attack >= .5)),
        'recall_source_threshold_strict_gt': float(np.mean(attack > source_threshold)),
    })
result = pd.DataFrame(rows).sort_values('attack_n', ascending=False)
result.to_csv(OUTPUT / 'per_family_ranking.csv', index=False)

print('\n=== ATTACK FAMILY AUC VS SAME EXTERNAL BENIGN ===')
columns = ['attack_family','attack_n','roc_auc_vs_same_external_benign',
           'attack_score_median','fraction_score_exactly_source_threshold',
           'recall_fixed_0p5','recall_source_threshold_strict_gt']
print(result[columns].to_string(index=False, float_format=lambda x: f'{x:.5f}'))

# Group summary: note that family-weighted and attack-weighted means differ.
for group, subset in [('SOURCE HOLDOUT', source), ('EXTERNAL TARGET', external)]:
    y = subset['true_attack'].to_numpy(int)
    scores = subset['attack_probability'].to_numpy(float)
    b = scores[y == 0]
    a = scores[y == 1]
    if len(a) == 0 or len(b) == 0:
        continue
    print(f'\n=== {group} ===')
    print('AUC:', f'{roc_auc_score(y, scores):.5f}',
          'AP:', f'{average_precision_score(y, scores):.5f}')
    for threshold, comparison in [(0.5, '>='), (source_threshold, '>')]:
        flagged = (scores >= threshold) if comparison == '>=' else (scores > threshold)
        print(f'Threshold {comparison} {threshold:.5f}: '
              f'attack recall = {np.mean(flagged[y == 1]):.4%}; '
              f'benign FPR = {np.mean(flagged[y == 0]):.4%}')

print('\nCaution: family AUC uses one shared benign reference but each family has a')
print('different attack prevalence; family AP values are not directly comparable.')
print('Target labels are only used for post-hoc measurement; do not use this')
print('table to choose the final model/threshold without a fresh held-out dataset.')
print('\nSaved:', OUTPUT / 'per_family_ranking.csv')
