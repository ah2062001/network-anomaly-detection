#!/usr/bin/env python3
"""Check internal consistency of uploaded result exports (no model retraining)."""
from pathlib import Path
import math
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
res=root/'research/results'
fail=[]

def check(label,condition):
    ok=bool(condition)
    print(('PASS' if ok else 'FAIL')+' '+label)
    if not ok:fail.append(label)

# Experiment 9: fixed partition, 20 model seeds.
r9=pd.read_csv(res/'experiment9_20seeds/per_run.csv')
o=r9[r9['arm']=='original']
check('Exp9 original 20 unique model seeds',len(o)==20 and o['seed'].nunique()==20)
check('Exp9 original AUC mean',abs(o['external_auc'].mean()-0.806419)<0.000002)
check('Exp9 original AUC sample SD',abs(o['external_auc'].std(ddof=1)-0.040194)<0.000002)
# Paired A/B features across the two target files.
ab=pd.read_csv(res/'paired_AB_20seeds/paired_by_seed.csv')
summary=pd.read_csv(res/'paired_AB_20seeds/paired_summary.csv')
check('A/B 40 rows, 20 seeds × 2 targets',len(ab)==40 and all(ab.groupby('target').seed.nunique()==20))
for target in (2018,2019):
    sub=ab[ab['target'].astype(str)==str(target)]
    check(f'{target} B−A derived from stored scores',np.allclose(sub['B_auc']-sub['A_auc'],sub['delta_auc_B_minus_A'],atol=1e-12))
    actual=sub['delta_auc_B_minus_A'].mean()
    expected=summary[(summary['target'].astype(str)==str(target)) & (summary['metric']=='delta_auc_B_minus_A')]['mean'].iloc[0]
    check(f'{target} mean paired AUC summary matches',abs(actual-expected)<1e-10)
# Exp 10 protocol totals reproduce original global confusion matrix.
p=pd.read_csv(res/'experiment10_protocol_scoring/stratified_metrics.csv')
p=p[p.dataset=='target_2019']
check('Exp10 target attack flows and TP',p.attack_n.sum()==333540 and p.tp.sum()==193833)
check('Exp10 target benign flows and FP',p.benign_n.sum()==97831 and p.fp.sum()==10421)
# Exp11 confusion matrices and rates for the two within-target controls.
for year in (2018,2019):
    where=('experiment11_2018' if year==2018 else 'experiment11_within_target')
    row=pd.read_csv(res/where/'within_target_summary.csv').iloc[0]
    expected_attack={2018:65641,2019:66785}[year]
    expected_benign={2018:34359,2019:19051}[year]
    check(f'Exp11 {year} test confusion counts',int(row.tp+row.fn)==expected_attack and int(row.fp+row.tn)==expected_benign)
    check(f'Exp11 {year} recall',abs(row.tp/(row.tp+row.fn)-row.attack_recall)<1e-12)
    check(f'Exp11 {year} FPR',abs(row.fp/(row.fp+row.tn)-row.benign_fpr)<1e-12)
print(f'\n{len(fail)} failures; does not verify original raw labels, file semantics or reference articles.')
raise SystemExit(bool(fail))
