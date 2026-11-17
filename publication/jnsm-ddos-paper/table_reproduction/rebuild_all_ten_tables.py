#!/usr/bin/env python3
"""Rebuild 6 main and 4 supplementary tabular bodies from archived numerical CSVs.

Requires only the Python standard library and the evidence_for_tables/ folder.
This regenerates LaTeX table bodies from archived result files.
It does not retrain models; input counts are taken from retained run manifests.
"""
from pathlib import Path
import csv, json, statistics as stats, re, sys, math
ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence_for_tables'
OUT=ROOT/'regenerated_tables'
OUT.mkdir(exist_ok=True)

def read(name):
 with (E/name).open(newline='') as f:return list(csv.DictReader(f))
def fmt(n,p=6): return f'{float(n):.{p}f}'
def comma(n):return f'{int(n):,}'
def tabular(headers,rows,spec):
 if any(len(r)!=len(headers) for r in rows):raise ValueError('column count mismatch')
 bs='\\'
 pieces=['\\begin{tabular}{'+spec+'}','\\toprule',' & '.join(headers)+' '+bs+bs,'\\midrule']
 pieces += [' & '.join(map(str,row))+' '+bs+bs for row in rows]
 pieces += ['\\bottomrule','\\end{tabular}']
 return '\n'.join(pieces)

data=json.loads((E/'input_dataset_counts.json').read_text())
T={}
main1=[]
for name,title,label in [('CIC-IDS2017','CIC-IDS2017, Friday source','DDoS'),('CIC-DDoS2019 redistributed','Redistributed CIC-DDoS2019, exploratory target','Multiple family labels'),('CSE-CIC-IDS2018 Wednesday-21','CSE-CIC-IDS2018, 21 February target','HOIC and LOIC-UDP')]:
 d=data[name];main1.append([title,comma(d['rows']),comma(d['benign']),comma(d['attack']),label]);assert d['rows']==d['benign']+d['attack']
T['main_T1']=tabular(['File and role','Rows','Benign','Attack','Positive classes'],main1,'@{}lrrrl@{}')
summary=read('reports/score_diagnostics/summary.csv')
idx={(r['dataset'],r['threshold_name']):r for r in summary}
main2=[]
for dataset,rule,name,rulename in [('CIC-IDS2017_holdout','source_1pct_FPR','2017 source holdout','score \\textgreater{} 0.01'),('CIC-DDoS2019_external','source_1pct_FPR','Redistributed 2019','score \\textgreater{} 0.01'),('CIC-DDoS2019_external','fixed_0.50','Redistributed 2019','score ≥ 0.50')]:
 d=idx[(dataset,rule)];main2.append([name,rulename,fmt(d['roc_auc']),fmt(d['attack_recall'],5 if rule=='fixed_0.50' else 6),fmt(d['false_positive_rate'],5 if rule=='fixed_0.50' else 6)])
T['main_T2']=tabular(['Evaluation','Decision rule','ROC-AUC','Attack recall','FPR'],main2,'@{}llrrr@{}')
runs=read('research/results/experiment9_20seeds/per_run.csv'); original=[r for r in runs if r['arm']=='original']; assert len(original)==20
main3=[]
for label,key,scale,precision in [('2019 ROC-AUC','external_auc',1,6),('Attack recall (\\%)','external_recall',100,2),('Benign FPR (\\%)','external_fpr',100,2)]:
 v=[float(r[key])*scale for r in original]
 main3.append([label,fmt(stats.mean(v),precision),fmt(stats.stdev(v),precision),fmt(min(v),precision),fmt(max(v),precision)])
T['main_T3']=tabular(['Metric','Mean','SD across seeds','Minimum','Maximum'],main3,'@{}lrrrr@{}')
paired=read('paired_by_seed.csv');summ=read('paired_summary.csv'); sidx={(r['target'],r['metric']):r for r in summ}
main4=[]
for target,label in [('2019','Redistributed 2019'),('2018','2018 daily file')]:
 rs=[r for r in paired if r['target']==target];assert len(rs)==20
 a=[float(r['A_auc']) for r in rs];b=[float(r['B_auc']) for r in rs]
 d=sidx[(target,'delta_auc_B_minus_A')]
 sd=f'{float(d["mean"]):+.4f}\\;[{float(d["nominal_seed_level_95pct_lower"]):+.4f},{float(d["nominal_seed_level_95pct_upper"]):+.4f}]'
 main4.append([label,f'${stats.mean(a):.4f}\\pm{stats.stdev(a):.4f}$',f'${stats.mean(b):.4f}\\pm{stats.stdev(b):.4f}$','$'+sd+'$',f'{sum(float(r["delta_auc_B_minus_A"])>0 for r in rs)}/20'])
T['main_T4']=tabular(['Target file','A AUC (mean $\\pm$ SD)','B AUC (mean $\\pm$ SD)','Mean B$-$A AUC (95\\% interval)','B above A'],main4,'@{}lrrrl@{}')
protocol=read('research/results/experiment10_protocol_scoring/stratified_metrics.csv')
lookup={r['group']:r for r in protocol if r['dataset']=='target_2019'}
main5=[]
for key,label in [('TCP (explicit)','TCP (\\texttt{6})'),('UDP (explicit)','UDP (\\texttt{17})'),('Other/unknown protocol','Other/unspecified (\\texttt{0})')]:
 r=lookup[key];main5.append([label,comma(r['attack_n']),comma(r['benign_n']),fmt(r['recall']),fmt(r['fpr']),fmt(r['auc_within_group'])])
T['main_T5']=tabular(['Protocol in evaluated 2019 CSV','Attack rows','Benign rows','Attack recall','Benign FPR','Within-group AUC'],main5,'@{}lrrrrr@{}')
main6=[]
for name,lab in [('research/results/experiment11_2018/within_target_summary.csv','2018 → 2018'),('research/results/experiment11_within_target/within_target_summary.csv','2019 → 2019')]:
 d=read(name)[0];main6.append([lab,comma(d['test_rows_used']),fmt(d['roc_auc'],10),fmt(d['attack_recall']),fmt(d['benign_fpr']),comma(d['tp']),comma(d['fp'])])
T['main_T6']=tabular(['Target-trained → same-file test','Test rows','ROC-AUC','Attack recall','Benign FPR','TP','FP'],main6,'@{}lrrrrrr@{}')
ops=read('reports/score_diagnostics/operating_point_audit.csv')
opidx={(r['dataset'],r['rule']):r for r in ops}
supp1=[]
for key,lab in [('>0.0','score \\textgreater{} 0.00'),('>0.01','score \\textgreater{} 0.01'),('>=0.01','score ≥ 0.01'),('>0.05','score \\textgreater{} 0.05'),('>0.1','score \\textgreater{} 0.10'),('>=0.5','score ≥ 0.50')]:
 r=opidx[('EXTERNAL TARGET',key)];supp1.append([lab,fmt(r['attack_recall'],5),fmt(r['benign_fpr'],5)])
T['supp_S1']=tabular(['Decision rule','Attack recall','FPR'],supp1,'@{}lrr@{}')
abl=read('reports/experiment_05/metrics.csv');design=json.loads((E/'reports/experiment_05/design.json').read_text());am={(r['arm'],r['dataset'],r['operating_point']):r for r in abl}
supp2=[]
for arm,label in [('A_common','A: common'),('B_core','B: core'),('C_expanded','C: expanded aliases')]:
 k=(arm,'external_2019','source_calibrated');r=am[k]; count=len(design['feature_sets'][arm]);supp2.append([label,str(count),fmt(r['roc_auc']),fmt(r['average_precision']),fmt(r['recall']),fmt(r['fpr'])])
T['supp_S2']=tabular(['Representation','Features','External ROC-AUC','External AP','External recall','External FPR'],supp2,'@{}lrrrrr@{}')
ded=read('reports/experiment_06/fixed_split_dedup_comparison.csv');did={r['variant']:r for r in ded}
supp3=[]
for key,label in [('original_training','Original training partition'),('training_only_dedup','Deduplicate training partition only')]:
 r=did[key];supp3.append([label,comma(r['train_rows']),fmt(r['source_holdout_auc']),fmt(r['external_auc']),fmt(r['external_recall']),fmt(r['external_fpr'])])
T['supp_S3']=tabular(['Training variant','Training rows','Source AUC','External AUC','External recall','External FPR'],supp3,'@{}lrrrrr@{}')
fams=read('research/results/experiment10_protocol_followup/family_protocol_performance.csv')
famlookup={(r['family'],r['protocol_group']):r for r in fams};supp4=[]
for family in ['DrDoS_NTP','TFTP','Portmap','UDP-lag']:
 t=famlookup[(family,'TCP (explicit)')];u=famlookup[(family,'UDP (explicit)')]
 supp4.append([family.replace('_','\\_'),comma(t['attack_rows']),fmt(t['recall']),comma(u['attack_rows']),fmt(u['recall'])])
T['supp_S4']=tabular(['Named family','TCP attacks','TCP recall','UDP attacks','UDP recall'],supp4,'@{}lrrrr@{}')

for key,tab in T.items(): (OUT/(key+'.tex')).write_text(tab+'\n')
print('Regenerated',len(T),'tabular bodies from archived CSV/JSON files.')
# Optionally compare regenerated tabular bodies with a compiled-source project.
import argparse
parser=argparse.ArgumentParser(description='Rebuild tables from saved experiment results')
parser.add_argument('--overleaf',type=Path,default=None,
                    help='Directory containing main.tex and ESM_1_source.tex')
args=parser.parse_args()
if args.overleaf is not None:
 pat=r'\\begin\{tabular\}\{[^\n]*\}[\s\S]*?\\end\{tabular\}'
 for file,keys,offset in [('main.tex',[f'main_T{i}' for i in range(1,7)],0),('ESM_1_source.tex',[f'supp_S{i}' for i in range(1,5)],1)]:
  body=(args.overleaf/file).read_text(); all_tabs=re.findall(pat,body)
  assert len(all_tabs)==len(keys)+offset,(file,len(all_tabs),len(keys),offset)
  for i,key in enumerate(keys):
   a=all_tabs[i+offset].strip();b=T[key].strip()
   if a != b:
    import difflib
    dif=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='manuscript/'+key,tofile='regenerated/'+key))
    (OUT/(key+'_difference.txt')).write_text(dif)
    print('MISMATCH',key);print(dif[:900])
    sys.exit(1)
   print('PASS',key)
 print('All 10 manuscript table bodies match the saved numerical results.')
else:
 print('No manuscript directory supplied; created table bodies without comparing LaTeX.')
