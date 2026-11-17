#!/usr/bin/env python3
"""Generate manuscript figures from supplied/verified JNSM experiment result files.

Inputs:
 - archived run summaries in the adjacent jnsm_plot_sources directory
 - paired_by_seed.csv containing the 20-seed feature contrasts
 - two three-feature template count comparisons pasted by author in terminal

Outputs: PDF/SVG/PNG figures, captions, source-data CSVs, Overleaf instructions.
This script is not an experimental reproduction; it draws validated aggregate results.
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from scipy import stats

ROOT=Path(__file__).resolve().parent
S=ROOT/'jnsm_plot_sources'
OUT=ROOT
F=OUT/'figures'
D=OUT/'figure_data'
for d in (F,D): d.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':10.0,'axes.labelsize':10.2,'axes.titlesize':10.5,
 'axes.titleweight':'semibold','xtick.labelsize':9.1,'ytick.labelsize':9.1,
 'legend.fontsize':9.0,'figure.dpi':180,'savefig.dpi':600,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,
 'axes.axisbelow':True,'grid.alpha':0.25,'lines.linewidth':1.4,
 'axes.linewidth':0.8,'xtick.major.size':3.5,'ytick.major.size':3.5,
 'savefig.facecolor':'white',
})

P=None  # Matplotlib's default cycle colors only

def save(fig, name):
    for ext in ['pdf','svg','png']:
        fig.savefig(F/(name+'.'+ext),bbox_inches='tight',pad_inches=0.11,dpi=600 if ext=='png' else None)
    plt.close(fig)

def panel_label(ax,label):
    ax.text(-0.09,1.13,label, transform=ax.transAxes,weight='bold',size=10.3,ha='left',va='top')

def grid_y(ax):
    ax.grid(True,axis='y',linestyle=':',linewidth=0.8)

def pvalue(x):return f'{100*x:.1f}%'

# Figure 1: initial source-to-external transfer at source-calibrated threshold
summary=pd.read_csv(S/'reports/score_diagnostics/summary.csv')
source=summary.query('dataset=="CIC-IDS2017_holdout" and threshold_name=="source_1pct_FPR"').iloc[0]
target=summary.query('dataset=="CIC-DDoS2019_external" and threshold_name=="source_1pct_FPR"').iloc[0]
assert source['threshold']==target['threshold']==0.01 and int(source.n)==45149 and int(target.n)==431371
fig,ax=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained')
names=['2017 source\nholdout','2019 external\ntarget']
x=np.arange(2)
for k,(metric,title,ylim) in enumerate([('attack_recall','Attack recall',105),('false_positive_rate','Benign false-positive rate',16)]):
 vals=[100*source[metric],100*target[metric]]
 bars=ax[k].bar(x,vals,width=.54,edgecolor='0.18',linewidth=.55)
 ax[k].set_xticks(x,names);ax[k].set_ylabel('Percent of class (%)');ax[k].set_title(title,pad=10)
 ax[k].set_ylim(0,ylim)
 for b,v in zip(bars,vals):ax[k].text(b.get_x()+b.get_width()/2,v+ylim*.025,f'{v:.2f}%',ha='center',va='bottom',fontsize=9.3)
 grid_y(ax[k]);panel_label(ax[k],'(a)' if k==0 else '(b)')
save(fig,'F01_source_to_target_alerts')

# Figure 2: Experiment 9 20-seed original/dedup/placebo
per=pd.read_csv(S/'research/results/experiment9_20seeds/per_run.csv')
pa=pd.read_csv(S/'research/results/experiment9_20seeds/paired_by_seed.csv').sort_values('seed')
assert set(pa.seed)==set(range(42,62)) and len(per)==80 and pa['original'].round(10).equals(per.query('arm=="original"').sort_values('seed')['external_auc'].reset_index(drop=True).round(10))
fig,ax=plt.subplots(1,2,figsize=(7.5,3.20),layout='constrained')
xs=pa.seed.to_numpy()
for y,label,marker in [('original','Original','o'),('deduplicated','Deduplicated','s'),('placebo','Placebo (2 subsets, mean)','^')]:
 ax[0].plot(xs,pa[y].to_numpy(),marker=marker,markersize=3.1,label=label,lw=1.1)
ax[0].set_title('External ROC-AUC by model seed',pad=9)
ax[0].set_ylabel('2019 ROC-AUC');ax[0].set_xlabel('Random Forest seed')
ax[0].set_ylim(.67,.96);ax[0].set_yticks(np.arange(.70,.96,.05));ax[0].legend(loc='upper right',frameon=True,fontsize=8.0)
for y,label,marker in [('dedup_minus_original','Deduplicated − original','o'),('placebo_minus_original','Placebo − original','s')]:
 ax[1].plot(xs,pa[y].to_numpy(),marker=marker,markersize=3.2,lw=1.1,label=label)
ax[1].axhline(0,color='0.20',ls='--',lw=1.1)
ax[1].set_title('Paired ROC-AUC differences',pad=9)
ax[1].set_ylabel('Difference in external ROC-AUC');ax[1].set_xlabel('Random Forest seed')
ax[1].set_ylim(-.14,.16);ax[1].legend(loc='upper left',fontsize=8.0,frameon=True)
for a,letter in zip(ax,['(a)','(b)']):
 a.set_xlim(41.5,61.5);a.set_xticks([42,46,50,54,58,61]);grid_y(a);panel_label(a,letter)
save(fig,'F02_exp9_seed_sensitivity')

# Figure 3: 57 vs 37 feature, 20 paired source-grouped splits with other protocol than exp9
ab=pd.read_csv(ROOT/'paired_by_seed.csv')
assert len(ab)==40 and set(ab.seed)==set(range(42,62)) and set(ab.target.astype(str))=={'2018','2019'}
# verify raw deltas
for met in ['auc','recall','fpr']:
    assert np.allclose(ab[f'delta_{met}_B_minus_A'],ab[f'B_{met}']-ab[f'A_{met}'],atol=1e-12)
fig,axs=plt.subplots(1,2,figsize=(7.35,3.22),layout='constrained')
for axx,year,letter in [(axs[0],'2019','(a)'),(axs[1],'2018','(b)')]:
 t=ab[ab.target.astype(str)==year].sort_values('seed')
 dv=t.delta_auc_B_minus_A.to_numpy();mean=float(dv.mean())
 axx.plot(t.seed,dv,marker='o',markersize=3.4,lw=1.1,label='Paired difference (B − A)')
 axx.axhline(0,lw=1.1,ls='--',color='0.20',label='No change')
 axx.axhline(mean,lw=1.2,ls=':',color='0.45',label='20-seed mean')
 axx.set_title(f'{year} external target',pad=9)
 axx.set_xlabel('Grouped-source split / model seed')
 axx.set_ylabel('ROC-AUC difference (37 − 57 features)')
 axx.set_xlim(41.5,61.5);axx.set_xticks([42,46,50,54,58,61]);grid_y(axx);panel_label(axx,letter)
 ci=stats.t.interval(0.95,len(dv)-1,loc=mean,scale=stats.sem(dv))
 axx.text(.03,.055,f'Mean {mean:+.3f}\n95% seed interval [{ci[0]:+.3f}, {ci[1]:+.3f}]',
          transform=axx.transAxes,size=8.4,va='bottom',ha='left',bbox=dict(facecolor='white',edgecolor='0.78',boxstyle='round,pad=.3'))
 axs[0].set_ylim(-.065,.14)
 axs[1].set_ylim(-.46,.53)
axs[0].legend(fontsize=8.1,loc='upper right',frameon=True)
save(fig,'F03_exp78_feature_representation_20seeds')

# Figure 4: Experiment 10 explicit protocol groups and named-family attack recall
prot=pd.read_csv(S/'research/results/experiment10_protocol_followup/stratified_metrics.csv')
prot=prot[prot.dataset=='target_2019']
order=['TCP (explicit)','UDP (explicit)','Other/unknown protocol']
prot=prot.set_index('group').loc[order].reset_index()
assert prot.attack_n.sum()==333540 and prot.benign_n.sum()==97831 and prot.fp.sum()==10421
fam=pd.read_csv(S/'research/results/experiment10_protocol_followup/family_protocol_performance.csv')
focus=['DrDoS_NTP','TFTP','Portmap','UDP-lag']
fig,axs=plt.subplots(1,2,figsize=(8.15,3.36),layout='constrained',gridspec_kw={'width_ratios':[.91,1.15]})
a=axs[0];xx=np.arange(3);width=.35
for j,(metric,label) in enumerate([('recall','Attack recall'),('fpr','Benign FPR')]):
 vals=100*prot[metric].to_numpy()
 bs=a.bar(xx+(-.5+j)*width,vals,width,label=label,linewidth=.45,edgecolor='0.22',hatch='///' if j else '')
 for b,v in zip(bs,vals): a.annotate(f'{v:.1f}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=7.7)
a.set_xticks(xx,['TCP (6)','UDP (17)','Other (0)'])
a.set_ylabel('Percent within protocol group (%)');a.set_title('Detection by explicit protocol',pad=9)
a.set_ylim(0,113);a.legend(frameon=True,loc='upper center',fontsize=7.8)
grid_y(a);panel_label(a,'(a)')
# family grouped bars protocol is separate context from (a), legend explicitly labels TCP/UDP
b=axs[1];x=np.arange(4);w=.36
for j,(proto,label) in enumerate([(6,'TCP'),(17,'UDP')]):
 sub=fam[(fam.family.isin(focus))&(fam.protocol_value==proto)].set_index('family').loc[focus]
 vals=(100*sub['recall'].to_numpy()); n=sub.attack_rows.to_numpy()
 bs=b.bar(x+(-.5+j)*w,vals,w,label=label,edgecolor='0.22',linewidth=.45,hatch='///' if j else '')
 for bar,v,ni in zip(bs,vals,n):
  b.annotate(f'{v:.1f}%',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8.3)
b.set_xticks(x,focus,fontsize=8.3);b.set_ylabel('Attack recall (%)')
b.set_title('Within-family recall by protocol',pad=9)
b.set_ylim(0,110);b.legend(frameon=True,loc='upper left',fontsize=8.2);grid_y(b);panel_label(b,'(b)')
save(fig,'F04_exp10_protocol_and_family')

# Figure 5: 2018 cross-file class-associated three-feature tuple observations
# Source here is the exact counted terminal output supplied by author in conversation.
templates=pd.DataFrame([
 ('2017','benign',97718,0,1632),('2017','attack',128027,0,0),
 ('2018','benign',360833,358523,103),('2018','attack',687742,0,522262),
 ('2019','benign',97831,0,5235),('2019','attack',333540,0,1215)
 ],columns=['dataset','class','class_n','tuple_935_187_5','tuple_0_0_2'])
assert templates.query('dataset=="2018"').class_n.sum()==1048575
for c in ('tuple_935_187_5','tuple_0_0_2'):
 templates[c+'_pct']=100*templates[c]/templates.class_n
templates.to_csv(D/'author_terminal_template_counts.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.25,3.05),layout='constrained')
for axx,(col,title) in zip(axs,[('tuple_935_187_5_pct','Template (935, 187, 5)'),('tuple_0_0_2_pct','Template (0, 0, 2)')]):
 labs=['2017','2018','2019'];xx=np.arange(3);w=.35
 for j,cat in enumerate(['benign','attack']):
  sub=templates[templates['class']==cat].set_index('dataset').loc[labs]
  v=sub[col].to_numpy()
  bars=axx.bar(xx+(-.5+j)*w,v,w,label=cat.capitalize(),linewidth=.5,edgecolor='0.2',hatch='///' if j else '')
  for bb,z in zip(bars,v):
   fmt='0' if z==0 else f'{z:.3f}' if z<1 else f'{z:.2f}'
   axx.annotate(fmt,(bb.get_x()+bb.get_width()/2,z),xytext=(0,4 if z>=1 else 8),textcoords='offset points',ha='center',va='bottom',fontsize=8.1)
 axx.set_xticks(xx,labs);axx.set_ylim(0,114);axx.set_yticks(np.arange(0,101,20))
 axx.set_title(title,pad=10);axx.set_ylabel('Within-class occurrence (%)');axx.set_xlabel('Evaluated file')
 axx.legend(loc='upper left',frameon=True,fontsize=8.1);grid_y(axx)
 panel_label(axx,'(a)' if col.startswith('tuple_935') else '(b)')
save(fig,'F05_exp11_cross_file_feature_templates')

# Supplement S1: source/target threshold condition tie behavior at 0.01
ops=pd.read_csv(S/'reports/score_diagnostics/operating_point_audit.csv')
fig,axs=plt.subplots(1,2,figsize=(7.25,3.1),layout='constrained')
for i,(metric,title,lim) in enumerate([('attack_recall','Attack recall',105),('benign_fpr','Benign FPR',17.5)]):
 a=axs[i];xx=np.arange(2);w=.36
 for j,(ds,label) in enumerate([('SOURCE HOLDOUT','2017 holdout'),('EXTERNAL TARGET','2019 target')]):
  sub=ops[(ops.dataset==ds)&ops.rule.isin(['>0.01','>=0.01'])].set_index('rule').loc[['>0.01','>=0.01']]
  v=100*sub[metric].to_numpy()
  bs=a.bar(xx+(-.5+j)*w,v,w,label=label,edgecolor='0.25',linewidth=.5,hatch='///' if j else '')
  for bb,z in zip(bs,v):a.annotate(f'{z:.2f}',(bb.get_x()+bb.get_width()/2,z),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8.3)
 a.set_title(title,pad=9);a.set_xticks(xx,['score > 0.01','score ≥ 0.01']);a.set_ylim(0,lim)
 a.set_ylabel('Percent of class (%)');a.legend(loc='lower right' if i==0 else 'upper left',fontsize=8.2,frameon=True)
 grid_y(a);panel_label(a,'(a)' if i==0 else '(b)')
save(fig,'S01_threshold_tie_effect')

# Supplement S2: compare univariate AUC (train-direction fixed) with RF AUC; panel each 2018/2019.
fig,axs=plt.subplots(1,2,figsize=(7.6,3.2),layout='constrained')
for i,year in enumerate(['2018','2019']):
 univ=pd.read_csv(S/f'research/results/experiment11_shortcut_audit_{year}/single_feature_auc.csv')
 univ=univ.sort_values('test_single_feature_auc_fixed_direction',ascending=False)
 # top 5 distinctive input columns; de-duplicate closely equivalent forms by input column names if any
 if year=='2018':
  wanted=['Fwd Packet Length Max','Total Fwd Packets','Subflow Fwd Bytes','Fwd Header Length','Bwd IAT Min']
 else:
  wanted=['Fwd Packet Length Min','Bwd Packets/s','Fwd Packet Length Mean','Bwd Header Length','Total Backward Packets']
 t=univ.set_index('feature').loc[wanted]
 a=axs[i]; yy=np.arange(len(t));v=t.test_single_feature_auc_fixed_direction.to_numpy()
 bars=a.barh(yy,v,height=.65,edgecolor='0.2',linewidth=.4)
 for bb,z in zip(bars,v):a.annotate(f'{z:.3f}',(z,bb.get_y()+bb.get_height()/2),xytext=(4,0),textcoords='offset points',ha='left',va='center',fontsize=8.2)
 a.set_yticks(yy,[q.replace('Packet Length','Pkt Len').replace('Total Backward Packets','Total Bwd Pkts') for q in wanted],fontsize=8.1)
 a.invert_yaxis();a.set_xlim(.5,1.075);a.set_xticks([.5,.6,.7,.8,.9,1.0]);a.set_xlabel('Held-out raw-feature ROC-AUC')
 a.set_title(f'{year} within-file evaluation',pad=9)
 model=pd.read_csv(S/f'research/results/experiment11_{"2018" if year=="2018" else "within_target"}/within_target_summary.csv').iloc[0]
 a.axvline(model.roc_auc,ls='--',lw=1.1,color='0.3',label=f'Full RF AUC ≈ {model.roc_auc:.5f}')
 # Full-model reference is documented separately in the figure caption to keep bars legible.
 grid_y(a);panel_label(a,'(a)' if i==0 else '(b)')
save(fig,'S02_exp11_single_feature_vs_full_model')

captions='''# Journal figure captions (separate from figure artwork)

All values below are drawn from the saved experiment result CSVs, except the exact three-feature tuple counts in F05, which were transcribed from the author's terminal audit. These figures are descriptive, not claims of causality or independent deployment evidence. Vector PDF/SVG and 600-dpi PNG versions are provided.

## Main-text candidates

**F01 — Source-trained alert reliability.** Attack recall (a) and benign false-positive rate (b) for the original seed-42 57-feature Random Forest on its CIC-IDS2017 holdout and on redistributed CIC-DDoS2019. Both use the strict threshold score > 0.01 calibrated exclusively from source benign flows. The same decision rule gives 99.98% versus 58.11% recall and 0.64% versus 10.65% benign FPR. Different traffic populations are compared; this is a descriptive external test of one fitted model, not an estimate of deployed alert burden.

**F02 — Training-duplicate sensitivity over 20 model seeds.** (a) External ROC-AUC on redistributed CIC-DDoS2019 for the original, deduplicated and class-matched placebo-removal arms; (b) paired ROC-AUC differences versus the original arm. The source training, calibration and holdout partitions were fixed for this experiment; the placebo curve is the mean of two once-selected matched subsets per seed. The mean dedup-minus-original AUC difference was −0.00269; the observed direction was not consistent across seeds. A line connects seed-ordered settings for readability; seed is not time.

**F03 — Paired feature-representation comparison over 20 source-grouped runs.** Changes in external ROC-AUC (37 minus 57 features) for (a) redistributed CIC-DDoS2019 and (b) the evaluated CSE-CIC-IDS2018 daily file. Each point contrasts representations trained under the same source grouping, split and model seed. Dashed lines indicate zero; dotted lines indicate the across-seed mean. Intervals shown in the panel are nominal seed-level t intervals, not uncertainty across independent network captures. The 37-feature subset was originally explored using 2019, so that comparison remains exploratory. This experiment varied the source partition with the seed and is distinct from the fixed-partition experiment in F02.

**F04 — Protocol-associated differences in detection.** (a) Attack recall and benign FPR within explicit Protocol groups in redistributed CIC-DDoS2019 for the 2017-trained seed-42 model; (b) within-named-family recall by explicit TCP versus UDP protocol for attack-family cells containing at least 100 attack flows. The Protocol=0 group is included in panel (a) as other/unspecified, not treated as an invalid record. The source 2017 CSV has no explicit Protocol column, so the results show associations and cannot establish a causal protocol effect or source protocol coverage.

**F05 — Cross-file prevalence of two three-feature flow signatures.** The fraction of benign and attack-labeled records matching (a) (forward packet length maximum, mean, total forward packets) = (935, 187, 5) and (b) (0, 0, 2), in the evaluated 2017, 2018 and 2019 CSVs. The (935,187,5) pattern occurs in 99.36% of 2018 benign flows but nowhere else in the evaluated rows. The (0,0,2) pattern occurs in 75.94% of 2018 attack flows, which are HOIC-dominated, but has different class associations in the other files. Templates were selected post hoc and use only three values; they are not evidence of duplicate complete vectors, incorrect labels or comparable extraction semantics.

## Optional supporting material

**S01 — Score tie and decision-rule sensitivity.** Attack recall and benign FPR under strict score > 0.01 versus inclusive score ≥ 0.01 in the initial seed-42 Random Forest on the source holdout and external 2019 file. The large change in external recall shows that a mass of scores at 0.01 makes the observed alert counts sensitive to the exact inequality. This is a model-specific tie effect, not a calibrated probability comparison.

**S02 — Univariate ranking signal within target files.** Held-out, train-oriented raw-feature ROC-AUC for five representative numeric features in each within-target dataset, with a dashed reference for the full Random Forest's within-target ROC-AUC. The raw-feature orientation was chosen from the training subset only; these bars do not represent trained nonlinear single-feature classifiers. Exact 57-feature-vector group disjointness does not enforce time-, host- or session-disjointness, and near-perfect internal discrimination does not establish feature semantic validity.
'''
(OUT/'Figure_Captions.md').write_text(captions,encoding='utf-8')
tex=r'''% Upload the PDFs in figures/ to your Overleaf project.
% Use \usepackage{graphicx} in your preamble.
% The PDF is vector, so it remains sharp in the compiled paper.
%
% Figures are candidates, not automatically inserted in the manuscript.
% Follow the manuscript's existing labels and numbering conventions.
'''
for idx, (name,caption,label) in enumerate([
 ('F01_source_to_target_alerts','Source-trained alert recall and false-positive rates under a source-calibrated strict decision threshold (seed 42).','fig:source-alerts'),
 ('F02_exp9_seed_sensitivity','External ROC-AUC and paired row-removal effects over 20 fixed-partition Random Forest seeds.','fig:seed-sensitivity'),
 ('F03_exp78_feature_representation_20seeds','Paired 37-versus-57-feature external AUC differences across 20 source-grouped runs.','fig:feature-contrast'),
 ('F04_exp10_protocol_and_family','Protocol- and attack-family-associated detection differences on redistributed CIC-DDoS2019.','fig:protocol-family'),
 ('F05_exp11_cross_file_feature_templates','Three-feature class-associated template prevalence across the evaluated files.','fig:templates'),
 ('S01_threshold_tie_effect','Sensitivity of the original classifier to strict versus inclusive score thresholds at 0.01.','fig:threshold-tie'),
 ('S02_exp11_single_feature_vs_full_model','Within-file univariate raw-feature ranking versus the full model.','fig:univariate')
 ]):
 tex+=f'''\n\\begin{{figure}}[!htbp]\n  \\centering\n  \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n\\end{{figure}}\n'''.replace('\\\\','\\')
(OUT/'Overleaf_figure_snippets.tex').write_text(tex,encoding='utf-8')
readme='''# JNSM figures — source-derived manuscript candidates

Seven figures were generated from the original experimental CSV exports and the author's counted 2018 three-feature template audit. They are designed as concise, two-panel journal figures, with plots generated directly from stored values.

- Five main-text candidates (F01–F05)
- Two optional supporting figures (S01–S02)
- PDF (vector, best for Overleaf), SVG (editable), and PNG (600 dpi)
- `Figure_Captions.md` contains the extended scientific caveats; do not use the short LaTeX snippet captions alone without reviewing these caveats.
- `Overleaf_figure_snippets.tex` provides optional LaTeX inclusion examples. Upload the `figures/` folder to the existing Overleaf project. To minimize main-text clutter, insert only figures that materially clarify the manuscript and move others to Online Resource 1.
- The script `reproduce_figures.py` is provided for figure reconstruction. It reads the adjacent saved experiment summaries and `paired_by_seed.csv`. It does not reconstruct traffic flows or train models.
- `figure_data/author_terminal_template_counts.csv` explicitly documents the template values transcribed from the author's 2018 cross-file terminal output, rather than claiming the current archive contains raw CSV flows.

## Upstream source files

- `JNSM_evidence_from_Mac.zip` provided by author, includes Exp9-11 CSV results and Exp1 threshold summaries.
- `paired_by_seed.csv` provided by author for A/B 20-seed results.
- Original raw packet/flow CSVs were not required to draw these figures. All figure statements remain bounded by their experiment protocols.

## Numerical assertions

The plotting script checks seed counts, paired differences, and the stored protocol-group totals. Source tables were independently audited during the previous manuscript stages.

## Suggested selection

Prioritize F01, F02, F03 and F04 if space permits. F05 may be particularly informative in the benchmark-validity discussion. Use S01 and S02 only when the supplement adds explanatory value. There is no obligation to include all seven.
'''
(OUT/'GENERATED_FIGURES_README.md').write_text(readme,encoding='utf-8')
print('Wrote', len(list(F.glob('*.pdf'))),'PDF figures, with SVG and PNG')
