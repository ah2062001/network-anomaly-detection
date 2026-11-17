#!/usr/bin/env python3
"""Experiment 11 follow-up: fixed-partition, diagnostic feature-family ablations.

Uses the original 2018 Experiment 11 row indices, 57 source-ordered features,
train-median imputation, RF settings, and 99th percentile calibration threshold.
A baseline REF CHECK must pass against the original saved 2018 score.

IMPORTANT: projected (ablated) vectors can newly overlap across original splits.
We report overlap and matched baseline-vs-ablation metrics on a test subset whose
ablated vectors are unseen in either train or calibration. This is exploratory,
not connection/host/time-disjoint, and does not establish causal mechanisms.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

SCRIPTDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTDIR.parent / 'exp11_within_target'))
import experiment11_within_target as orig

# Exclude the exact three-value benign signature directly, then its related
# packet-size/count feature family (including strongly redundant aliases).
THREE = {
    'Fwd Packet Length Max','Fwd Packet Length Mean','Total Fwd Packets',
}
PACKET_FAMILY = THREE | {
    'Fwd Packet Length Min','Fwd Packet Length Std',
    'Bwd Packet Length Max','Bwd Packet Length Mean','Bwd Packet Length Min',
    'Bwd Packet Length Std',
    'Avg Fwd Segment Size','Avg Bwd Segment Size',
    'Subflow Fwd Packets','Subflow Fwd Bytes',
    'Subflow Bwd Packets','Subflow Bwd Bytes',
    'Total Backward Packets', 'Fwd Header Length','Bwd Header Length',
    'Fwd Packets/s','Bwd Packets/s', 'Flow Packets/s','Flow Bytes/s',
    'Packet Length Mean','Packet Length Std','Packet Length Variance',
    'Down/Up Ratio',
}


def restore_saved_order(data_sorted, original_indices):
    # Streamed file rows arrive in ascending order; restore exact original
    # randomized row order for bootstrap reproducibility.
    order = np.argsort(np.argsort(original_indices))
    return data_sorted[order]


def stream_exact_saved_rows(csv_path, mapping, features, id_sets, chunk_size):
    wanted = set(mapping.values()) | {'Label'}
    ordered = [mapping[f] for f in features]
    chunks = {key:{'x':[], 'y':[], 'family':[]} for key in id_sets}
    sorted_idxs = {k: np.sort(v) for k,v in id_sets.items()}
    pos = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             usecols=lambda c: str(c).strip() in wanted, low_memory=False):
        chunk.columns = [str(s).strip() for s in chunk.columns]
        if set(chunk.columns) != wanted:
            raise RuntimeError(f'Read columns mismatch: {wanted-set(chunk.columns)}')
        end=pos+len(chunk)
        for name, idx in sorted_idxs.items():
            lo=int(np.searchsorted(idx,pos)); hi=int(np.searchsorted(idx,end))
            if lo==hi: continue
            sub=chunk.iloc[idx[lo:hi]-pos]
            x=np.column_stack([pd.to_numeric(sub[field], errors='coerce').to_numpy(dtype=np.float32)
                               for field in ordered])
            x[~np.isfinite(x)]=np.nan
            x[x == 0] = np.float32(0)
            chunks[name]['x'].append(x)
            chunks[name]['y'].append(orig.parse_y(sub['Label'],'2018'))
            chunks[name]['family'].append(sub['Label'].astype('string').str.strip().to_numpy(dtype=str))
        pos=end
    out={}
    for name, idx in id_sets.items():
        xs=chunks[name]
        if not xs['x']: raise RuntimeError(f'No rows streamed for {name}')
        x=restore_saved_order(np.concatenate(xs['x']),idx)
        y=restore_saved_order(np.concatenate(xs['y']),idx)
        fam=restore_saved_order(np.concatenate(xs['family']),idx)
        if x.shape != (len(idx),len(features)):
            raise RuntimeError(f'Unexpected shape {name} {x.shape}')
        out[name]=(x,y,fam)
    return out,pos


def metric_row(y, score, threshold):
    if len(np.unique(y)) != 2:
        return None
    pred=score>threshold
    tn,fp,fn,tp=(int(v) for v in confusion_matrix(y,pred,labels=[0,1]).ravel())
    return {'auc':float(roc_auc_score(y,score)),
            'average_precision':float(average_precision_score(y,score)),
            'threshold':float(threshold),'recall':tp/(tp+fn),
            'fpr':fp/(fp+tn),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'test_rows':int(len(y)),'attack_rows':int(np.sum(y)),
            'benign_rows':int(np.sum(y==0))}


def fit_score(features, subset, blocks, cfg):
    itr,ica,ite=(blocks[k][0][:,subset] for k in ('train','cal','test'))
    ytr=blocks['train'][1]; yca=blocks['cal'][1]
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    xt=imp.fit_transform(itr)
    xc=imp.transform(ica)
    xe=imp.transform(ite)
    if xt.shape[1] != len(subset) or not all(np.isfinite(a).all() for a in (xt,xc,xe)):
        raise RuntimeError('Imputation failed or feature shape changed')
    model=RandomForestClassifier(n_estimators=int(cfg['n_estimators']),
        class_weight=cfg['class_weight'], min_samples_leaf=int(cfg['min_samples_leaf']),
        max_leaf_nodes=int(cfg['max_leaf_nodes']), random_state=int(cfg['random_state']),
        n_jobs=int(cfg['n_jobs']))
    model.fit(xt,ytr)
    sc=model.predict_proba(xc)[:,1]
    st=model.predict_proba(xe)[:,1]
    th=float(np.quantile(sc[yca==0],0.99))
    return st,th,float(np.mean(sc[yca==0]>th))


def run(args):
    base=args.results_dir/'target_2018'
    metadata=json.loads((base/'run_metadata.json').read_text(encoding='utf-8'))
    original=pd.read_csv(base/'metrics.csv').iloc[0]
    feats=orig.read_features(args.features_json)
    if feats != metadata['ordered_features'] or metadata['dataset'] != '2018':
        raise RuntimeError('Dataset or original feature ordering disagrees with saved metadata')
    mapping=metadata['resolved_mapping_2017_names_to_target_names']
    if len(feats) != 57 or len(set(mapping.values())) != 57:
        raise RuntimeError('Cannot validate original 57 fields')
    sourcepath=Path(args.path) if args.path else Path(metadata['csv_path'])
    if not sourcepath.is_file():
        sourcepath=orig.choose_dataset_path(args.raw_root,'2018',None)
    digest=orig.digest(sourcepath)
    if digest != orig.EXPECTED['2018']['sha256'] or metadata['sha256'] != digest:
        raise RuntimeError('SHA-256 source mismatch. STOP')
    with np.load(base/'partitions.npz') as npz:
        idx={k:np.asarray(npz[f'used_{k}_idx'],dtype=np.int64) for k in ('train','cal','test')}
    if any(len(np.unique(arr))!=len(arr) for arr in idx.values()):
        raise RuntimeError('Duplicate source indices within partition')
    for i,ki in enumerate(idx):
        for kj in list(idx)[i+1:]:
            if np.intersect1d(idx[ki],idx[kj]).size:
                raise RuntimeError('Saved row indices overlap')
    print(f'2018 SHA-256 verified; loading same used train/cal/test rows... ',flush=True)
    blocks, total=stream_exact_saved_rows(sourcepath,mapping,feats,idx,args.chunk_size)
    if total!=orig.EXPECTED['2018']['rows']:
        raise RuntimeError(f'Row count mismatch {total}')
    for k,block in blocks.items():
        cc=metadata['used_partition_counts'][k]
        if len(block[1])!=cc['rows'] or int(block[1].sum())!=cc['attack']:
            raise RuntimeError(f'{k} labels mismatch original metadata')
    cfg=dict(metadata['model'])
    cfg['random_state']=int(metadata['seed_model'])
    # Preserve the recorded estimator configuration when reading either parameter format.
    model_params={'n_estimators':cfg['n_estimators'],'class_weight':cfg['class_weight'],
                  'min_samples_leaf':cfg['min_samples_leaf'],
                  'max_leaf_nodes':cfg['max_leaf_nodes'],
                  'random_state':cfg['random_state'],'n_jobs':args.n_jobs}
    all_ids=np.arange(57,dtype=int)
    print('Refitting reference 57-feature model (integrity check)...',flush=True)
    baseline_score,baseline_th,baseline_cal_fpr=fit_score(feats,all_ids,blocks,model_params)
    ytest=blocks['test'][1]
    ref=metric_row(ytest,baseline_score,baseline_th)
    if not (abs(ref['auc']-float(original['roc_auc']))<2e-8
            and abs(ref['threshold']-float(original['threshold']))<1e-10
            and ref['tp']==int(original['tp']) and ref['fp']==int(original['fp'])
            and ref['tn']==int(original['tn']) and ref['fn']==int(original['fn'])):
        raise RuntimeError(f'REFERENCE FAILED. Current={ref}; saved={original.to_dict()}')
    print('REFERENCE CHECK PASS: baseline AUC %.10f, TP %d FP %d FN %d, threshold %.8f' %
          (ref['auc'],ref['tp'],ref['fp'],ref['fn'],ref['threshold']),flush=True)
    results=[]; overlaps=[]; family=[]
    for arm, removed in [('full_57',set()),('without_template_three',THREE),
                         ('without_packet_size_count_family',PACKET_FAMILY)]:
        missing=removed-set(feats)
        if missing: raise RuntimeError(f'Ablation field absent: {missing}')
        keep=np.array([j for j,f in enumerate(feats) if f not in removed],dtype=int)
        print(f'\nARM {arm}: {len(keep)}/{len(feats)} features retained',flush=True)
        if arm=='full_57':
            score=baseline_score;threshold=baseline_th;cfpr=baseline_cal_fpr
        else:
            score,threshold,cfpr=fit_score(feats,keep,blocks,model_params)
        row=metric_row(ytest,score,threshold)
        row.update({'arm':arm,'cohort':'all_original_test',
                    'kept_features':len(keep),'calibration_fpr':cfpr})
        results.append(row)
        print('  AUC %.8f | recall %.4f | FPR %.4f | threshold %.6f' %
              (row['auc'],row['recall'],row['fpr'],threshold),flush=True)
        # The original split is disjoint in 57-dim feature space; removing columns
        # may create NEW exact-vector overlaps. Audit them and evaluate a subset
        # where reduced vectors were absent in both training and calibration.
        gr={k:orig.hash_groups(blocks[k][0][:,keep]) for k in blocks}
        seen=np.union1d(gr['train'],gr['cal'])
        train_cal_overlap=int(np.isin(gr['cal'],gr['train']).sum())
        novel=~np.isin(gr['test'],seen)
        badtest=int(np.sum(~novel))
        overlap={'arm':arm,'test_projected_vector_collision_rows':badtest,
                 'test_projected_vector_collision_fraction':badtest/len(ytest),
                 'calibration_projected_vector_collision_with_train_rows':train_cal_overlap,
                 'novel_test_rows':int(novel.sum()),
                 'novel_attack':int(ytest[novel].sum()),
                 'novel_benign':int(np.sum(ytest[novel]==0))}
        overlaps.append(overlap)
        print(f'  Projected-vector test collisions with train/cal: {badtest:,}/{len(ytest):,}',flush=True)
        if len(np.unique(ytest[novel])) == 2:
            ablated=metric_row(ytest[novel],score[novel],threshold)
            ablated.update({'arm':arm,'cohort':'projected_vector_novel_test',
                            'kept_features':len(keep),'calibration_fpr':cfpr})
            results.append(ablated)
            matched=metric_row(ytest[novel],baseline_score[novel],baseline_th)
            matched.update({'arm':arm,'cohort':'baseline_on_same_novel_subset',
                            'kept_features':57,'calibration_fpr':baseline_cal_fpr})
            results.append(matched)
            print('  Novel subset: AUC %.8f (ablated) vs %.8f (full baseline on SAME rows)' %
                  (ablated['auc'],matched['auc']),flush=True)
        else:
            print('  WARNING: projected-vector-novel subset has one class; metrics omitted',flush=True)
        for lab in ('DDOS attack-HOIC','DDOS attack-LOIC-UDP'):
            ix=blocks['test'][2]==lab
            support=int(ix.sum())
            family.append({'arm':arm,'family':lab,'test_attack_n':support,
                           'detected':int(np.sum(score[ix]>threshold)),
                           'recall':float(np.mean(score[ix]>threshold)) if support else None})
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(args.out/'ablation_metrics.csv',index=False)
    pd.DataFrame(overlaps).to_csv(args.out/'projected_vector_overlap.csv',index=False)
    pd.DataFrame(family).to_csv(args.out/'family_recall.csv',index=False)
    manifest={'file_sha256':digest,'original_run_metadata_path':str(base/'run_metadata.json'),
              'original_metrics_path':str(base/'metrics.csv'),
              'all_splits_unchanged':True,'reference_check':'PASS',
              'exclusion_groups':{'without_template_three':sorted(THREE),
                                  'without_packet_size_count_family':sorted(PACKET_FAMILY)},
              'model':model_params,'study_type':'post-hoc exploratory feature ablation',
              'cautions':['Ablation chosen after examining target labels; not a confirmatory test',
                          'Exact reduced-feature group collisions are audited, not silently ignored',
                          'Calibrated FPR uses the same calibration rows and rule for each arm',
                          'No host/time/connection independence established',
                          'High/low discrimination does not identify label/extractor validity']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'\nDONE. {args.out}/ablation_metrics.csv')
    print('Also review projected_vector_overlap.csv and family_recall.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=Path('research/results/experiment11_2018'))
    p.add_argument('--out',type=Path,default=Path('research/results/experiment11_ablation_2018'))
    p.add_argument('--features-json',type=Path,default=Path('reports/score_diagnostics/metadata.json'))
    p.add_argument('--raw-root',type=Path,default=Path('data/raw'))
    p.add_argument('--path',type=Path,default=None)
    p.add_argument('--chunk-size',type=int,default=50000)
    p.add_argument('--n-jobs',type=int,default=2)
    args=p.parse_args()
    run(args)

if __name__=='__main__':
    main()
