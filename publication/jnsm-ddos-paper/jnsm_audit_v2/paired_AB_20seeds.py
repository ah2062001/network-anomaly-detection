#!/usr/bin/env python3
"""Faithful 20-seed extension of the user's Experiments 7 and 8.

Run from project root after reviewing README:
  python3 paired_AB_20seeds.py --project . --verify-three-only
  python3 paired_AB_20seeds.py --project . --n-jobs 2

Source: CIC-IDS2017 Friday DDoS subset, 37-core-feature group splitting.
A/B: 57 and 37 ordered flow fields from reports/experiment_05/design.json.
Targets: redistributed CIC-DDoS2019 and Feb 21 CSE-CIC-IDS2018.
Reference: reproduces both Exp7 and Exp8 rows for seeds 42,43,44 BEFORE
accepting an extended 20-seed comparison. Original train/test/calibration
partition sizes follow GroupShuffleSplit on the 37-feature source hash.
Models: 100-tree balanced RF with fixed matching seed, median-imputed source
training data. Strict > benign source-calibration 99th percentile threshold.

This changes only the seed set, not the original target-informed exploratory
feature selection (B) or the comparability limitations across collections.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
import sklearn

EXPECTED_HASH={
    '2017':'6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee',
    '2018':'a5f4a1c2689e0aa6566c03a58466de9c407c0be0cbd3cc69306544026611be04',
    '2019':'97af3b16e45f5309a6492cc16d947a74fdc135267f03756b8cea2eb13f4b2379'
}
COLUMNS={
    '2017':'data/raw/ddos.csv',
    '2018':'data/raw/cicids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    '2019':'data/raw/cicddos2019/cicddos2019_dataset.csv'
}
EXPECTED_ROWCOUNTS={'2018':1048575,'2019':431371}
REFERENCE_SEEDS=(42,43,44)
ALL_SEEDS=tuple(range(42,62))

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_aliases(project):
    # Read the exact fixed alias mapping from Experiment 8 without executing it.
    src=(project/'research/experiments/experiment_08_final_validation.py').read_text()
    tree=ast.parse(src)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='ALIASES' for v in node.targets):
            aliases=ast.literal_eval(node.value)
            if not isinstance(aliases,dict):raise RuntimeError('Unexpected alias mapping type')
            return aliases
    raise RuntimeError('Could not find the Experiment 8 ALIASES dictionary')

def split_once(indices, group_ids, fraction, seed):
    splitter=GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed)
    first,second=next(splitter.split(np.zeros(len(indices)),groups=group_ids))
    return indices[first], indices[second]

def target_chunks(filepath, dataset, all_features, aliases, chunksize):
    label_col='Class' if dataset=='2019' else 'Label'
    source_names={feature:(aliases.get(feature,feature) if dataset=='2018' else feature)
                  for feature in all_features}
    if len(set(source_names.values()))!=len(all_features):
        raise RuntimeError('Nonunique mapped target names')
    needed=set(source_names.values())|{label_col}
    target_header=pd.read_csv(filepath,nrows=0).columns.str.strip()
    missing=needed-set(target_header)
    if missing:raise RuntimeError(f'{dataset} mapped header is missing columns: {sorted(missing)}')
    for chunk in pd.read_csv(filepath,
                usecols=lambda c:c.strip() in needed,
                chunksize=chunksize,low_memory=False):
        chunk.columns=chunk.columns.str.strip()
        # Map to the SOURCE-named 57 model input fields, preserving their order.
        rename={mapped:feature for feature,mapped in source_names.items()}
        X=chunk.rename(columns=rename)[all_features]
        for feature in all_features:
            X[feature]=pd.to_numeric(X[feature],errors='coerce')
        X=X.replace([np.inf,-np.inf],np.nan)
        if dataset=='2019':
            labels=chunk['Class'].astype(str).str.strip()
            if not labels.isin(['Attack','Benign']).all():
                raise RuntimeError('Unexpected 2019 class labels')
            y=labels.eq('Attack').to_numpy(dtype=np.uint8)
        else:
            labels=chunk['Label'].astype(str).str.strip()
            if not labels.isin(['Benign','DDOS attack-HOIC','DDOS attack-LOIC-UDP']).all():
                raise RuntimeError('Unexpected 2018 family labels')
            y=labels.ne('Benign').to_numpy(dtype=np.uint8)
        yield X,y

def build_target_cache(project, dataset, features, aliases, size, cache_dir, chunksize):
    cache_dir.mkdir(parents=True,exist_ok=True)
    filename=cache_dir/f'target_{dataset}_57_float64.npy'
    y_filename=cache_dir/f'target_{dataset}_labels_uint8.npy'
    array=np.lib.format.open_memmap(filename,mode='w+',dtype=np.float64,shape=(size,len(features)))
    y_array=np.lib.format.open_memmap(y_filename,mode='w+',dtype=np.uint8,shape=(size,))
    n=0
    for frame,labels in target_chunks(project/COLUMNS[dataset],dataset,features,aliases,chunksize):
        stop=n+len(frame)
        if stop>size:raise RuntimeError(f'Row count unexpectedly exceeds {size}: {dataset}')
        array[n:stop,:]=frame.to_numpy(dtype=np.float64)
        y_array[n:stop]=labels
        n=stop
        if n%250000<len(frame):print(f'  {dataset}: cached {n:,} rows',flush=True)
    array.flush();y_array.flush()
    del array,y_array
    if n!=size:raise RuntimeError(f'Wrong number rows {dataset}: {n} vs {size}')
    y=np.load(y_filename,mmap_mode='r')
    attack=int(np.count_nonzero(y))
    expected_att=333540 if dataset=='2019' else 687742
    if attack!=expected_att:raise RuntimeError(f'Wrong {dataset} attack count: {attack} vs {expected_att}')
    print(f'{dataset} cached {size:,} rows, {attack:,} attacks; cache is local and can be deleted',flush=True)
    return filename,y_filename

def get_source(project,features,core):
    # Exact same read path and column coercion as Experiment 8 for splitting.
    source=pd.read_csv(project/COLUMNS['2017'],low_memory=False)
    source.columns=source.columns.str.strip()
    missing=set(features+['Label'])-set(source.columns)
    if missing:raise RuntimeError(f'Missing source fields: {sorted(missing)}')
    labels=source['Label'].astype(str).str.strip()
    if set(labels)!={'BENIGN','DDoS'}:raise RuntimeError(f'Unexpected 2017 labels: {set(labels)}')
    y=labels.eq('DDoS').to_numpy(dtype=bool)
    groups=pd.util.hash_pandas_object(
        source[core].replace([np.inf,-np.inf],np.nan),index=False
    ).to_numpy(dtype=np.uint64)
    if len(source)!=225745 or y.sum()!=128027:
        raise RuntimeError('Unexpected 2017 source counts')
    print('Source:',len(source),'flows;',np.unique(groups).size,'core-vector groups',flush=True)
    return source,y,groups

def score_cache(model,cache_files,all_features,features,chunksize):
    idx=[all_features.index(f) for f in features]
    Xall=np.load(cache_files[0],mmap_mode='r')
    yall=np.load(cache_files[1],mmap_mode='r')
    scores=np.empty((len(yall),),dtype=np.float64)
    for lo in range(0,len(yall),chunksize):
        hi=min(lo+chunksize,len(yall))
        # A fresh frame provides feature names/order exactly as the source fit.
        frame=pd.DataFrame(Xall[lo:hi][:,idx],columns=features)
        scores[lo:hi]=model.predict_proba(frame)[:,1]
    return yall,scores

def score_stream(model,project,dataset,all_features,features,aliases,chunksize):
    y_chunks=[]; score_chunks=[]
    for X,y in target_chunks(project/COLUMNS[dataset],dataset,all_features,aliases,chunksize):
        y_chunks.append(y)
        score_chunks.append(model.predict_proba(X[features])[:,1])
    return np.concatenate(y_chunks),np.concatenate(score_chunks)

def evaluation(y,scores,threshold):
    y=np.asarray(y,dtype=bool)
    if not (y.any() and (~y).any()):raise RuntimeError('Missing class in target')
    pred=scores>threshold
    tp=int(np.count_nonzero(pred&y));fn=int(np.count_nonzero(~pred&y))
    fp=int(np.count_nonzero(pred&~y));tn=int(np.count_nonzero(~pred&~y))
    return dict(external_auc=float(roc_auc_score(y,scores)),
                external_ap=float(average_precision_score(y,scores)),
                external_recall=tp/(tp+fn),external_fpr=fp/(fp+tn),
                TP=tp,FN=fn,FP=fp,TN=tn)

def compare_to_legacy(result,seed,arm,filename,dataset):
    z=pd.read_csv(filename)
    matches=z[(z.seed==seed)&(z.arm==arm)]
    if len(matches)!=1:raise RuntimeError(f'Legacy {dataset}: expected one seed {seed}/{arm}, found {len(matches)}')
    ref=matches.iloc[0]
    attrs=['threshold','source_auc','external_auc','external_ap','external_recall','external_fpr']
    for key in attrs:
        if not np.isclose(float(result[key]),float(ref[key]),rtol=0,atol=1e-6):
            raise RuntimeError(f'REFERENCE FAIL {dataset} seed={seed} arm={arm} {key}: new={result[key]}, saved={ref[key]}')
    if dataset=='2018':
        for key in ('TP','FP','TN','FN'):
            if int(result[key])!=int(ref[key]):
                raise RuntimeError(f'REFERENCE FAIL 2018 seed={seed} arm={arm} {key}')
    return True

def summarize(out,completed_seeds):
    df=pd.read_csv(out/'per_run.csv')
    if len(df)!=len(completed_seeds)*4:
        raise RuntimeError(f'Expected four rows for each completed seed, got {len(df)}')
    rows=[]
    for seed in completed_seeds:
        for target in ('2019','2018'):
            x=df[(df.seed==seed)&(df.target==int(target))]
            if len(x)!=2:raise RuntimeError('Expected A and B for every target and seed')
            a=x[x.arm=='A_common'].iloc[0]; b=x[x.arm=='B_core'].iloc[0]
            rows.append(dict(seed=seed,target=target,
                delta_auc_B_minus_A=b.external_auc-a.external_auc,
                delta_recall_B_minus_A=b.external_recall-a.external_recall,
                delta_fpr_B_minus_A=b.external_fpr-a.external_fpr,
                A_auc=a.external_auc,B_auc=b.external_auc,
                A_recall=a.external_recall,B_recall=b.external_recall,
                A_fpr=a.external_fpr,B_fpr=b.external_fpr))
    diff=pd.DataFrame(rows)
    diff.to_csv(out/'paired_by_seed.csv',index=False)
    if len(completed_seeds)>=2:
        from scipy.stats import t
        summaries=[]
        for target,frame in diff.groupby('target',sort=True):
            for metric in ('delta_auc_B_minus_A','delta_recall_B_minus_A','delta_fpr_B_minus_A'):
                x=frame[metric].to_numpy(float)
                mean=float(x.mean());sd=float(x.std(ddof=1));n=len(x)
                half=float(t.ppf(0.975,n-1)*sd/np.sqrt(n))
                summaries.append(dict(target=target,metric=metric,n_seeds=n,
                    mean=mean,sd=sd,min=float(x.min()),max=float(x.max()),
                    nominal_seed_level_95pct_lower=mean-half,
                    nominal_seed_level_95pct_upper=mean+half))
        pd.DataFrame(summaries).to_csv(out/'paired_summary.csv',index=False)
    print('Paired CSVs updated for',len(completed_seeds),'completed seeds',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('research/results/paired_AB_20seeds'))
    p.add_argument('--verify-three-only',action='store_true',help='Run only seeds 42, 43, 44, requiring original Exp7/8 reference PASS')
    p.add_argument('--n-jobs',type=int,default=2)
    p.add_argument('--chunk-size',type=int,default=50000)
    p.add_argument('--target-mode',choices=['stream','memmap'],default='memmap',help='memmap requires about 700 MB disk; stream uses less disk and more CSV rereads')
    p.add_argument('--keep-target-cache',action='store_true',help='Retain large .npy caches for debugging (not for upload)')
    args=p.parse_args()
    if args.chunk_size<1000 or args.n_jobs<1:raise RuntimeError('Invalid chunk size or worker count')
    project=args.project.expanduser().resolve()
    out=(project/args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    # Prevent silently combining incompatible previous runs.
    if (out/'per_run.csv').exists():
        raise RuntimeError('Output per_run.csv exists. Use a fresh --out directory to avoid ambiguous duplicate runs.')
    for dataset,p in COLUMNS.items():
        f=project/p
        if not f.is_file():raise FileNotFoundError(f)
        actual=sha256(f)
        if actual!=EXPECTED_HASH[dataset]:raise RuntimeError(f'Checksum mismatch {dataset}: {actual}')
        print(f'{dataset} SHA-256 verified',flush=True)
    design=json.loads((project/'reports/experiment_05/design.json').read_text())
    featsA=list(design['feature_sets']['A_common'])
    featsB=list(design['feature_sets']['B_core'])
    if len(featsA)!=57 or len(featsB)!=37 or not set(featsB).issubset(set(featsA)):
        raise RuntimeError('Unexpected 57/37 feature design')
    meta=json.loads((project/'reports/score_diagnostics/metadata.json').read_text())
    if featsA!=meta['features']:raise RuntimeError('A feature order mismatch with Experiment 1 metadata')
    aliases=load_aliases(project)
    source,y,groups=get_source(project,featsA,featsB)
    cache={};cache_dir=out/'_local_memmap_cache'
    try:
        if args.target_mode=='memmap':
            print('Creating local target memmaps once; do not include these in uploads...',flush=True)
            for ds in ('2019','2018'):
                cache[ds]=build_target_cache(project,ds,featsA,aliases,EXPECTED_ROWCOUNTS[ds],cache_dir,args.chunk_size)
        r7=project/'reports/experiment_07/metrics_by_seed.csv'
        r8=project/'reports/experiment_08/final_metrics.csv'
        results=[];done=[];verified=0
        # Check the three reference seeds before running the full 20-seed batch.
        seeds=REFERENCE_SEEDS if args.verify_three_only else ALL_SEEDS
        for seed in seeds:
            idx=np.arange(len(source))
            dev,holdout=split_once(idx,groups,0.20,seed)
            train,cal=split_once(dev,groups[dev],0.125,seed+1000)
            if set(groups[train].tolist()) & set(groups[cal].tolist()):raise RuntimeError('Group leakage train/cal')
            if set(groups[train].tolist()) & set(groups[holdout].tolist()):raise RuntimeError('Group leakage train/test')
            if set(groups[cal].tolist()) & set(groups[holdout].tolist()):raise RuntimeError('Group leakage cal/test')
            if not (y[train].any() and (~y[train]).any() and y[cal].any() and (~y[cal]).any() and y[holdout].any() and (~y[holdout]).any()):
                raise RuntimeError('Missing class in partition')
            print(f'SEED {seed} | train={len(train):,}, cal={len(cal):,}, holdout={len(holdout):,}',flush=True)
            for arm, features in (('A_common',featsA),('B_core',featsB)):
                # Identical to Exp7/8: median imputation on SOURCE training rows only.
                model=Pipeline([
                    ('imputer',SimpleImputer(strategy='median')),
                    ('rf',RandomForestClassifier(n_estimators=100,random_state=seed,class_weight='balanced',n_jobs=args.n_jobs))
                ])
                Xtr=source.iloc[train][features].replace([np.inf,-np.inf],np.nan)
                Xcal=source.iloc[cal][features].replace([np.inf,-np.inf],np.nan)
                Xho=source.iloc[holdout][features].replace([np.inf,-np.inf],np.nan)
                model.fit(Xtr,y[train])
                cal_scores=model.predict_proba(Xcal)[:,1]
                threshold=float(np.quantile(cal_scores[~y[cal]],0.99))
                src_scores=model.predict_proba(Xho)[:,1]
                src_auc=float(roc_auc_score(y[holdout],src_scores))
                cal_fpr=float(np.mean(cal_scores[~y[cal]]>threshold))
                if cal_fpr>0.01+1e-12:raise RuntimeError('Source calibration FPR budget exceeded')
                print(f'  {arm}: source AUC={src_auc:.6f}; threshold={threshold:.8f}',flush=True)
                for dataset in ('2019','2018'):
                    yext,scores=(score_cache(model,cache[dataset],featsA,features,args.chunk_size)
                                if args.target_mode=='memmap'
                                else score_stream(model,project,dataset,featsA,features,aliases,args.chunk_size))
                    if len(yext)!=EXPECTED_ROWCOUNTS[dataset]:raise RuntimeError(f'Wrong target row count {dataset}')
                    metrics=evaluation(yext,scores,threshold)
                    row=dict(seed=seed,arm=arm,target=int(dataset),train_rows=len(train),cal_rows=len(cal),
                             holdout_rows=len(holdout),threshold=threshold,calibration_fpr=cal_fpr,
                             source_auc=src_auc,**metrics)
                    if seed in REFERENCE_SEEDS:
                        compare_to_legacy(row,seed,arm,r7 if dataset=='2019' else r8,dataset)
                        verified+=1
                        print(f'    REFERENCE PASS {dataset}, seed={seed}, {arm}',flush=True)
                    else:
                        print(f'    {dataset}: AUC={metrics["external_auc"]:.6f}; recall={metrics["external_recall"]:.4f}; FPR={metrics["external_fpr"]:.4f}',flush=True)
                    results.append(row)
                    pd.DataFrame(results).to_csv(out/'per_run.csv',index=False)
                del model
            done.append(seed)
            summarize(out,done)
        if verified!=12:raise RuntimeError(f'Expected 12 original reference checks, obtained {verified}')
        status='20_SEEDS_COMPLETE' if len(done)==20 else 'THREE_SEED_REPRODUCTION_ONLY'
        manifest=dict(status=status,seeds=done,n_reference_checks=verified,
                      data_sha256=EXPECTED_HASH,source_partition_method='GroupShuffleSplit on source 37-feature hash, holdout seed; calibration seed+1000',
                      threshold='strict score > source benign calibration 99th percentile',
                      feature_counts={'A':len(featsA),'B':len(featsB)},
                      target_mode=args.target_mode,python=sys.version,sklearn=sklearn.__version__,pandas=pd.__version__,numpy=np.__version__,platform=platform.platform())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('SUCCESS:',status,'Results in',out,flush=True)
    finally:
        if args.target_mode=='memmap' and not args.keep_target_cache:
            for p in cache_dir.glob('*.npy'):
                try:p.unlink()
                except OSError: pass
            try:cache_dir.rmdir()
            except OSError:pass

if __name__=='__main__':main()
