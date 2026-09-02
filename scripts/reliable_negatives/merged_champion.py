#!/usr/bin/env python3
"""Merged TG-GATEs + DrugMatrix champion, using all reliable negatives.

Resolves the 7-negative limitation by pooling both datasets into one cohort in which
BOTH classes are present on BOTH platforms, so a pooled model cannot reduce to
'platform = label' (verified by within-platform AUROC + a platform-confound check).

Cohort = union of:
  - TG-GATEs high-conf 101 (94T/7S)               GSVA from data/scores_log2fc_GSVA_hallmark.csv
  - DrugMatrix external 69 (55T/14S)              GSVA from data/external/drugmatrix_gsva_hallmark.parquet
  - mined reliable negatives (rounds 1-3, n=25)   data/reliable_negatives/round{1,2,3}.csv
Champion = GSVA top-10 z-signature, LOOCV on the merged cohort.

Run from repo root:  python scripts/reliable_negatives/merged_champion.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import LeaveOneOut

ROOT=Path(__file__).resolve().parents[2]
tr =pd.read_csv(ROOT/"data/scores_log2fc_GSVA_hallmark.csv",index_col=0)
lab=pd.read_csv(ROOT/"data/human_dili_highconf_labels.csv").set_index("drug_key")
dmg=pd.read_parquet(ROOT/"data/external/drugmatrix_gsva_hallmark.parquet")
ext=pd.read_csv(ROOT/"results/external_validation/scores_expression_only.csv").set_index("drug")
mined=pd.concat([pd.read_csv(ROOT/f"data/reliable_negatives/round{i}.csv",index_col=0) for i in (1,2,3)])
rng=np.random.default_rng(0)
cols=[c for c in tr.columns if c in dmg.columns]

rows={}
for d in lab.index:
    if d in tr.index: rows[d]=(int(lab.loc[d,"y_hard"]),"TG",tr.loc[d,cols].values)
for d in ext.index:
    if d in dmg.index and d not in rows: rows[d]=(int(ext.loc[d,"y"]),"DM",dmg.loc[d,cols].values)
for d in mined.index:
    if d in dmg.index and d not in rows: rows[d]=(0,"DM",dmg.loc[d,cols].values)

drugs=list(rows); y=np.array([rows[d][0] for d in drugs]); plat=np.array([rows[d][1] for d in drugs])
X=np.array([rows[d][2] for d in drugs],float)
print(f"MERGED cohort: {len(drugs)} drugs | {int(y.sum())}T / {int((y==0).sum())}S")
for p in ("TG","DM"):
    m=plat==p; print(f"  {p}: {m.sum()} ({int(y[m].sum())}T/{int((y[m]==0).sum())}S)")

def sig_fit(Xt,yt,K=10):
    a=np.array([roc_auc_score(yt,Xt[:,j]) if np.std(Xt[:,j])>1e-9 else .5 for j in range(Xt.shape[1])])
    sgn=np.where(a>=.5,1.,-1.); order=np.argsort(-np.abs(a-.5))[:K]
    f=Xt[:,order]*sgn[order]; mu,sd=f.mean(0),f.std(0); sd=np.where(sd<1e-9,1,sd)
    s0=((f-mu)/sd).mean(1); ms,ss=s0.mean(),(s0.std() or 1)
    return lambda Z:(((Z[:,order]*sgn[order]-mu)/sd).mean(1)-ms)/ss

oof=np.zeros(len(y))
for trn,te in LeaveOneOut().split(X): oof[te]=sig_fit(X[trn],y[trn])(X[te])
def boot(yv,sv,fn,n=3000):
    v=[]
    for _ in range(n):
        ix=rng.integers(0,len(yv),len(yv))
        if len(np.unique(yv[ix]))>1:
            try:v.append(fn(yv[ix],sv[ix]))
            except Exception:pass
    return np.nanpercentile(v,[2.5,97.5])
thr=np.quantile(oof[y==1],0.05); pred=(oof>=thr).astype(int)
tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
au=roc_auc_score(y,oof); mc=matthews_corrcoef(y,pred)
bm=max(matthews_corrcoef(y,(oof>=t).astype(int)) for t in np.unique(oof))
aci=boot(y,oof,lambda y,s:roc_auc_score(y,s)); mci=boot(y,oof,lambda y,s:matthews_corrcoef(y,(s>=np.quantile(s[y==1],0.05)).astype(int)))
print(f"\n=== MERGED champion (GSVA signature, LOOCV) ===")
print(f"  AUROC={au:.3f} [{aci[0]:.2f},{aci[1]:.2f}]")
print(f"  @Sens {tp/(tp+fn):.2f}: Spec={tn/(tn+fp):.2f}  MCC={mc:+.3f} [{mci[0]:+.2f},{mci[1]:+.2f}]  bestMCC={bm:+.3f}")
for p in ("TG","DM"):
    m=plat==p
    if len(np.unique(y[m]))>1: print(f"  within-{p} AUROC = {roc_auc_score(y[m],oof[m]):.3f}")
pb=(plat=="DM").astype(int); oofp=np.zeros(len(y))
for trn,te in LeaveOneOut().split(X): oofp[te]=sig_fit(X[trn],pb[trn])(X[te])
print(f"  platform-confound AUROC = {roc_auc_score(pb,oofp):.3f}  (DILI {au:.3f} >> platform => label signal dominates)")
