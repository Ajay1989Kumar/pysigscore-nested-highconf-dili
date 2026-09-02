#!/usr/bin/env python3
"""Batch-correction sensitivity: merged champion, ComBat-corrected vs uncorrected.

Batch = platform (TG-GATEs vs DrugMatrix). ComBat (inmoose.pycombat_norm) is run
UNSUPERVISED (no DILI-label covariate) so the correction cannot leak the outcome.
Requires `inmoose`. Run from repo root:
  python scripts/reliable_negatives/combat_sensitivity.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from inmoose.pycombat import pycombat_norm
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
print(f"MERGED: {len(drugs)} drugs ({int(y.sum())}T/{int((y==0).sum())}S) | TG={int((plat=='TG').sum())} DM={int((plat=='DM').sum())}")
Xc=pycombat_norm(pd.DataFrame(X.T,columns=range(len(drugs))), batch=np.where(plat=="TG",0,1)).values.T

def sig_fit(Xt,yt,K=10):
    a=np.array([roc_auc_score(yt,Xt[:,j]) if np.std(Xt[:,j])>1e-9 else .5 for j in range(Xt.shape[1])])
    sgn=np.where(a>=.5,1.,-1.); order=np.argsort(-np.abs(a-.5))[:K]
    f=Xt[:,order]*sgn[order]; mu,sd=f.mean(0),f.std(0); sd=np.where(sd<1e-9,1,sd)
    s0=((f-mu)/sd).mean(1); ms,ss=s0.mean(),(s0.std() or 1)
    return lambda Z:(((Z[:,order]*sgn[order]-mu)/sd).mean(1)-ms)/ss
def loocv(M,yv):
    oof=np.zeros(len(yv))
    for trn,te in LeaveOneOut().split(M): oof[te]=sig_fit(M[trn],yv[trn])(M[te])
    return oof
mcc_at=lambda y,s:matthews_corrcoef(y,(s>=np.quantile(s[y==1],0.05)).astype(int))
def boot(yv,sv,fn,n=3000):
    v=[]
    for _ in range(n):
        ix=rng.integers(0,len(yv),len(yv))
        if len(np.unique(yv[ix]))>1:
            try:v.append(fn(yv[ix],sv[ix]))
            except Exception:pass
    return np.nanpercentile(v,[2.5,97.5])
def report(tag,M):
    oof=loocv(M,y); au=roc_auc_score(y,oof)
    thr=np.quantile(oof[y==1],0.05); p=(oof>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    mc=matthews_corrcoef(y,p); bm=max(matthews_corrcoef(y,(oof>=t).astype(int)) for t in np.unique(oof))
    aci=boot(y,oof,lambda y,s:roc_auc_score(y,s)); mci=boot(y,oof,mcc_at)
    pb=(plat=="DM").astype(int); pauc=roc_auc_score(pb,loocv(M,pb))
    print(f"{tag}\n   AUROC={au:.3f}[{aci[0]:.2f},{aci[1]:.2f}] | MCC@Sn.95={mc:+.3f}[{mci[0]:+.2f},{mci[1]:+.2f}] "
          f"Spec={tn/(tn+fp):.2f} | bestMCC={bm:+.3f}\n   within-TG={roc_auc_score(y[plat=='TG'],oof[plat=='TG']):.3f}  "
          f"within-DM={roc_auc_score(y[plat=='DM'],oof[plat=='DM']):.3f}  | platform-confound AUROC={pauc:.3f}")
report("=== UNCORRECTED (GSVA-native) ===",X); print()
report("=== ComBat-CORRECTED (unsupervised) ===",Xc)
