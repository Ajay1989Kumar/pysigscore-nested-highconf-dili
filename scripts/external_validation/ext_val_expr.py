import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression

# training GSVA (TG-GATEs, my rebuild) + labels
tg=pd.read_parquet('../gsva_mine_160x50.parquet')
strict=pd.read_csv('/tmp/pysig-repo/data/human_dili_highconf_labels.csv'); strict['drug_key']=strict['drug_key'].astype(str); strict=strict.set_index('drug_key')
full=pd.read_csv('/tmp/kept446.csv'); full['drug_key']=full['drug_key'].astype(str); full=full.set_index('drug_key')
dm=pd.read_parquet('dm_gsva.parquet')
ext=json.load(open('dm_external_highconf_drugs.json'))

# common hallmark columns
cols=[c for c in tg.columns if c in dm.columns]
train=[d for d in strict.index if d in tg.index]
Xtr=tg.loc[train,cols].values; ytr=strict.loc[train,'y_hard'].values.astype(int)
Xte=dm.loc[ext,cols].values; yte=np.array([int(full.loc[d,'y_hard']) for d in ext])
print(f"train {Xtr.shape} ({ytr.sum()}T/{(ytr==0).sum()}S) | ext-test {Xte.shape} ({yte.sum()}T/{(yte==0).sum()}S) | pathways={len(cols)}")

# ---- Method 1: champion GSVA top-10 z-signature, fit on ALL train, apply to external ----
TOP_K=10
aucs=np.array([roc_auc_score(ytr,Xtr[:,j]) if np.std(Xtr[:,j])>1e-9 else .5 for j in range(Xtr.shape[1])])
sign=np.where(aucs>=.5,1.,-1.); order=np.argsort(-np.abs(aucs-.5))[:TOP_K]
f=Xtr[:,order]*sign[order]; mu,sd=f.mean(0),f.std(0); sd=np.where(sd<1e-9,1,sd)
sig_tr=((f-mu)/sd).mean(1); ms,ss=sig_tr.mean(),sig_tr.std() or 1
def sig(X): return (((X[:,order]*sign[order]-mu)/sd).mean(1)-ms)/ss
sig_te=sig(Xte)
auc_sig=roc_auc_score(yte,sig_te)

# ---- Method 2: L2 logistic on full GSVA, fit train, apply external ----
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
lr=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',C=0.3,max_iter=3000)).fit(Xtr,ytr)
p_lr=lr.predict_proba(Xte)[:,1]; auc_lr=roc_auc_score(yte,p_lr)

def boot(y,s,n=3000):
    rng=np.random.default_rng(13); b=[]
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2: continue
        b.append(roc_auc_score(y[ix],s[ix]))
    return np.quantile(b,[.025,.975])
lo1,hi1=boot(yte,sig_te); lo2,hi2=boot(yte,p_lr)
print(f"\n=== EXTERNAL VALIDATION (expression-only, frozen TG-GATEs model -> 69 DrugMatrix drugs) ===")
print(f"  GSVA top-10 signature : AUROC={auc_sig:.3f}  95%CI[{lo1:.2f},{hi1:.2f}]")
print(f"  L2-logistic (48 GSVA) : AUROC={auc_lr:.3f}  95%CI[{lo2:.2f},{hi2:.2f}]")
# operating point @ sens 0.95 for the signature
tox=sig_te[yte==1]; thr=np.quantile(tox,0.05); pred=(sig_te>=thr).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
print(f"  @Sens0.95 (sig): Spec={tn/max(tn+fp,1):.3f} ({tn}/{tn+fp})  MCC={matthews_corrcoef(yte,pred):+.3f}  TP/FN/FP/TN={tp}/{fn}/{fp}/{tn}")
pd.DataFrame({'drug':ext,'y':yte,'gsva_sig':sig_te.round(3),'lr_prob':p_lr.round(3)}).sort_values('gsva_sig',ascending=False).to_csv('ext_val_expr_scores.csv',index=False)
print("saved ext_val_expr_scores.csv")
