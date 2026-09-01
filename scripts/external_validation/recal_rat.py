import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# external cohort scores already have frozen GSVA sig (transfers); add DrugMatrix-native rat features
sc=pd.read_csv('ext_val_gbm_scores.csv')            # drug, y, gsva_sig(frozen), rat_prob(frozen,fails), gbm
meas=pd.read_parquet('dm_measured_endpoints.parquet')  # ALT,AST,fatty,necro,anyliv per drug
test=[d for d in sc.drug if d in meas.index]
sc=sc.set_index('drug').loc[test]
y=sc.y.values.astype(int); gsig=sc.gsva_sig.values
g=meas.loc[test]
# DrugMatrix-native rat feature matrix (raw, recalibrated by the learner itself)
Rn=np.column_stack([g['ALT'].fillna(g['ALT'].median()), g['AST'].fillna(g['AST'].median()),
                    g['fatty'].fillna(0), g['necro'].fillna(0), g['anyliv'].fillna(0)])
print(f"external cohort n={len(y)} ({y.sum()}T/{(y==0).sum()}S)")

def loo_prob(mk, X):
    oof=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        if len(np.unique(y[tr]))<2: oof[te]=y[tr].mean(); continue
        oof[te]=mk().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return oof
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',C=0.5,max_iter=2000))
def boot(s,n=3000):
    rng=np.random.default_rng(13); b=[]
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))>1: b.append(roc_auc_score(y[ix],s[ix]))
    return np.quantile(b,[.025,.975])
def bmcc(s): return max(matthews_corrcoef(y,(s>=t).astype(int)) for t in np.unique(s))
def show(nm2,s):
    lo,hi=boot(s); print(f"  {nm2:34s} AUROC={roc_auc_score(y,s):.3f} [{lo:.2f},{hi:.2f}]  bestMCC={bmcc(s):+.3f}")

print("\n=== RECALIBRATED on DrugMatrix (LOOCV within external cohort) ===")
show("expression-only (frozen GSVA sig)", gsig)                 # frozen, transfers
ratP=loo_prob(LR, Rn)
show("rat-arm RECALIBRATED (LR, DM-native)", ratP)
# combine frozen GSVA sig + recalibrated rat prob
comb=loo_prob(LR, np.column_stack([gsig, ratP]))
show("GSVA(frozen)+rat(recal) meta-LR", comb)
# also GBM combine + simple average
combz=(gsig-gsig.mean())/gsig.std() + (ratP-ratP.mean())/ratP.std()
show("GSVA(frozen)+rat(recal) z-sum", combz)
print("\n[reference] frozen-GBM (non-recal) external was AUROC 0.499")
