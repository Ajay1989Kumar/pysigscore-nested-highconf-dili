import openpyxl, json, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
ann={l.split('\t')[0]:l.rstrip('\n').split('\t')[1].strip('"') for l in open('CHM_annotation.txt') if '\t' in l}
def norm(s): return re.sub(r"[\s\-_,/().\[\]'\"`]+","",str(s).lower())

# ---- extract measured endpoint rows from xlsx (first 185 rows) ----
wb=openpyxl.load_workbook('pcp.xlsx',read_only=True); ws=wb['probe_clinical_pathology']
rows={}; tnames=None
for i,row in enumerate(ws.iter_rows(min_row=1,max_row=185,values_only=True)):
    if i==0: tnames=list(row)[1:]; continue
    fid=row[0]
    if fid and (str(fid).startswith('C0000') or str(fid).startswith('M0000')): rows[str(fid)]=list(row)[1:]
wb.close()
def S(fid): return pd.Series(rows[fid],index=tnames,dtype='float64') if fid in rows else pd.Series(np.nan,index=tnames)
def gmax(ids):
    p=[i for i in ids if i in rows]
    return pd.concat([S(i) for i in p],axis=1).max(1) if p else pd.Series(np.nan,index=tnames)
grp=lambda kw,extra=(): [k for k,v in ann.items() if k[0]=='M' and 'HEPATOCYTE' in v and kw in v]+list(extra)
Lfatty=grp('LIPID ACCUMULATION'); Lnecro=grp('NECROSIS',('M000138',)); Lhyper=grp('HYPERTROPHY')
Linfil=[k for k,v in ann.items() if k[0]=='M' and ('INFILTRAT' in v or 'INFLAMMATION' in v)]
Lall=[k for k,v in ann.items() if k[0]=='M' and ('HEPATOCYTE' in v or 'BILE DUCT' in v or 'KUPFFER' in v)]
meas=pd.DataFrame({'ALT':S('C000001'),'AST':S('C000004'),'fatty':gmax(Lfatty),'necro':gmax(Lnecro),
                   'hyper':gmax(Lhyper),'infil':gmax(Linfil),'anyliv':gmax(Lall)})
meas['drug']=[norm(re.sub(r'-[.\d]+d-[.\d]+.*$','',str(t))) for t in meas.index]
g=meas.groupby('drug').max(numeric_only=True)
# build 14-endpoint panel (TG-GATEs schema)
AB,SB=45.0,90.0; SEV=1.0
def P14(gg):
    P=pd.DataFrame(index=gg.index)
    f=lambda c,t:(gg[c]>=t).fillna(False).astype(int)
    P['path_any_high']=f('anyliv',SEV); P['path_any_treated']=(gg['anyliv']>0).fillna(False).astype(int)
    P['path_necrosis_high']=f('necro',SEV); P['path_hypertrophy_high']=f('hyper',SEV)
    P['path_vacuol_fatty_high']=f('fatty',SEV); P['path_infiltration_high']=f('infil',SEV)
    P['path_mod_severe_high']=f('anyliv',2.0); P['path_necrosis_modsev_high']=f('necro',2.0)
    P['chem_ALT_2x_high']=f('ALT',2*AB); P['chem_ALT_3x_high']=f('ALT',3*AB); P['chem_AST_2x_high']=f('AST',2*SB)
    P['chem_LWrel_1.2x_high']=0; P['chem_LWrel_1.5x_high']=0
    P['composite_necrosis_or_ALT2x']=((gg['necro']>=SEV)|(gg['ALT']>=2*AB)).fillna(False).astype(int)
    return P
dm_panel=P14(g)
print("DM measured 14-panel:",dm_panel.shape,"| OR+ rate:",round(((dm_panel[['path_vacuol_fatty_high','path_mod_severe_high','chem_ALT_2x_high']].max(1))>0).mean(),2))

# ---- TG-GATEs training: GSVA + rat panel (measured, from repo) ----
tg=pd.read_parquet('../gsva_mine_160x50.parquet')
ep=pd.read_csv('/tmp/rat_endpoints_full.csv'); ep['drug_key']=ep['drug_key'].astype(str); ep=ep.set_index('drug_key')
strict=pd.read_csv('/tmp/pysig-repo/data/human_dili_highconf_labels.csv'); strict['drug_key']=strict['drug_key'].astype(str); strict=strict.set_index('drug_key')
full=pd.read_csv('/tmp/kept446.csv'); full['drug_key']=full['drug_key'].astype(str); full=full.set_index('drug_key')
dm_gsva=pd.read_parquet('dm_gsva.parquet'); ext=json.load(open('dm_external_highconf_drugs.json'))
TGEND=['path_any_high','path_any_treated','path_necrosis_high','path_hypertrophy_high','path_vacuol_fatty_high','path_infiltration_high','path_mod_severe_high','path_necrosis_modsev_high','chem_ALT_2x_high','chem_ALT_3x_high','chem_AST_2x_high','chem_LWrel_1.2x_high','chem_LWrel_1.5x_high','composite_necrosis_or_ALT2x']
train=[d for d in strict.index if d in tg.index and d in ep.index]
cols=[c for c in tg.columns if c in dm_gsva.columns]
Xg_tr=tg.loc[train,cols].values; ytr=strict.loc[train,'y_hard'].values.astype(int)
Rp_tr=ep.loc[train,TGEND].fillna(0).astype(int).values
# external test cohort: has GSVA + measured endpoint panel
test=[d for d in ext if d in dm_gsva.index and d in dm_panel.index]
Xg_te=dm_gsva.loc[test,cols].values; Rp_te=dm_panel.loc[test,TGEND].values
yte=np.array([int(full.loc[d,'y_hard']) for d in test])
print(f"train {len(train)} ({ytr.sum()}T/{(ytr==0).sum()}S) | external test {len(test)} ({yte.sum()}T/{(yte==0).sum()}S)")

# frozen pieces trained on TG-GATEs
# 1) GSVA top-10 signature
TOP_K=10
aucs=np.array([roc_auc_score(ytr,Xg_tr[:,j]) if np.std(Xg_tr[:,j])>1e-9 else .5 for j in range(Xg_tr.shape[1])])
sgn=np.where(aucs>=.5,1.,-1.); order=np.argsort(-np.abs(aucs-.5))[:TOP_K]
f=Xg_tr[:,order]*sgn[order]; mu,sd=f.mean(0),f.std(0); sd=np.where(sd<1e-9,1,sd)
strn=((f-mu)/sd).mean(1); ms,ss=strn.mean(),strn.std() or 1
gsig=lambda X:(((X[:,order]*sgn[order]-mu)/sd).mean(1)-ms)/ss
sig_tr,sig_te=gsig(Xg_tr),gsig(Xg_te)
# 2) rat-panel HGB -> prob (LOOCV oof on train; fit-all applied to test)
def hgb(): return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,random_state=13,l2_regularization=1.0)
oofR=np.zeros(len(ytr))
for tr,te in LeaveOneOut().split(Rp_tr): oofR[te]=hgb().fit(Rp_tr[tr],ytr[tr]).predict_proba(Rp_tr[te])[:,1]
ratP_tr=oofR; ratP_te=hgb().fit(Rp_tr,ytr).predict_proba(Rp_te)[:,1]
# 3) top GBM on [gsva_sig, rat_prob]
def hgb2(): return HistGradientBoostingClassifier(max_depth=2,learning_rate=0.1,max_iter=200,random_state=13,l2_regularization=2.0)
top=hgb2().fit(np.column_stack([sig_tr,ratP_tr]),ytr)
score=top.predict_proba(np.column_stack([sig_te,ratP_te]))[:,1]

def boot(y,s,n=3000):
    rng=np.random.default_rng(13); b=[]
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))>1: b.append(roc_auc_score(y[ix],s[ix]))
    return np.quantile(b,[.025,.975])
def best_mcc(y,s):
    return max(matthews_corrcoef(y,(s>=t).astype(int)) for t in np.unique(s))
print("\n=== FULL GBM EXTERNAL VALIDATION (frozen TG-GATEs -> DrugMatrix) ===")
for nm2,s in [('expression-only (GSVA sig)',sig_te),('rat-panel-only (prob)',ratP_te),('FULL GBM (GSVA+rat)',score)]:
    au=roc_auc_score(yte,s); lo,hi=boot(yte,s); bm=best_mcc(yte,s)
    print(f"  {nm2:28s} AUROC={au:.3f} [{lo:.2f},{hi:.2f}]  bestMCC={bm:+.3f}")
pd.DataFrame({'drug':test,'y':yte,'gsva_sig':sig_te.round(3),'rat_prob':ratP_te.round(3),'gbm':score.round(3)}).sort_values('gbm',ascending=False).to_csv('ext_val_gbm_scores.csv',index=False)
print("saved ext_val_gbm_scores.csv")
