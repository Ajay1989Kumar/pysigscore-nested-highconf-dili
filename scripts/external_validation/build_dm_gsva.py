import pandas as pd, numpy as np, json, re, warnings
warnings.filterwarnings('ignore')
meta=json.load(open('gse57815_meta.json'))
def norm(s): return re.sub(r"[\s\-_,/().\[\]'\"`]+","",str(s).lower())

# --- load RMA (genes x samples) ---
print("loading affy_dm_affy.tsv ...", flush=True)
rma=pd.read_csv('affy_dm_affy.tsv', sep='\t', index_col=0)
print("RMA:", rma.shape, flush=True)
# map column -> GSM (col like GSM1392188_92992)
col2gsm={c: c.split('_')[0] for c in rma.columns}
# build treated/control sample lists per compound
from collections import defaultdict
treated=defaultdict(list); controls_by_vehtime=defaultdict(list); dose_of={}
for c in rma.columns:
    g=col2gsm[c]; d=meta.get(g)
    if not d: continue
    dose=d.get('dose',''); veh=d.get('vehicle',''); tm=d.get('time','')
    if dose.startswith('0 ') or dose.startswith('0m') or dose=='0 mg/kg':
        controls_by_vehtime[(veh,tm)].append(c)
    else:
        comp=d.get('compound','')
        if comp:
            try: dval=float(dose.split()[0])
            except: dval=np.nan
            treated[comp].append((c,dval,veh,tm))
print("compounds treated:",len(treated),"control groups:",len(controls_by_vehtime), flush=True)

# per-compound High-dose log2FC = mean(High treated) - mean(matched vehicle controls)
rows={}
for comp,samps in treated.items():
    dvals=[s[1] for s in samps if not np.isnan(s[1])]
    if not dvals: continue
    hi=max(dvals)
    hi_samps=[s for s in samps if s[1]==hi]
    tcols=[s[0] for s in hi_samps]
    # matched controls: same vehicle(s) & time(s) as the high-dose samples
    ccols=[]
    for _,_,veh,tm in hi_samps:
        ccols+=controls_by_vehtime.get((veh,tm),[])
    if not ccols:  # fallback: any control of same vehicle
        vehs=set(s[2] for s in hi_samps)
        for (veh,tm),cs in controls_by_vehtime.items():
            if veh in vehs: ccols+=cs
    if not ccols: continue
    lfc = rma[tcols].mean(1) - rma[list(set(ccols))].mean(1)
    rows[norm(comp)]=lfc.values
log2fc=pd.DataFrame(rows).T; log2fc.columns=rma.index
print("DrugMatrix log2FC:", log2fc.shape, "(drugs x rat-entrez)", flush=True)

# ortholog collapse -> human symbols (reuse map)
orth=json.load(open('../ortholog_entrez2human.json'))
gene_ids=[str(g) for g in log2fc.columns]
keep=[i for i,g in enumerate(gene_ids) if g in orth]
sub=log2fc.iloc[:,keep].copy(); sub.columns=[orth[gene_ids[i]] for i in keep]
human=sub.T.groupby(level=0).mean().T
print("human-symbol log2FC:", human.shape, flush=True)

# gseapy GSVA (same params as TG-GATEs build)
import gseapy
hm=json.load(open('../hallmark50.json'))
expr=human.T; expr=expr[~expr.index.duplicated()]
res=gseapy.gsva(data=expr, gene_sets=hm, min_size=5, max_size=1000, kcdf='Gaussian', seed=13, threads=4, outdir=None)
mat=res.res2d.pivot(index='Name',columns='Term',values='ES')
# reconcile hallmark names to frozen columns
fz=pd.read_csv('/tmp/pysig-repo/data/scores_log2fc_GSVA_hallmark.csv',index_col=0)
def nm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
fzmap={nm(c.replace('HALLMARK_','')):c for c in fz.columns}
colmap={c:fzmap[nm(c)] for c in mat.columns if nm(c) in fzmap}
mat=mat.rename(columns=colmap)[[colmap[c] for c in mat.columns if c in colmap]]
mat.to_parquet('dm_gsva.parquet')
print("SAVED dm_gsva.parquet:", mat.shape, "drugs x hallmarks", flush=True)
print("sample drugs:", list(mat.index)[:8])
