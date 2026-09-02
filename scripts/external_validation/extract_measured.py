import openpyxl, json, re, numpy as np, pandas as pd
ann={l.split('\t')[0]:l.rstrip('\n').split('\t')[1].strip('"') for l in open('CHM_annotation.txt') if '\t' in l}
def norm(s): return re.sub(r"[\s\-_,/().\[\]'\"`]+","",str(s).lower())
# target endpoint rows (0-indexed) from scan
want={4:'C000001',7:'C000004'}  # ALT, AST
for r in [72,73,74,75,79,80,81,82,87,88,91,92,96,97,98,99,102,179]:
    pass
# read first 185 rows only (endpoint block)
wb=openpyxl.load_workbook('pcp.xlsx', read_only=True)
ws=wb['probe_clinical_pathology']
rows={}
treatments=None
for i,row in enumerate(ws.iter_rows(min_row=1,max_row=185,values_only=True)):
    if i==0: treatments=list(row); continue
    fid=row[0]
    if fid and (str(fid).startswith('C0000') or str(fid).startswith('M0000')):
        rows[str(fid)]=list(row)
wb.close()
print("extracted endpoint rows:",len(rows),"| treatments(cols):",len(treatments)-1)

# treatment columns start at index 1
tnames=treatments[1:]
def val(fid):
    r=rows.get(fid)
    return None if r is None else r[1:]
# clin chem: ALT=C000001, AST=C000004 ; liver histopath groups
Lfatty=[k for k,v in ann.items() if k[0]=='M' and 'HEPATOCYTE' in v and 'LIPID ACCUMULATION' in v]
Lnecro=[k for k,v in ann.items() if k[0]=='M' and 'HEPATOCYTE' in v and 'NECROSIS' in v]+['M000138']
Lall=[k for k,v in ann.items() if k[0]=='M' and ('HEPATOCYTE' in v or 'BILE DUCT' in v or 'KUPFFER' in v)]
def series(fid):
    v=val(fid)
    return pd.Series(v,index=tnames,dtype='float64') if v is not None else pd.Series(np.nan,index=tnames)
def group_max(ids):
    present=[i for i in ids if i in rows]
    if not present: return pd.Series(np.nan,index=tnames)
    return pd.concat([series(i) for i in present],axis=1).max(axis=1)
alt=series('C000001'); ast=series('C000004')
fatty=group_max(Lfatty); necro=group_max(Lnecro); anyliv=group_max(Lall)
print(f"MEASURED coverage (non-null treatments): ALT={alt.notna().sum()} AST={ast.notna().sum()} fatty={fatty.notna().sum()} necro={necro.notna().sum()} anyliv={anyliv.notna().sum()}")
print(f"ALT measured: median={alt.median():.1f} max={alt.max():.1f}")
print(f"necro severity measured: max={necro.max():.2f} ; fatty max={fatty.max():.2f}")
# per-drug aggregate (max across treatments)
df=pd.DataFrame({'ALT':alt,'AST':ast,'fatty':fatty,'necro':necro,'anyliv':anyliv})
df['drug']=[norm(re.sub(r'-[.\d]+d-[.\d]+.*$','',str(t))) for t in df.index]
g=df.groupby('drug').max(numeric_only=True)
g.to_parquet('dm_measured_endpoints.parquet')
print("\nper-drug measured endpoints:",g.shape)
ext=json.load(open('dm_external_highconf_drugs.json'))
covA=sum(1 for d in ext if d in g.index and not np.isnan(g.loc[d,'ALT']))
covH=sum(1 for d in ext if d in g.index and not np.isnan(g.loc[d,'anyliv']))
print(f"external cohort {len(ext)}: measured ALT for {covA}, measured histopath for {covH}")
