#!/usr/bin/env python3
"""Reproduce docs/GSVA_TRANSFER_TGGATES_TO_DRUGMATRIX.md from committed data.

GSVA top-10 pathway signature:
  - TG-GATEs internal LOOCV        (n=101)
  - DrugMatrix within-dataset LOOCV (n=57 measured-panel, matched protocol)
  - TG-GATEs -> DrugMatrix frozen transfer (n=57 and n=69)
Run from the repo root:  python scripts/reproduce_gsva_transfer.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
tr  = pd.read_csv(ROOT/"data/scores_log2fc_GSVA_hallmark.csv", index_col=0)
lab = pd.read_csv(ROOT/"data/human_dili_highconf_labels.csv").set_index("drug_key")
dm  = pd.read_parquet(ROOT/"data/external/drugmatrix_gsva_hallmark.parquet")
me  = pd.read_parquet(ROOT/"data/external/drugmatrix_measured_liver_endpoints.parquet")
ext = pd.read_csv(ROOT/"results/external_validation/scores_expression_only.csv").set_index("drug")

cols = [c for c in tr.columns if c in dm.columns]          # 48 shared Hallmarks
tg   = [d for d in lab.index if d in tr.index]
Xtg  = tr.loc[tg, cols].values; ytg = lab.loc[tg, "y_hard"].astype(int).values

def sig_fit(Xtr, ytr, K=10):
    a = np.array([roc_auc_score(ytr, Xtr[:, j]) if np.std(Xtr[:, j]) > 1e-9 else .5
                  for j in range(Xtr.shape[1])])
    sgn = np.where(a >= .5, 1., -1.); order = np.argsort(-np.abs(a - .5))[:K]
    f = Xtr[:, order] * sgn[order]; mu, sd = f.mean(0), f.std(0); sd = np.where(sd < 1e-9, 1, sd)
    s0 = ((f - mu) / sd).mean(1); ms, ss = s0.mean(), (s0.std() or 1)
    return lambda X: ((((X[:, order] * sgn[order]) - mu) / sd).mean(1) - ms) / ss

def best_mcc(y, s): return max(matthews_corrcoef(y, (s >= t).astype(int)) for t in np.unique(s))
def perm_p(y, s, obs, n=10000):
    rng = np.random.default_rng(7)
    return (sum(roc_auc_score(rng.permutation(y), s) >= obs for _ in range(n)) + 1) / (n + 1)

def loocv_sig(X, y):
    oof = np.zeros(len(y))
    for tri, tei in LeaveOneOut().split(X): oof[tei] = sig_fit(X[tri], y[tri])(X[tei])
    return oof

def line(tag, y, s):
    au = roc_auc_score(y, s)
    print(f"  {tag:42s} AUROC={au:.3f}  perm-p={perm_p(y,s,au):.3f}  bestMCC={best_mcc(y,s):+.3f}  "
          f"({int(y.sum())}T/{int((y==0).sum())}S)")

meas = [d for d in ext.index if d in me.index and d in dm.index]
full = [d for d in ext.index if d in dm.index]
ym = ext.loc[meas, "y"].astype(int).values; Xm = dm.loc[meas, cols].values
yf = ext.loc[full, "y"].astype(int).values; Xf = dm.loc[full, cols].values

print("GSVA top-10 signature — TG-GATEs vs DrugMatrix\n")
line("TG-GATEs internal LOOCV (n=101)", ytg, loocv_sig(Xtg, ytg))
line("DrugMatrix within-dataset LOOCV (n=57)", ym, loocv_sig(Xm, ym))
sigF = sig_fit(Xtg, ytg)
line("TG->DrugMatrix frozen transfer (n=57)", ym, sigF(Xm))
line("TG->DrugMatrix frozen transfer (n=69)", yf, sigF(Xf))
