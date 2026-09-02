#!/usr/bin/env python3
# PROVENANCE ONLY — round-1 reliable-negative mining.
# Requires external inputs (hardcoded paths): the 402-drug rat-expression cache
# (rat_expr_to_human_dili_402) and the 7-source fusion table (dili-diffcoex-446).
# The mined OUTPUTS are committed under data/reliable_negatives/round{1,2,3}.csv,
# which is what merged_champion.py consumes; this file documents how they were made.

"""Expand the 7-negative class by mining RELIABLE new negatives via dual agreement.

Anchors  : 101 high-conf drugs (94 toxic / 7 safe) — trusted, low-noise labels.
Feature vote : out-of-fold DILI probabilities from the validated rat-expression
               hybrid model (loocv_hybrid_latefusion; expr AUROC 0.728, fused 0.721),
               i.e. each drug scored by a model trained on the OTHER drugs.
Label vote   : independent-Bayes fusion probability p_combined (7-source).
Rule     : a candidate (non-anchor) is a RELIABLE NEW NEGATIVE iff BOTH agree safe —
             (i) label leans safe:  p_combined <= P_SAFE, and
            (ii) feature model safe: p_expr <= band set by the 7 safe anchors.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R402="/Users/ajaykumar/rat_expr_to_human_dili_402"
LAB ="/Users/ajaykumar/dili-diffcoex-446/data/labels_kept.csv"
HC  ="/Users/ajaykumar/pysigscore-nested-highconf-dili/data/human_dili_highconf_labels.csv"

pred=pd.read_csv(f"{R402}/results/loocv_hybrid_latefusion_predictions.seed13.csv").set_index("drug_key")
lab =pd.read_csv(LAB).set_index("drug_key")[["p_combined","y_hard"]]
hc  =pd.read_csv(HC).set_index("drug_key"); hc_ix=set(hc.index)

drugs=[d for d in pred.index if d in lab.index]
df=pd.DataFrame(index=drugs)
df["p_expr"]=pred.loc[drugs,"p_expr"]          # feature vote (expression model, OOF)
df["p_mean"]=pred.loc[drugs,"p_mean"]          # feature vote (expr+chem fused, OOF)
df["p_combined"]=lab.loc[drugs,"p_combined"]   # label vote
df["y_fusion"]=lab.loc[drugs,"y_hard"].astype(int)
df["anchor"]=[d in hc_ix for d in drugs]
df["y_anchor"]=[int(hc.loc[d,"y_hard"]) if d in hc_ix else -1 for d in drugs]

A=df[df.anchor]
print(f"cohort with feature scores: {len(df)} drugs | anchors: {len(A)} ({int((A.y_anchor==1).sum())}T/{int((A.y_anchor==0).sum())}S)")
print(f"feature model on anchors: AUROC(p_expr)={roc_auc_score(A.y_anchor,A.p_expr):.3f}  "
      f"AUROC(p_mean)={roc_auc_score(A.y_anchor,A.p_mean):.3f}")

# FIXED meaningful thresholds: both votes must genuinely lean safe.
# (anchor-derived band is unusable: the expr model was trained on imbalanced full-402
#  labels, so even safe anchors score p_expr~0.99; we use absolute "leans-safe" cutoffs.)
P_EXPR=0.50    # feature vote: expression model leans safe
safeA=A[A.y_anchor==0]
print(f"safe-anchor p_expr: median={safeA.p_expr.median():.3f} (imbalance-inflated) -> using fixed P_EXPR={P_EXPR}")

cand=df[~df.anchor].copy()
cand=cand[cand.y_fusion==0]                       # require fusion to call it safe (not toxic)
fusion_neg=cand
# confidence tiers by dual agreement
tierA=cand[(cand.p_combined<=0.20)&(cand.p_expr<=P_EXPR)].sort_values("p_expr")   # both strong
tierB=cand[(cand.p_combined<=0.40)&(cand.p_expr<=P_EXPR)].sort_values("p_expr")   # both moderate
new_neg=tierB                                    # headline set = dual-agreement, fusion-negative

print(f"\n=== RELIABLE NEW NEGATIVES (dual agreement: label safe AND feature p_expr<= {P_EXPR}) ===")
print(f"candidate pool (non-anchor, fusion-negative): {len(fusion_neg)}")
print(f"features corroborate (p_expr<= {P_EXPR}) on {int((fusion_neg.p_expr<=P_EXPR).sum())}/{len(fusion_neg)}")
print(f"  Tier A (p_combined<=0.20 AND p_expr<=0.50): {len(tierA)}")
print(f"  Tier B (p_combined<=0.40 AND p_expr<=0.50): {len(tierB)}  <- headline")
n0=int((A.y_anchor==0).sum()); n1=int((A.y_anchor==1).sum())
print(f"\nNEGATIVE CLASS: {n0} high-conf -> {n0+len(tierA)} (+TierA) / {n0+len(tierB)} (+TierB)")
print(f"new working balance: {n1} toxic anchors / {n0+len(tierB)} safe  (was {n1}/{n0}, imbalance {n1/n0:.0f}:1 -> {n1/(n0+len(tierB)):.1f}:1)")

print("\nreliable new negatives (Tier B, both votes safe):")
print(new_neg[["p_combined","p_expr","p_mean"]].round(3).to_string())

new_neg.to_csv("/private/tmp/claude-501/-Users-ajaykumar/a80ad061-cbf3-47ee-a942-4eb86fd06175/scratchpad/reliable_new_negatives.csv")
df.to_csv("/private/tmp/claude-501/-Users-ajaykumar/a80ad061-cbf3-47ee-a942-4eb86fd06175/scratchpad/all_scored_dualvote.csv")
print("\nsaved reliable_new_negatives.csv, all_scored_dualvote.csv")
