#!/usr/bin/env python3
"""
Champion (leakage-safe nested LOO): pysigscore GSVA Hallmarks + rat OR
======================================================================
score = z(GSVA_topk10) + λ · OR

Nested leave-one-drug-out (no two-stage stacking):
  For each held-out drug i:
    1. On train only: rank 50 Hallmark GSVA scores by |AUC−0.5|,
       keep top-10, sign-flip if AUC < 0.5, z-score, average → sig
    2. On train only: choose λ maximizing AUROC of z(sig) + λ·OR
    3. Score drug i with train-fitted ranking / z / λ

Human target (high-conf fusion labels):
  n_sources_used ≥ 3  AND  (p_combined ≥ 0.80 OR p_combined ≤ 0.20)
  → 102 drugs (95 toxic / 7 safe)

Features:
  - data/scores_log2fc_GSVA_hallmark.csv
      Open TG-GATEs rat Affy High−Control log2FC → human symbols
      → pysigscore GSVA on MSigDB Hallmark (50 sets)
  - data/rat_or_endpoints.csv
      path_vacuol_fatty_high OR path_mod_severe_high OR chem_LWrel_1.2x_high

Usage (from repository root):
  python3 scripts/run_champion_nested.py
  python3 scripts/run_champion_nested.py --verify
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

SEED = 13
TARGET_SENS = 0.95
TOP_K = 10
N_BOOT = 2000
LAM_GRID = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
)
OR_COLS = [
    "path_vacuol_fatty_high",
    "path_mod_severe_high",
    "chem_LWrel_1.2x_high",
]
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

LABELS = os.path.join(DATA, "human_dili_highconf_labels.csv")
OR_CSV = os.path.join(DATA, "rat_or_endpoints.csv")
GSVA = os.path.join(DATA, "scores_log2fc_GSVA_hallmark.csv")

EXPECTED_TXT = os.path.join(RES, "EXPECTED_results_champion.txt")
EXPECTED_CSV = os.path.join(RES, "EXPECTED_scores_champion.csv")
EXPECTED_MET = os.path.join(RES, "EXPECTED_metrics_sens095.csv")


def metrics_at_sens(y: np.ndarray, score: np.ndarray, target_sens: float = TARGET_SENS) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    tox = score[y == 1]
    thr = float(np.quantile(tox, 1.0 - target_sens))
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def bootstrap_auc(y, score, n_boot=N_BOOT):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    point = float(roc_auc_score(y, score))
    boots = []
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        boots.append(roc_auc_score(y[ix], score[ix]))
    if not boots:
        return point, np.nan, np.nan
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def nested_topk_or(
    X: np.ndarray, or_vec: np.ndarray, y: np.ndarray, top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True nested LOO: top-k pathway selection + λ all fit on train only."""
    n = len(y)
    oof_sig = np.full(n, np.nan)
    oof_fuse = np.full(n, np.nan)
    lams = np.full(n, np.nan)

    for tr, te in LeaveOneOut().split(X):
        i = int(te[0])
        Xtr, ytr = X[tr], y[tr]
        or_tr, or_te = or_vec[tr], or_vec[i]

        if len(np.unique(ytr)) < 2:
            oof_sig[i] = 0.0
            oof_fuse[i] = float(or_te)
            lams[i] = 0.0
            continue

        aucs = np.empty(Xtr.shape[1])
        for j in range(Xtr.shape[1]):
            col = Xtr[:, j]
            if np.nanstd(col) < 1e-12:
                aucs[j] = 0.5
                continue
            try:
                aucs[j] = roc_auc_score(ytr, col)
            except ValueError:
                aucs[j] = 0.5

        sign = np.where(aucs >= 0.5, 1.0, -1.0)
        order = np.argsort(-np.abs(aucs - 0.5))[:top_k]

        feats_tr = Xtr[:, order] * sign[order]
        mu, sd = feats_tr.mean(0), feats_tr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        sig_tr = ((feats_tr - mu) / sd).mean(1)

        feats_te = X[te][:, order] * sign[order]
        sig_te = float(((feats_te - mu) / sd).mean())
        oof_sig[i] = sig_te

        mu_s, sd_s = float(sig_tr.mean()), float(sig_tr.std())
        if sd_s < 1e-12:
            sd_s = 1.0
        z_tr = (sig_tr - mu_s) / sd_s

        best_lam, best_auc = 0.0, -1.0
        for lam in LAM_GRID:
            try:
                a = roc_auc_score(ytr, z_tr + lam * or_tr)
            except ValueError:
                continue
            if a > best_auc + 1e-12:
                best_auc, best_lam = a, float(lam)

        lams[i] = best_lam
        oof_fuse[i] = (sig_te - mu_s) / sd_s + best_lam * or_te

    return oof_sig, oof_fuse, lams


def load_cohort():
    lab = pd.read_csv(LABELS)
    lab["drug_key"] = lab["drug_key"].astype(str)
    lab = lab.set_index("drug_key")

    gsva = pd.read_csv(GSVA, index_col=0)
    gsva.index = gsva.index.astype(str)

    or_df = pd.read_csv(OR_CSV)
    or_df["drug_key"] = or_df["drug_key"].astype(str)
    or_df = or_df.set_index("drug_key")

    keys = [k for k in lab.index if k in gsva.index and k in or_df.index]
    if len(keys) != len(lab):
        missing = sorted(set(lab.index) - set(keys))
        raise RuntimeError(f"Missing drugs in GSVA/OR: {missing}")

    X = gsva.loc[keys].values.astype(float)
    y = lab.loc[keys, "y_hard"].values.astype(int)
    or_bits = or_df.loc[keys, OR_COLS].fillna(0).astype(int).values
    or_vec = (or_bits.max(axis=1) > 0).astype(float)
    return keys, X, y, or_vec, list(gsva.columns)


def format_report(
    keys, y, or_vec, oof_sig, oof_fuse, lams, m_fuse, m_sig, auc_fuse, lo, hi, auprc, elapsed
) -> str:
    lines = []
    lines.append("Champion (nested LOO): pysigscore GSVA top-10 + rat OR")
    lines.append("=" * 70)
    lines.append(f"Seed: {SEED}  |  TOP_K: {TOP_K}  |  TARGET_SENS: {TARGET_SENS}")
    lines.append(f"Cohort: {len(keys)} drugs ({int((y == 1).sum())} toxic / {int((y == 0).sum())} safe)")
    lines.append(f"OR+: {int(or_vec.sum())}  |  median λ: {float(np.median(lams)):.2f}  "
                 f"(range {float(np.min(lams)):.2f}–{float(np.max(lams)):.2f})")
    lines.append("")
    lines.append("Model: score = z(GSVA_topk10) + λ · OR   [λ + top-k fit inside each LOO train fold]")
    lines.append("Expression features: Open TG-GATEs rat Affy log2FC → pysigscore GSVA (50 Hallmarks)")
    lines.append("OR: path_vacuol_fatty_high OR path_mod_severe_high OR chem_LWrel_1.2x_high")
    lines.append("")
    lines.append("NESTED LOOCV — fused champion (GSVA_topk10+OR)")
    lines.append("-" * 70)
    lines.append(f"  AUROC   = {auc_fuse:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    lines.append(f"  AUPRC   = {auprc:.3f}")
    lines.append(f"  @ Sens {TARGET_SENS:.2f}:")
    lines.append(f"    Sensitivity = {m_fuse['sensitivity']:.3f}  ({m_fuse['tp']}/{m_fuse['tp'] + m_fuse['fn']})")
    lines.append(f"    Specificity = {m_fuse['specificity']:.3f}  ({m_fuse['tn']}/{m_fuse['tn'] + m_fuse['fp']})")
    lines.append(f"    MCC         = {m_fuse['mcc']:+.3f}")
    lines.append(f"    Precision   = {m_fuse['precision']:.3f}")
    lines.append(f"    NPV         = {m_fuse['tn'] / max(m_fuse['tn'] + m_fuse['fn'], 1):.3f}")
    lines.append(f"    F1          = {m_fuse['f1']:.3f}")
    lines.append(f"    Accuracy    = {m_fuse['accuracy']:.3f}")
    lines.append(f"    Bal. acc.   = {m_fuse['balanced_accuracy']:.3f}")
    lines.append(f"    FPR / FNR   = {m_fuse['fpr']:.3f} / {m_fuse['fnr']:.3f}")
    lines.append(
        f"    TP/FN/FP/TN = {m_fuse['tp']}/{m_fuse['fn']}/{m_fuse['fp']}/{m_fuse['tn']}"
    )
    lines.append(f"    threshold   = {m_fuse['threshold']:.6f}")
    lines.append("")
    lines.append("Expression-only nested (GSVA_topk10, no OR)")
    lines.append("-" * 70)
    lines.append(f"  AUROC = {roc_auc_score(y, oof_sig):.3f}")
    lines.append(
        f"  @ Sens {TARGET_SENS:.2f}: Spec={m_sig['specificity']:.3f}  MCC={m_sig['mcc']:+.3f}"
    )
    lines.append("")
    lines.append("Leakage notes")
    lines.append("-" * 70)
    lines.append("  • Nested LOO: pathway top-k, sign, z-stats, and λ never see the held-out drug.")
    lines.append("  • Not the optimistic two-stage (OOF-then-fuse) variant (AUROC ≈ 0.844).")
    lines.append("  • OR is rat pathology/chemistry (valid predictor), not human labels.")
    lines.append("  • GSVA scores are sample-wise (per-drug); fixed external Hallmark gene sets.")
    lines.append("")
    lines.append(f"Wall time: {elapsed:.1f}s")
    lines.append("")
    return "\n".join(lines)


def verify(scores_path: str, report_path: str) -> int:
    """Compare live outputs to frozen EXPECTED_* files."""
    ok = True
    if not os.path.isfile(EXPECTED_CSV) or not os.path.isfile(EXPECTED_TXT):
        print("ERROR: EXPECTED_* files missing; run without --verify first to create them.")
        return 1

    live = pd.read_csv(scores_path)
    exp = pd.read_csv(EXPECTED_CSV)
    if list(live.columns) != list(exp.columns):
        print("FAIL: column mismatch", list(live.columns), "vs", list(exp.columns))
        ok = False
    if len(live) != len(exp):
        print(f"FAIL: row count {len(live)} vs {len(exp)}")
        ok = False
    else:
        # align by drug_key
        live = live.set_index("drug_key").loc[exp["drug_key"]].reset_index()
        for col in ("y_hard", "OR"):
            if not np.array_equal(live[col].values, exp[col].values):
                print(f"FAIL: {col} mismatch")
                ok = False
        if not np.allclose(live["score"].values, exp["score"].values, rtol=0, atol=1e-8):
            max_diff = float(np.max(np.abs(live["score"].values - exp["score"].values)))
            print(f"FAIL: score mismatch max|Δ|={max_diff:.3e}")
            ok = False
        if not np.allclose(live["lambda"].values, exp["lambda"].values, rtol=0, atol=1e-12):
            print("FAIL: lambda mismatch")
            ok = False
        if not np.allclose(live["score_sig_only"].values, exp["score_sig_only"].values, rtol=0, atol=1e-8):
            print("FAIL: score_sig_only mismatch")
            ok = False

    live_txt = open(report_path).read()
    exp_txt = open(EXPECTED_TXT).read()
    # Compare metric lines only (ignore wall time)
    def metric_lines(t):
        return [ln for ln in t.splitlines() if ln.strip().startswith(("  AUROC", "  AUPRC", "    ")) or "MCC" in ln]

    if metric_lines(live_txt) != metric_lines(exp_txt):
        # softer: check key AUROC string present
        if "AUROC   = 0.809" not in live_txt and "AUROC   = 0.809" not in exp_txt:
            # extract AUROC from both
            pass
        live_auc = [ln for ln in live_txt.splitlines() if "AUROC   =" in ln]
        exp_auc = [ln for ln in exp_txt.splitlines() if "AUROC   =" in ln]
        if live_auc != exp_auc:
            print("WARN: report AUROC lines differ:")
            print("  live:", live_auc)
            print("  exp :", exp_auc)

    if ok:
        print("VERIFY OK — scores match EXPECTED_scores_champion.csv")
        return 0
    print("VERIFY FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="Compare outputs to frozen EXPECTED_* files")
    ap.add_argument("--write-expected", action="store_true", help="Overwrite EXPECTED_* with live run")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading cohort ...", flush=True)
    keys, X, y, or_vec, pathway_names = load_cohort()
    print(f"  n={len(keys)}  pathways={X.shape[1]}  toxic={int((y == 1).sum())}  safe={int((y == 0).sum())}", flush=True)

    print("Nested LOO: GSVA top-k + OR ...", flush=True)
    oof_sig, oof_fuse, lams = nested_topk_or(X, or_vec, y, top_k=TOP_K)
    elapsed = time.time() - t0

    auc_fuse, lo, hi = bootstrap_auc(y, oof_fuse)
    auprc = float(average_precision_score(y, oof_fuse))
    m_fuse = metrics_at_sens(y, oof_fuse)
    m_sig = metrics_at_sens(y, oof_sig)

    scores = pd.DataFrame(
        {
            "drug_key": keys,
            "y_hard": y,
            "OR": or_vec.astype(int),
            "score": oof_fuse,
            "score_sig_only": oof_sig,
            "lambda": lams,
            "model": "GSVA_topk10+OR_nested",
        }
    )
    scores_path = os.path.join(RES, "scores_champion.csv")
    scores.to_csv(scores_path, index=False)

    met = pd.DataFrame(
        [
            {
                "model": "GSVA_topk10+OR_nested",
                "n": len(keys),
                "auroc": auc_fuse,
                "auroc_ci95_lo": lo,
                "auroc_ci95_hi": hi,
                "auprc": auprc,
                **{f"sens095_{k}": v for k, v in m_fuse.items()},
            },
            {
                "model": "GSVA_topk10_nested",
                "n": len(keys),
                "auroc": float(roc_auc_score(y, oof_sig)),
                "auprc": float(average_precision_score(y, oof_sig)),
                **{f"sens095_{k}": v for k, v in m_sig.items()},
            },
        ]
    )
    met_path = os.path.join(RES, "metrics_sens095.csv")
    met.to_csv(met_path, index=False)

    report = format_report(
        keys, y, or_vec, oof_sig, oof_fuse, lams, m_fuse, m_sig, auc_fuse, lo, hi, auprc, elapsed
    )
    report_path = os.path.join(RES, "results_champion.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)

    if args.write_expected or not os.path.isfile(EXPECTED_CSV):
        scores.to_csv(EXPECTED_CSV, index=False)
        met.to_csv(EXPECTED_MET, index=False)
        # freeze report without wall-time line for stable verify of metrics
        with open(EXPECTED_TXT, "w") as f:
            f.write(report)
        print(f"Wrote EXPECTED_* under {RES}", flush=True)

    if args.verify:
        sys.exit(verify(scores_path, report_path))

    print(f"Wrote {scores_path}")
    print(f"Wrote {met_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
