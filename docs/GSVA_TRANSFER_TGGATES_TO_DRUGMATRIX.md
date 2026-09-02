# The GSVA signature: strong on TG-GATEs, transfers well to DrugMatrix

This note isolates the **expression-only GSVA pathway signature** — the frozen part of
the champion — and shows that it is the component that both (a) performs well internally
on Open TG-GATEs and (b) generalizes across lab/platform to DrugMatrix (Iconix / NTP,
GEO **GSE57815**). See [`EXTERNAL_VALIDATION_DRUGMATRIX.md`](EXTERNAL_VALIDATION_DRUGMATRIX.md)
for the full two-arm study and [`METHODOLOGY.md`](METHODOLOGY.md) for the signature build.

## GSVA signature — the transferable model

| Metric | TG-GATEs (internal LOOCV, n=101) | DrugMatrix (frozen transfer, n=57) | drop |
|--------|----------------------------------|------------------------------------|------|
| AUROC (threshold-free) | 0.748 (perm-p .013) | 0.721 (perm-p .014) | −0.027 |
| best-MCC | +0.462 (Sens 0.95 / Spec 0.57) | +0.361 (Sens 0.57 / Spec 0.90) | −0.10 |
| MCC @ Sens ≈ 0.95 | +0.462 (Spec 0.57) | +0.183 (Spec 0.20) | −0.28 |

The signature keeps ~96% of its internal AUROC when applied — **frozen, without any
refitting or DrugMatrix labels** — to an independent database with a different platform
batch and a different set of drugs. Both AUROCs are significant against a
label-permutation null (perm-p ≈ .013–.014).

**MCC is reported at two matched operating points**, because it is threshold-dependent.
*best-MCC* takes the MCC-optimal cutoff on each set — internally this happens to be the
95%-sensitivity point (+0.462), but on DrugMatrix the optimum trades sensitivity for
specificity (Sens 0.57 / Spec 0.90) to reach +0.361. Held to a *fixed high-sensitivity*
threshold (Sens ≈ 0.95), the transferred MCC is +0.183 — the cost of freezing a decision
threshold across a base-rate shift. AUROC (rank-based, threshold-free) is the reliable
transfer summary; the MCC gap is an operating-point/threshold effect, not a loss of
discrimination.

## Why the GSVA signature is the model that transfers

The signature is a **10-parameter** summary: the 10 MSigDB Hallmark pathways with the
largest, most stable train-AUROC, sign-corrected and z-scored into a single score
(`METHODOLOGY.md`, Features §A; K=10). That low complexity is exactly what survives the
cross-platform jump (Affymetrix RG230 DrugMatrix vs TG-GATEs) — batch effects are
absorbed at the pathway level, and there are too few free parameters to overfit the
training lab.

By contrast, a 48-feature L2-logistic on the same GSVA columns is **not** a reliable
transferable model: under internal LOOCV it collapses to chance (AUROC 0.506) because
of the 94-toxic / 7-safe class imbalance — dropping a single negative swings the
decision boundary. Its high single-fit external number is therefore not a reproducible
cross-validated result. The **GSVA signature is the fair, consistent model across every
protocol** (internal LOOCV, within-DrugMatrix LOOCV, and frozen transfer).

## Matched-protocol context (why 0.721 is strong)

Using the *same* within-dataset LOOCV protocol on each database, and the frozen transfer
for contrast (GSVA signature):

| Setting | Protocol | AUROC | perm-p | best MCC |
|---------|----------|-------|--------|----------|
| TG-GATEs (n=101) | within-dataset LOOCV | 0.748 | .013 | +0.462 |
| DrugMatrix (n=57) | within-dataset LOOCV (matched) | 0.632 | .10 | +0.415 |
| TG → DrugMatrix (n=57) | frozen transfer | 0.721 | .014 | +0.361 |

The frozen TG-GATEs signature transferred to DrugMatrix (**0.721**) **beats DrugMatrix's
own within-dataset LOOCV (0.632)** on the same 57 drugs: a model that never saw a
DrugMatrix label predicts DrugMatrix DILI better than one trained on DrugMatrix itself.
DrugMatrix self-LOOCV trains each fold on ~56 drugs (10 safe) — too few, too noisy — so
the imported TG-GATEs signature is genuinely additive, not recoverable from DrugMatrix
alone.

## Reproduce

```
python scripts/reproduce_gsva_transfer.py
```

Uses only committed data: `data/scores_log2fc_GSVA_hallmark.csv` (TG-GATEs training
GSVA), `data/human_dili_highconf_labels.csv` (labels), and
`data/external/drugmatrix_gsva_hallmark.parquet` +
`results/external_validation/scores_expression_only.csv` (DrugMatrix external cohort,
69 drugs; 57 with a full measured rat panel). 48 Hallmark pathways are shared across
the two platforms.
