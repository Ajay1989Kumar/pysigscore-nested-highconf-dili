# Methodology — nested GSVA top-10 + rat OR

## Target

High-confidence human DILI labels from multi-source independent-Bayes fusion:

```text
n_sources_used ≥ 3  AND  (p_combined ≥ 0.80 OR p_combined ≤ 0.20)
```

Cohort: **102 drugs** (95 toxic / 7 safe).  
Source: [entropy-or-highconf-human-dili](https://github.com/Ajay1989Kumar/entropy-or-highconf-human-dili).

## Features

### A. pysigscore GSVA Hallmark scores (50)

1. Open TG-GATEs **rat** Affymetrix liver expression.
2. Per drug: High (prefer) / Middle / Low mean − Control mean → **log2FC**.
3. Map rat Entrez → human gene symbols (ortholog table); average multi-probe symbols.
4. Score MSigDB Hallmark gene sets with **[pysigscore](https://github.com/bioinformatics-hub/pysigscore) GSVA** (sample-wise).

Bundled frozen matrix: `data/scores_log2fc_GSVA_hallmark.csv` (102 × 50).

### B. Rat OR rule (binary)

```text
OR = path_vacuol_fatty_high
  OR path_mod_severe_high
  OR chem_LWrel_1.2x_high
```

Open TG-GATEs pathology / relative liver weight at High dose.  
**Not** human labels — valid cross-species predictor.

## Nested leave-one-drug-out

For each held-out drug \(i\):

1. **Train-only pathway selection:** for each of 50 GSVA columns, compute train AUROC vs `y_hard`. Rank by \(|\mathrm{AUC}-0.5|\). Keep top \(K=10\). Sign-flip columns with AUC &lt; 0.5.
2. **Train-only z-score** of those 10 columns; **mean** → continuous signature `sig`.
3. **Train-only λ:** grid search maximizing train AUROC of \(z(\mathrm{sig})+\lambda\cdot\mathrm{OR}\).
4. Apply train ranking / z / λ to drug \(i\).

This avoids **two-stage stacking leakage** (OOF-then-fuse), which inflated AUROC from **0.809 → 0.844** in an earlier draft.

## Operating point

Threshold so that ≈95% of toxic drugs score above it (post-hoc on OOF toxic scores).  
Primary ranking metric: **AUROC** (threshold-free).

## Headline nested metrics

| Model | AUROC | Spec@Sens0.95 | MCC@Sens0.95 |
|-------|------:|--------------:|-------------:|
| **GSVA_topk10+OR (nested)** | **0.809** | **0.571** | **+0.462** |
| GSVA_topk10 (nested, no OR) | 0.743 | 0.286 | +0.233 |
| Two-stage OOF+fuse (do not use) | 0.844 | 0.571 | +0.462 |

See `docs/LEAKAGE_AUDIT_REPORT.txt` for the full audit.
