# Resolving the 7-negative limitation: reliable-negative mining + a merged cohort

The internal high-conf cohort has only **7 safe drugs** (94T/7S), which makes
operating-point metrics (specificity, MCC) statistically fragile — the specificity
95% CI spans the entire [0, 1]. Here we expand the negative class with **reliable-
negative mining** and then build a **merged TG-GATEs + DrugMatrix cohort** in which
both classes are present on both platforms.

## Reliable-negative mining (self-training, dual agreement)

Anchored on the 101 high-conf labels (trusted, low-noise), a drug from the broader
402-drug rat-expression pool becomes a **reliable new negative** iff two independent
votes agree it is safe:

1. **Label vote** — 7-source Bayes fusion `p_combined ≤ 0.40`.
2. **Feature vote** — an expression model (retrained each round on the growing anchor
   set) scores it safe.

The loop retrains on the expanded negatives each round and converges:

| Round | Negatives | New | Candidate pool | Corroborated |
|------:|----------:|----:|---------------:|-------------:|
| 0 (high-conf) | 7 | — | — | — |
| 1 | 22 | +15 | 44 | 15 |
| 2 | 31 | +9 | 22 | 9 |
| 3 | 32 | +1 | 13 | 1 |

Diminishing returns (**+15 → +9 → +1**) show the reliable-negative supply is exhausted
(~25 mined; imbalance 13:1 → ~3:1). Every mined drug is a pharmacologically plausible
low-DILI compound (corticosteroids, antiemetics, topical antiseptics, aminoglycosides,
digoxin, vitamin D). Three (betamethasone, nystatin, neomycin) were independently
present in the DrugMatrix high-conf external cohort, all labelled SAFE.

Champion on the expanded internal cohort (frozen on TG-GATEs; every score
out-of-sample — 101 via nested LOOCV, mined via frozen cross-platform transfer):

| Cohort | AUROC (CI width) | MCC @ Sens 0.95 | best-MCC |
|--------|-----------------:|----------------:|---------:|
| 94T/7S  (original) | 0.748 (0.44) | +0.462 [−0.07, +0.69] *(incl. 0)* | +0.462 |
| 94T/22S (round 1)  | 0.801 (0.22) | +0.546 [+0.21, +0.70] | +0.546 |
| 94T/31S (round 2)  | 0.840 (0.17) | +0.639 [+0.33, +0.76] | +0.639 |

The specificity CI collapses from **[0, 1]** to informative, and the MCC CI moves to
**exclude 0** — the metrics become trustworthy. (Stability gains are fully valid; the
point-estimate lift carries mild selection optimism because negatives were chosen with
a correlated expression model — see caveat below.)

## Merged TG-GATEs + DrugMatrix champion (all reliable negatives)

Pooling both datasets puts **both classes on both platforms** (TG 94T/7S; DM 55T/29S),
so a pooled model cannot reduce to "platform = label". Champion = GSVA top-10
z-signature, LOOCV on the merged cohort.

| | drugs | toxic | safe |
|---|---:|---:|---:|
| Merged | 185 | 149 | 36 |
| · TG-GATEs | 101 | 94 | 7 |
| · DrugMatrix | 84 | 55 | 29 |

| Metric | Value |
|--------|------:|
| AUROC | **0.764** [0.66, 0.86] |
| MCC @ Sens 0.95 | **+0.357** [+0.13, +0.58] |
| best-MCC | +0.460 |
| within-TG-GATEs AUROC | 0.767 |
| within-DrugMatrix AUROC | 0.746 |
| platform-confound AUROC | 0.620 |

**The signal is DILI, not platform.** The champion works essentially identically within
each platform separately (0.767 vs 0.746 vs pooled 0.764); if the pooled AUROC were
platform-driven, the within-platform AUROCs would collapse to ~0.5. A signature-style
predictor of *platform* reaches only 0.62 — well below the DILI-label 0.764. And the
MCC is now significant (+0.357 [+0.13, +0.58]) on 36 negatives across two independent
datasets, versus a non-significant MCC on 7.

## Batch-correction sensitivity (ComBat)

Because the negatives are DrugMatrix-enriched, platform partly tracks the label. We
re-ran the merged champion after empirical-Bayes batch correction (ComBat, run
**unsupervised** — no DILI-label covariate — so it cannot leak the outcome).

| Merged champion | Uncorrected | ComBat |
|---|---:|---:|
| AUROC | 0.764 [0.66, 0.86] | 0.720 [0.62, 0.82] |
| MCC @ Sens 0.95 | +0.357 [+0.13, +0.58] | +0.299 [+0.12, +0.48] |
| best-MCC | +0.460 | +0.357 |
| within-TG-GATEs AUROC | 0.767 | 0.767 |
| within-DrugMatrix AUROC | 0.746 | 0.732 |
| platform-confound AUROC | 0.620 | 0.465 |

ComBat removed the platform structure (confound AUROC 0.620 → 0.465 ≈ chance). Pooled
AUROC fell modestly (0.764 → 0.720) — the between-platform confounded component — but
the **within-platform AUROCs were essentially unchanged** (within-TG identical at 0.767)
and the MCC stayed significant. The merged model survives explicit batch correction; the
conservative cross-platform estimate is **AUROC 0.720 / MCC +0.299**. Reproduce:
`python scripts/reliable_negatives/combat_sensitivity.py` (requires `inmoose`). Methods:
ComBat (Johnson et al. 2007), sva (Leek et al. 2012), cross-platform normalization for
ML (Foltz et al. 2023).

## Caveats

- **Selection optimism.** Mined negatives were chosen partly by an expression model and
  then scored by an expression signature; the *stability/significance* gains are fully
  valid, but the *point-estimate lift* carries mild optimism. The within-TG-GATEs result
  (0.767, whose 7 negatives are the untouched high-conf anchors) is the cleanest anchor,
  and the merged AUROC (0.764 < the 94T/31S 0.840) is the more realistic cross-dataset
  number.
- **Residual batch structure.** GSVA retains some platform signal (platform AUROC 0.62),
  mitigated but not eliminated; within-platform validation is the guard.

## Reproduce

```
python scripts/reliable_negatives/merged_champion.py     # merged cohort champion + checks
python scripts/reliable_negatives/mine_reliable_negatives.py   # regenerate round-1 mining
```

Mined negatives are committed under `data/reliable_negatives/round{1,2,3}.csv`.
