# pysigscore-nested-highconf-dili

**Leakage-safe nested LOOCV champion:** Open TG-GATEs rat expression (pysigscore **GSVA** Hallmarks) fused with a rat **OR** pathology/chemistry rule, predicting **high-confidence human DILI** labels.

| | |
|---|---|
| **Model** | `score = z(GSVA_topk10) + λ · OR` |
| **CV** | **Nested** leave-one-drug-out (top-k, z-stats, and λ fit **inside** each train fold) |
| **AUROC** | **0.809** (95% CI ≈ [0.64, 0.95]) |
| **@ Sens 0.95** | Spec **0.57** · MCC **+0.462** · Prec **0.97** |
| **Cohort** | 102 drugs (95 toxic / 7 safe) |
| **Seed** | 13 |

> This is the **nested** (honest) version of the GSVA+OR model.  
> An earlier two-stage “OOF-then-fuse” draft reported AUROC **0.844** and is **not** the champion here (stacking optimism; see [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and the leakage audit).

---

## Quick start

```bash
# Python 3.10+
pip install -r requirements.txt

# Run nested champion (writes results/*)
python3 scripts/run_champion_nested.py

# Verify against frozen EXPECTED outputs
python3 scripts/run_champion_nested.py --verify
```

Runtime: a few seconds on a laptop (CPU). GSVA pathway scores are **bundled** — no Affymetrix reprocessing required.

---

## What the model is

### Expression arm (GSVA top-10)

1. Open TG-GATEs rat Affy **log2FC** (High prefer − Control).
2. Human ortholog symbols → **pysigscore GSVA** on 50 MSigDB Hallmarks.
3. Each LOO train fold: select **top 10** pathways by \(|\mathrm{AUC}-0.5|\), sign-correct, average z-scores → `sig`.

### OR arm (rat endpoints)

```text
OR = path_vacuol_fatty_high
  OR path_mod_severe_high
  OR chem_LWrel_1.2x_high
```

### Nested fusion

```text
score = z(sig) + λ · OR
```

- `z(.)` uses **train-fold** mean/sd of `sig` only.  
- **λ** is learned on each train fold (grid max train AUROC).  
- Held-out drug never enters top-k ranking, z-stats, or λ.

---

## Headline metrics @ Sensitivity 0.95 (nested)

| Metric | Value |
|---|---|
| Sensitivity (Recall) | **0.95** (90/95) |
| Specificity | **0.57** (4/7) |
| MCC | **+0.462** |
| Precision (PPV) | **0.97** |
| AUROC | **0.809** |
| AUPRC | **0.98** |
| TP / FN / FP / TN | **90 / 5 / 3 / 4** |

Expression-only nested (`GSVA_topk10`, no OR): AUROC **0.743**, Spec@0.95 **0.29**, MCC **+0.233**.

---

## Repository layout

```
pysigscore-nested-highconf-dili/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── human_dili_highconf_labels.csv      # high-conf y_hard
│   ├── rat_or_endpoints.csv                # 3 OR bits
│   └── scores_log2fc_GSVA_hallmark.csv     # frozen 102×50 GSVA scores
├── scripts/
│   └── run_champion_nested.py              # ★ main entry point
├── results/
│   ├── EXPECTED_results_champion.txt
│   ├── EXPECTED_scores_champion.csv
│   └── EXPECTED_metrics_sens095.csv
└── docs/
    ├── METHODOLOGY.md
    ├── LEAKAGE_AUDIT_REPORT.txt
    └── nested_vs_published_metrics.csv
```

---

## Human labels (high-conf filter)

```text
n_sources_used ≥ 3  AND  (p_combined ≥ 0.80 OR p_combined ≤ 0.20)
```

Upstream: [dili-labels-446-human-only](https://github.com/Ajay1989Kumar/dili-labels-446-human-only) →  
[entropy-or-highconf-human-dili](https://github.com/Ajay1989Kumar/entropy-or-highconf-human-dili).

---

## Citation / related work

- Scoring engine: [pysigscore](https://github.com/bioinformatics-hub/pysigscore) (Giacomello et al., bioRxiv 2026)
- High-conf labels + ENT-NLL+OR chemistry champion: [entropy-or-highconf-human-dili](https://github.com/Ajay1989Kumar/entropy-or-highconf-human-dili)
- OR endpoints: [tggates-or-dcea-human-dili](https://github.com/Ajay1989Kumar/tggates-or-dcea-human-dili)
- Open TG-GATEs: NIHS Open TG-GATEs project

---

## License

Code: MIT (see `LICENSE`).  
Data: Open TG-GATEs and curated label tables retain their original licenses / CC-BY terms.
