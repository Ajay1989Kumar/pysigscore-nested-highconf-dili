# pysigscore-nested-highconf-dili

**Leakage-safe nested LOOCV champion:** Open TG-GATEs rat expression (pysigscore **GSVA** Hallmarks) fused with a rat **OR** pathology/chemistry rule, predicting **high-confidence human DILI** labels.

| | |
|---|---|
| **Model** | `score = z(GSVA_topk10) + λ · OR` |
| **CV** | **Nested** leave-one-drug-out (top-k, z-stats, and λ fit **inside** each train fold) |
| **AUROC** | **0.790** (95% CI ≈ [0.58, 0.97]) |
| **@ Sens 0.95** | Spec **0.57** · MCC **+0.462** · Prec **0.97** |
| **Cohort** | 101 drugs (94 toxic / 7 safe), high-conf labels **without Manual** |
| **Seed** | 13 |

> This is the **nested** (honest) version of the GSVA+OR model.  
> Two-stage “OOF-then-fuse” on this same cohort is AUROC **0.845** and is **not** the champion (stacking optimism; see [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and the leakage audit).

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

### Unsupervised clustering (toxic vs safe geometry)

```bash
# Needs matplotlib + seaborn (see requirements.txt)
python3 scripts/unsupervised_clustering.py
```

Labels are used **only post-hoc** for colouring/metrics — never for fitting PCA, t-SNE, k-means, or hierarchical clustering.  
Frozen figures and tables live under [`results/unsupervised/`](results/unsupervised/) (see report there).

**Headline:** full Hallmark pathway space does **not** form clean toxic vs safe clusters (silhouette &lt; 0, k-means ARI ≈ 0, balanced accuracy ≈ 0.5). Supervised nested GSVA top-10 + OR remains the predictive champion.

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
| Sensitivity (Recall) | **0.95** (89/94) |
| Specificity | **0.57** (4/7) |
| MCC | **+0.462** |
| Precision (PPV) | **0.97** |
| AUROC | **0.790** |
| AUPRC | **0.98** |
| TP / FN / FP / TN | **89 / 5 / 3 / 4** |

Expression-only nested (`GSVA_topk10`, no OR): AUROC **0.746**, Spec@0.95 **0.29**, MCC **+0.233**.

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
│   └── scores_log2fc_GSVA_hallmark.csv     # frozen 101×50 GSVA scores
├── scripts/
│   ├── run_champion_nested.py              # ★ nested LOO champion
│   └── unsupervised_clustering.py          # PCA / t-SNE / k-means / heatmap
├── results/
│   ├── EXPECTED_results_champion.txt
│   ├── EXPECTED_scores_champion.csv
│   ├── EXPECTED_metrics_sens095.csv
│   └── unsupervised/                       # clustering figures + metrics
│       ├── unsupervised_report.txt
│       ├── summary_unsupervised_GSVA.png
│       ├── pca_*.png / tsne_*.png
│       ├── heatmap_GSVA_clustered.png
│       └── separation_metrics.csv
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

Seven bulk **human-clinical** sources only (DILIrank, DILIst, LiverTox, SIDER hepatic, Greene HUMANS, T3DB-clinical, ATSDR-clinical). The hand-curated **Manual** source is **excluded**. Dropping Manual removes **hexachlorobenzene** (it had only 2 remaining informative sources).

Upstream: [dili-labels-446-human-only](https://github.com/Ajay1989Kumar/dili-labels-446-human-only) →  
[entropy-or-highconf-human-dili](https://github.com/Ajay1989Kumar/entropy-or-highconf-human-dili) (`human_dili_highconf_labels_nomanual.csv`).

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
