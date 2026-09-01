# External validation on DrugMatrix

Independent, cross-dataset validation of the nested GSVA + rat-endpoint DILI model
(the champion in [`README.md`](../README.md)) on the **DrugMatrix** toxicogenomics
database (Iconix / NTP, GEO **GSE57815**) — a different lab, platform batch, and set
of drugs from the Open TG-GATEs cohort the model was trained on.

**Headline:** the model architecture generalizes. Combining the (frozen) expression
arm with a **recalibrated** rat-endpoint arm reaches **AUROC 0.791 / MCC +0.478** on
independent DrugMatrix drugs — matching the internal champion (0.795 / +0.462) — and
on a cohort with **10 safe drugs** (vs 7 internally). But the two arms transfer
differently, which is the main scientific finding (below).

---

## Cohort

| | |
|---|---|
| DrugMatrix drugs with rebuilt GSVA | 200 |
| Overlap with TG-GATEs training (excluded to avoid leakage) | 42 |
| **External high-conf labeled (not in training)** | **69** (55 toxic / 14 safe) |
| External drugs with full measured rat panel | 57 (47 toxic / 10 safe) |

High-conf labels use the same rule as the champion (`n_sources ≥ 3` and
`p_combined ≥ 0.80` or `≤ 0.20`), from the 7 bulk human-clinical sources (no Manual).

## Results

| Stage | Model | AUROC | 95% CI | best MCC |
|---|---|---:|---|---:|
| Expression-only | GSVA top-10 signature (frozen) | 0.655 | [0.47, 0.82] | — |
| Expression-only | L2-logistic, 48 GSVA (frozen) | 0.706 | [0.53, 0.85] | — |
| Frozen GBM | expression arm (frozen) | 0.723 | [0.55, 0.88] | +0.353 |
| Frozen GBM | rat-panel arm (frozen) | **0.470** | [0.32, 0.61] | +0.173 |
| Frozen GBM | full GBM (GSVA + rat), frozen | **0.499** | [0.32, 0.69] | +0.081 |
| Diagnostic | DrugMatrix-native ALT → DILI | 0.737 | — | — |
| Diagnostic | DrugMatrix-native OR-rule → DILI | 0.627 | — | — |
| Recalibrated | expression (frozen) | 0.723 | [0.54, 0.88] | +0.353 |
| Recalibrated | rat arm (refit on DM scale) | 0.736 | [0.58, 0.87] | +0.408 |
| Recalibrated | GSVA(frozen) + rat(recal), meta-LR | 0.755 | [0.59, 0.89] | +0.394 |
| **Recalibrated** | **GSVA(frozen) + rat(recal), z-sum** | **0.791** | **[0.64, 0.92]** | **+0.478** |

Recalibrated rows are LOOCV **within** the 57-drug external cohort (the rat arm is
refit on DrugMatrix's own endpoint scale); the expression arm is always the frozen
TG-GATEs model applied without retraining. See
[`results/external_validation/metrics_summary.csv`](../results/external_validation/metrics_summary.csv).

## Findings

1. **Expression (GSVA) transfers frozen.** The TG-GATEs-trained signature scores
   **0.72** on unseen DrugMatrix drugs with no retraining — matching its internal
   performance. Pathway-level Hallmark enrichment is platform-robust.

2. **The rat-endpoint signal is real and replicates**, but is **dataset-bound in
   encoding**. DrugMatrix's own ALT predicts human DILI at **0.737** and its native
   OR-rule at 0.63 — so the cross-species rat→human signal is not a TG-GATEs
   artifact. However, a rat model *frozen* on TG-GATEs endpoints collapses to chance
   on DrugMatrix (0.47) because endpoint definitions and base rates differ
   (OR+ base rate 0.68 in TG-GATEs vs 0.51 in DrugMatrix). This is **domain shift,
   not signal absence**.

3. **Recalibration recovers champion-level performance.** Refitting only the rat-arm
   thresholds/weights on DrugMatrix, then fusing with the frozen expression arm,
   restores **AUROC 0.791 / MCC +0.478** — matching the internal champion, on
   independent data with more negatives. Complementarity holds cross-dataset
   (combined > either arm alone).

**Deployment takeaway:** ship the expression arm as-is (portable across platforms);
**refit the rat-endpoint thresholds on each new dataset.**

## Caveats

- Recalibrated results are LOOCV within a 57-drug / 10-safe external cohort, so CIs
  are wide. The point estimate matching the internal champion **on independent data**
  is the strength; the operating-point MCC remains sensitive to the 10 negatives.
- DrugMatrix histopathology from the ToxCompl *completed* matrix is positivity-biased
  (predicted cells never reach severity 0); this analysis therefore uses **measured**
  endpoint values only (from the ToxCompl input matrix), which are unbiased but sparser.
- Cross-platform batch effects (Affymetrix RG230 DrugMatrix vs TG-GATEs) are absorbed
  at the pathway level but not eliminated.

## Data sources

- **Expression (measured):** DrugMatrix-Affymetrix GSE57815 RMA (BrainArray ENTREZG),
  mirrored in the unified matrix on Hugging Face
  (`ajaygeetakumar/dili-toxicogenomics-expression`, `unified-446-rat-liver/affy_dm_affy.tsv`).
  Vehicle controls (dose 0, matched by vehicle + time) identified from the GEO
  GSE57815 series-matrix sample annotations.
- **Measured liver endpoints:** ToxCompl input matrix
  `probe_clinical_pathology.xlsx` and endpoint dictionary `C_H_M annotation.txt` from
  NTP CEBS (`cebs-ext.niehs.nih.gov/cahs/file/download/ornl/`); paper:
  Combs et al., *Completion of the DrugMatrix Toxicogenomics Database*
  (bioRxiv 2024.03.26.586669). Endpoint definitions also in Te / AbdulHameed et al.,
  *Characterization of Chemically Induced Liver Injuries*, PLoS ONE 2014
  (`10.1371/journal.pone.0107230`).
- **Human DILI labels:** high-conf fusion from
  [entropy-or-highconf-human-dili](https://github.com/Ajay1989Kumar/entropy-or-highconf-human-dili).

## Pipeline (reproduction)

Scripts under [`scripts/external_validation/`](../scripts/external_validation/)
(paths are hardcoded to the working directory and must be adapted; large source
files are not committed — see Data sources):

1. `build_dm_gsva.py` — DrugMatrix per-drug High−(vehicle control) log2FC → human
   orthologs (mygene) → **gseapy GSVA** on 50 MSigDB Hallmarks. The identical pipeline
   reproduces the frozen 101 TG-GATEs GSVA at per-drug r = **0.949** (validity gate).
2. `extract_measured.py` — pull measured ALT/AST + liver histopathology severities
   from the ToxCompl input matrix (endpoint rows are the first ~180 rows).
3. `run_gbm_external.py` — apply the frozen TG-GATEs GBM (GSVA signature + rat panel)
   to the external cohort; decompose by arm.
4. `recal_rat.py` — refit the rat arm on DrugMatrix-native endpoints and re-fuse
   with the frozen expression arm (LOOCV).

Bundled derived artifacts:
[`data/external/drugmatrix_gsva_hallmark.parquet`](../data/external/drugmatrix_gsva_hallmark.parquet)
(200×48), `drugmatrix_measured_liver_endpoints.parquet`, and
`CHM_endpoint_annotation.txt`.
