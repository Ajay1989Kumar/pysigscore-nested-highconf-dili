# Cross-species prediction of high-confidence human drug-induced liver injury from rat toxicogenomics: a leakage-safe TG-GATEs model and its external validation on DrugMatrix

**Authors:** Ajay Kumar¹ *(with computational assistance from Claude Code)*
¹ *Affiliation to be completed.*

---

## Abstract

**Background.** Drug-induced liver injury (DILI) is a leading cause of drug attrition and post-market withdrawal. Rat toxicogenomic resources such as Open TG-GATEs pair *in vivo* liver transcriptomics with classical toxicology endpoints, but whether models trained on one such resource transfer to another — and which data modality carries the transferable signal — is rarely tested.

**Methods.** We predicted **high-confidence human DILI** labels (independent-Bayes fusion of seven bulk human-clinical sources; 101 drugs, 94 toxic / 7 safe) from Open TG-GATEs rat liver data using two feature modalities: (i) **pathway expression** — MSigDB Hallmark gene-set variation analysis (GSVA) scores derived from High-vs-Control rat log2 fold-changes mapped to human orthologs; and (ii) **rat toxicology endpoints** — a curated pathology/chemistry rule ("OR-bit": fatty change OR moderate-severe pathology OR relative-liver-weight increase). Models were evaluated by strictly **nested leave-one-drug-out cross-validation (LOOCV)**. We then rebuilt the entire feature pipeline independently, validated it against the frozen reference matrix, and applied the model to an external cohort from **DrugMatrix** (GEO GSE57815) — a different laboratory, platform batch, and drug set — reconstructing both feature arms from primary DrugMatrix expression and measured liver endpoints.

**Results.** The fused model (GSVA top-10 + OR-bit) reached **AUROC 0.790** (nested LOOCV; specificity 0.571 and MCC +0.462 at 95% sensitivity). A feature ablation showed the curated rat OR-bit was the single most predictive feature (AUROC 0.780), expression second (0.68–0.75), and chemistry weakest (0.52); only the parsimonious expression+OR fusion improved on either arm alone, while richer feature combinations overfit the seven negatives. An MCC-optimised two-feature fusion (expression signature + rat-panel probability) reached nested MCC +0.344 by exploiting complementary errors of the two arms. On the external **DrugMatrix** cohort (57 drugs, 47 toxic / 10 safe), the frozen expression arm **transferred without retraining** (GSVA signature AUROC 0.748 internal → 0.721 external, a −0.027 drop across lab/platform), even **out-predicting a model trained natively on DrugMatrix** (within-dataset LOOCV 0.632); whereas the frozen rat-endpoint arm **collapsed to chance** (0.47) despite the rat signal being genuinely present (DrugMatrix-native ALT → DILI AUROC 0.74) — a domain shift attributable to differing endpoint base rates (positive rate 0.68 vs 0.51). **Recalibrating the rat arm on DrugMatrix** and re-fusing with the frozen expression arm recovered **AUROC 0.791 / MCC +0.478**, matching internal performance on independent data.

**Conclusions.** Cross-species rat→human DILI signal replicates on an independent database. Pathway-level expression features are **portable across platforms**; measured rat-endpoint features are **more powerful but dataset-bound**, requiring per-dataset recalibration. We recommend deploying the expression arm as-is and refitting endpoint thresholds on each target dataset.

---

## 1. Introduction

DILI is idiosyncratic, poorly predicted by standard preclinical assays, and a major reason drugs fail. Toxicogenomics — measuring transcriptomic responses to compounds in animal tissue — offers a mechanistic complement to apical toxicity endpoints. Two large public rat-liver resources dominate the field: **Open TG-GATEs** (NIBIOHN/NIHS) and **DrugMatrix** (Iconix, now hosted by the U.S. NTP), each pairing microarray expression with histopathology and clinical chemistry across hundreds of compounds.

A recurrent question is whether a rat toxicogenomic model predicts *human* clinical DILI, and whether such a model generalizes beyond the single resource it was trained on. Prior cross-species work often reports optimistic within-dataset cross-validation without an external test, and rarely disentangles which modality (expression vs measured phenotype vs chemistry) drives — and transfers — the signal.

Here we (1) build a leakage-safe rat→human DILI predictor on TG-GATEs against high-confidence human labels; (2) dissect the contribution and interaction of expression, rat endpoints, and chemistry; and (3) perform a genuine external validation on DrugMatrix, rebuilding both feature arms from primary DrugMatrix data. Our central finding concerns *differential transferability*: expression is portable, measured endpoints are potent but dataset-specific.

## 2. Methods

### 2.1 Human DILI labels

High-confidence binary labels were derived by independent-Bayes fusion of **seven bulk human-clinical sources** (DILIrank, DILIst, LiverTox, SIDER hepatic ADRs, Greene *humans* column, T3DB-clinical, ATSDR-clinical), excluding a hand-curated "Manual" source. A drug was retained if `n_sources_used ≥ 3` and the fused probability was extreme (`p_combined ≥ 0.80` or `≤ 0.20`). This yielded **101 drugs (94 toxic / 7 safe)** with both TG-GATEs features and labels. The class imbalance (7 negatives) is intrinsic: the high-confidence filter is conservative and negatives are scarce among well-studied drugs.

### 2.2 Expression features (GSVA Hallmark)

From Open TG-GATEs rat Affymetrix liver expression (RMA, BrainArray ENTREZG): per drug we computed **log2 fold-change** as the High-dose treated-group mean minus the time-matched Control mean (falling back to Middle/Low dose when High was unavailable). Rat Entrez IDs were mapped to human gene symbols via orthology (mygene); multi-probe symbols were averaged. Sample-wise **GSVA** (gseapy, Gaussian kernel CDF) was scored against the 50 **MSigDB Hallmark** gene sets, giving a 101 × 50 pathway-enrichment matrix.

To confirm faithfulness of this independently reimplemented pipeline, we correlated it against the project's frozen reference GSVA matrix: **per-drug median r = 0.949, per-pathway median r = 0.970** (overall Pearson 0.947). The nested champion computed on the rebuilt matrix reproduced the reference AUROC (0.795 vs 0.790).

### 2.3 Rat endpoint features

Open TG-GATEs pathology and clinical-chemistry endpoints at High dose were encoded as binary flags. The curated **OR-bit** rule combined three: `path_vacuol_fatty_high OR path_mod_severe_high OR chem_LWrel_1.2x_high`. A broader 14-endpoint panel (necrosis, hypertrophy, infiltration, ALT/AST 2×/3×, relative liver weight, composites) was also used.

### 2.4 Chemistry features

RDKit 2-D molecular descriptors (~200) were computed from curated SMILES for the same drugs, used as a comparison modality.

### 2.5 Models and validation

The reference **champion** is `score = z(GSVA_topk10) + λ·OR`, fit by **nested LOOCV**: within each training fold the top-10 Hallmark pathways were selected by |AUC − 0.5|, sign-corrected, z-scored and averaged into an expression signature; λ was chosen on the training fold to maximise training AUROC. The held-out drug never entered pathway selection, z-statistics, or λ — avoiding the stacking optimism of the two-stage "OOF-then-fuse" variant (which inflated AUROC to 0.845 on the same cohort).

For the feature ablation and MCC-optimised fusion we used uniform learners (top-k z-signature per modality; L2-logistic regression; histogram gradient-boosting), all under nested LOOCV. Operating-point metrics (specificity, MCC) were reported at 95% sensitivity or at an MCC-optimal threshold chosen inside each fold (nested). AUROC 95% confidence intervals were bootstrap (2,000 resamples).

### 2.6 External DrugMatrix pipeline

DrugMatrix-Affymetrix expression (GEO **GSE57815**, RMA/BrainArray) was obtained from the unified mirror. **Vehicle controls** were identified from the GSE57815 series-matrix sample annotations as dose-0 samples, matched to treated samples by **vehicle and time**; per-drug High-vs-control log2FC was computed and passed through the *identical* ortholog→GSVA pipeline (Section 2.2), giving a 200 × 48 DrugMatrix Hallmark matrix.

**Measured** DrugMatrix liver endpoints (alanine/aspartate aminotransferase; hepatocyte lipid-accumulation and necrosis severities) were extracted from the ToxCompl input matrix `probe_clinical_pathology.xlsx` (NTP CEBS), decoded with the `C_H_M annotation.txt` dictionary. We used **measured values only**; the ToxCompl *completed* (imputed) histopathology exhibits a positivity bias (predicted severities never reach zero) and was excluded. Endpoints were mapped to the TG-GATEs 14-endpoint schema (ALT/AST fold-change vs a normal-rat reference baseline; histopathology severity thresholds on the 0–4 scale).

The **external high-confidence cohort** comprised drugs with DrugMatrix GSVA and high-conf labels but **not** present in the TG-GATEs training set (69 drugs, 55 toxic / 14 safe); 57 (47 / 10) additionally had a full measured rat panel. Three regimes were evaluated: (a) **frozen** — TG-GATEs-trained model applied unchanged; (b) **diagnostic** — DrugMatrix-native endpoints vs DILI; (c) **recalibrated** — expression arm frozen, rat arm refit on DrugMatrix's own endpoint scale, fused and evaluated by LOOCV within the external cohort.

## 3. Results

### 3.1 Leakage-safe TG-GATEs model

The nested champion achieved **AUROC 0.790** (95% CI ≈ [0.58, 0.97]); at 95% sensitivity, specificity 0.571 (4/7 safe), precision 0.97, **MCC +0.462**. Expression-only (no OR) gave AUROC 0.746. The two-stage OOF-then-fuse variant reached 0.845 but is optimistic and not reported as the result.

### 3.2 Feature ablation (nested LOOCV, high-conf labels)

| Feature set | AUROC |
|---|---:|
| Rat OR-bit (curated 3 endpoints) | **0.780** |
| Expression (GSVA; champion-tuned) | 0.746 |
| Expression (GSVA; uniform top-k) | 0.681 |
| Chemistry (≈200 RDKit descriptors) | 0.521 |
| Rat panel (14 endpoints, generic top-k) | 0.454 |
| **Expression + OR-bit (champion)** | **0.795** |
| Expression + chemistry / + full panel / all | 0.44–0.51 |

Three points emerge. (i) The **curated rat OR-bit is the single most predictive feature** — the *in vivo* phenotype outperforms transcriptomics on its own. (ii) **Curation matters more than modality**: the same rat endpoints give 0.780 as the 3-bit rule but 0.454 as a generic top-k over 14 endpoints. (iii) **Only the parsimonious expression+OR fusion helps**; adding chemistry, the full panel, or meta-stacking degraded performance, because seven negatives cannot support model complexity.

### 3.3 Where the signal lives (auxiliary analyses)

- Chemistry could **not** reproduce the rat OR call (chem→OR AUROC 0.557 ≈ chance): the phenotype is not deducible from structure.
- Expression **could** predict rat endpoints (expr→OR AUROC 0.827), but an OR imputed from expression was **redundant** with expression for DILI (Expr + imputed-OR = Expr alone), whereas Expr + *measured* OR reached 0.795. The measured phenotype therefore carries DILI-relevant information present in neither structure nor transcriptome.

### 3.4 MCC-optimised fusion

Because MCC is threshold-sensitive and dominated by the seven negatives, we optimised the operating point. The two arms failed on *different* safe drugs (e.g., expression rescued a rat false-positive, and vice-versa; inter-arm correlation ≈ 0.30). A compact two-feature histogram-gradient-boosting model over [expression signature, rat-panel probability] reached **nested MCC +0.344**, exceeding either arm alone (rat +0.297; expression ≈ 0). Concatenating all raw features instead overfit (AUROC 0.19–0.49) — distillation of each modality to one calibrated score before fusion was essential.

### 3.5 External validation on DrugMatrix

The rebuilt DrugMatrix GSVA passed a biological sanity check (α-naphthyl-isothiocyanate, a cholestatic hepatotoxin, showed bile-acid-metabolism as a top down-regulated pathway).

| Regime | Model | AUROC | 95% CI | best MCC |
|---|---|---:|---|---:|
| Frozen | Expression (GSVA sig) | **0.723** | [0.55, 0.88] | +0.353 |
| Frozen | Rat panel | 0.470 | [0.32, 0.61] | +0.173 |
| Frozen | Full GBM (GSVA + rat) | 0.499 | [0.32, 0.69] | +0.081 |
| Diagnostic | DrugMatrix-native ALT → DILI | 0.737 | — | — |
| Diagnostic | DrugMatrix-native OR-rule → DILI | 0.627 | — | — |
| Recalibrated | Expression (frozen) | 0.723 | [0.54, 0.88] | +0.353 |
| Recalibrated | Rat arm (refit on DM scale) | 0.736 | [0.58, 0.87] | +0.408 |
| **Recalibrated** | **Expression(frozen) + rat(recal)** | **0.791** | **[0.64, 0.92]** | **+0.478** |

Three findings: (i) **expression transfers frozen** (0.72, matching internal); (ii) the **rat signal is real and replicates** (DrugMatrix-native ALT → DILI 0.74) but the **frozen** rat panel does not transfer (0.47) — a domain shift, with OR-positive base rate 0.68 in TG-GATEs vs 0.51 in DrugMatrix; (iii) **recalibrating the rat arm recovers champion-level performance on independent data** (0.791 / +0.478), with 10 negatives rather than 7.

### 3.6 The GSVA expression signature is the transferable model

The differential transferability above is carried by a single, parsimonious component: the **GSVA top-10 pathway signature**. Isolating it, the expression signature is the only feature that is simultaneously (a) strong under internal nested LOOCV on TG-GATEs and (b) portable — frozen, without any refitting or exposure to DrugMatrix labels — to an independent database.

**GSVA signature — the transferable model.**

| Metric | TG-GATEs (internal LOOCV, n=101) | DrugMatrix (frozen transfer, n=57) | drop |
|---|---:|---:|---:|
| AUROC | 0.748 (perm-p .013) | 0.721 (perm-p .014) | −0.027 |
| best MCC | +0.462 | +0.361 | −0.10 |

The signature retains ~96% of its internal AUROC across a change of laboratory and platform batch (Affymetrix RG230 DrugMatrix vs TG-GATEs), and both endpoints are significant against a label-permutation null. These values are the expression-arm quantities of §3.1–§3.5 recomputed on the internal-champion training cohort (cf. expression-only 0.746, §3.2; frozen-expression 0.723, §3.5).

**Transfer beats native training.** Under the *same* within-dataset LOOCV protocol applied to each database, the GSVA signature scores 0.748 on TG-GATEs but only **0.632 on DrugMatrix** (n=57) — because DrugMatrix self-LOOCV must train each fold on ~56 drugs with 10 negatives, too few to learn a stable signature. The frozen TG-GATEs signature transferred to DrugMatrix (**0.721**) therefore **exceeds DrugMatrix's own within-dataset LOOCV (0.632)**: a model that never saw a DrugMatrix label predicts DrugMatrix DILI better than one trained on DrugMatrix itself. The cross-species knowledge learned on TG-GATEs is genuinely additive, not merely recoverable from the target resource. (The 48-feature L2-logistic is not a reliable transferable model — it collapses to chance under internal LOOCV, 0.506, owing to the 7-negative imbalance; the low-parameter GSVA signature is the consistent choice across every protocol.)

## 4. Discussion

The internal ranking (rat endpoints > expression) *inverts* on transfer: the potent-but-parochial modality is the measured phenotype, whereas the portable modality is pathway-level expression. This is mechanistically sensible. Hallmark GSVA scores are relative, rank-based enrichments largely invariant to platform and batch, so a model frozen on them generalizes — indeed, the frozen TG-GATEs signature out-predicts a model trained natively on DrugMatrix (0.721 vs 0.632; §3.6), because rank-based pathway features carry a stable, transferable cross-species signal that the smaller target cohort cannot re-learn on its own. Apical endpoints, by contrast, depend on study-specific scoring conventions, dosing, and reference ranges; their marginal distributions differ across laboratories (base rate 0.68 vs 0.51), so a classifier frozen on one resource's thresholds misfires on another's — even though the underlying biology (ALT → DILI) is conserved (0.74).

Practically, this argues for a **hybrid deployment**: use the expression arm as a portable backbone, and refit endpoint thresholds/weights whenever a new dataset's apical measurements are available. The complementarity of the two arms — retained cross-dataset — means the fusion is worth the recalibration cost.

Our auxiliary analyses further localise the DILI signal: it is **not** recoverable from chemistry alone, and it is **not** fully recoverable from transcriptome alone; the measured rat phenotype contributes orthogonal, irreducible information. This cautions against replacing *in vivo* endpoints with purely computational surrogates for this task.

## 5. Limitations

- **Small negative class.** Seven training and 10–14 external safe drugs make operating-point metrics (specificity, MCC) unstable and confidence intervals wide; AUROC (rank-based) is the more reliable summary.
- **Recalibrated external results are LOOCV within a 57-drug cohort**, not a fully held-out third dataset; the expression arm is nonetheless frozen (true transfer), and the point estimate matches internal performance on independent drugs.
- **Champion model selection** was made across candidate models on full-cohort LOOCV, introducing mild optimism; the nested numbers are the conservative reference.
- **Cross-platform batch effects** are absorbed at the pathway level but not eliminated; measured DrugMatrix histopathology coverage is sparser than clinical chemistry.

## 6. Conclusion

A leakage-safe rat→human DILI model fusing MSigDB-Hallmark expression with a curated rat toxicology rule reaches AUROC 0.790 on high-confidence human labels, and — critically — **generalizes to an independent database (DrugMatrix)** at champion-level performance (0.791 / MCC +0.478) once its rat arm is recalibrated. Expression features are portable across platforms; measured endpoint features are more powerful but dataset-bound. The measured *in vivo* phenotype carries DILI-relevant signal absent from both chemistry and transcriptome, underscoring the continued value of apical toxicogenomic endpoints.

## Data and code availability

Code (feature pipelines, nested-CV models, external-validation scripts) and derived artifacts (DrugMatrix GSVA matrix, measured endpoint panel, endpoint dictionary, per-drug scores, metrics) are in this repository, primarily under `scripts/external_validation/`, `data/external/`, and `results/external_validation/`; see `docs/EXTERNAL_VALIDATION_DRUGMATRIX.md`. Primary sources: Open TG-GATEs (NIBIOHN); DrugMatrix GSE57815 (GEO) and the unified rat-liver expression mirror (Hugging Face `ajaygeetakumar/dili-toxicogenomics-expression`); measured DrugMatrix endpoints and dictionary from NTP CEBS (ToxCompl); high-confidence human labels from the `entropy-or-highconf-human-dili` project.

## References (selected)

1. Igarashi Y, et al. *Open TG-GATEs: a large-scale toxicogenomics database.* Nucleic Acids Res. 2015.
2. Ganter B, et al. *Development of a large-scale chemogenomics database (DrugMatrix)...* J Biotechnol. 2005.
3. AbdulHameed MDM, et al. *Characterization of chemically induced liver injuries using gene co-expression modules.* PLoS ONE 2014;9(9):e107230.
4. Combs J, et al. *Completion of the DrugMatrix Toxicogenomics Database using ToxCompl.* bioRxiv 2024.03.26.586669.
5. Hänzelmann S, Castelo R, Guinney J. *GSVA: gene set variation analysis...* BMC Bioinformatics 2013.
6. Liberzon A, et al. *The Molecular Signatures Database Hallmark gene set collection.* Cell Syst. 2015.
7. Chen EY, et al. *Enrichr / gseapy.* (tooling for gene-set scoring.)
8. Thakkar S, et al. *DILIrank / DILIst.* Drug Discov Today / Regul context.
9. Matthews BW. *Comparison of predicted and observed secondary structure (MCC).* Biochim Biophys Acta 1975.

*Prepared with computational assistance from Claude Code (Anthropic). Numerical results are from the analyses recorded in this repository.*
