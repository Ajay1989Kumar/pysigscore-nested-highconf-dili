# External validation pipeline (DrugMatrix)

Working scripts for the DrugMatrix external validation — see
[`docs/EXTERNAL_VALIDATION_DRUGMATRIX.md`](../../docs/EXTERNAL_VALIDATION_DRUGMATRIX.md)
for the study, results, and data sources.

**Note:** these are the research working scripts. Paths are hardcoded to the original
working directory and must be adapted, and the large source files (DrugMatrix RMA,
508 MB measured-endpoint xlsx, GSE57815 series matrix) are not committed — fetch them
from the sources listed in the docs. Requires Python 3.12 with `pysigscore`/`gseapy`,
`mygene`, `openpyxl`, `pandas`, `scikit-learn`.

| Script | Purpose |
|---|---|
| `build_dm_gsva.py` | DrugMatrix log2FC (High − vehicle control) → human orthologs → gseapy GSVA (50 Hallmarks) |
| `extract_measured.py` | Extract measured ALT/AST + liver histopathology from the ToxCompl input matrix |
| `run_gbm_external.py` | Apply frozen TG-GATEs GBM to the external cohort; decompose by arm |
| `recal_rat.py` | Refit the rat arm on DrugMatrix-native endpoints; re-fuse with frozen expression arm (LOOCV) |
| `ext_val_expr.py` | Expression-only external validation |
