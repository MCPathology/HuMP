# Data Layout

The repository is code-first. Large data files are not tracked by Git.

## Excluded Data

The following file types and folders are ignored:

- WSI feature tensors: `*.pt`, `*.h5`, `*.hdf5`
- raw molecular matrices: `*.gz`, large count tables, probemaps
- generated gene features: `gene_features/`
- experiment outputs: `results*/`, `baseline/`, `deconf/`, `checkpoints/`, `heat_pt/`
- local review/manuscript working files

## Survival Prediction

Run survival experiments from `Survival/`. The expected structure is:

```text
Survival/
├── WSIdata/
│   ├── brca/
│   ├── blca/
│   ├── coadread/
│   ├── hnsc/
│   └── stad/
├── datasets_csv/
│   ├── metadata/tcga_*.csv
│   ├── clinical_data/*_clinical.csv
│   ├── raw_rna_data/combine/{cohort}/rna_clean.csv
│   └── pathway_compositions/combine_comps.csv
└── splits/
```

The default `main.py` loop runs TCGA cohorts listed in the script. WSI feature paths and split CSVs should match the values expected by `datasets/dataset_survival.py`.

## MIL Classification

Run classification experiments from `MIL/`. The expected external layout is:

```text
HuMP/
├── MIL/
├── MILdata/
│   ├── tcga_ImageNet/
│   ├── tcga_conch/
│   ├── CCRCC_r18/
│   └── CCRCC_conch/
└── gene_features/
    ├── NSCLC/samples/{PATIENT_ID}.npy
    └── CPTAC-CCRCC/samples/{PATIENT_ID}.npy
```

`MIL/preprocess_gene.py` generates `gene_features/` from raw transcriptomic matrices and pathway compositions. The default pathway file is:

```text
Survival/datasets_csv/pathway_compositions/combine_comps.csv
```

For public release, keep raw count matrices and generated features outside Git and document how to obtain them.
