# Survival Prediction

This folder contains the survival downstream code for HuMP.

## Main Files

- `main.py`: experiment entry point
- `models/model_HGNN.py`: HuMP survival model (`MHSurv`)
- `models/layers/lhyperbolic.py`: hyperbolic entailment, HGS, prototype clustering, and fusion layers
- `datasets/dataset_survival.py`: survival dataset construction and split handling
- `utils/core_utils.py`: training, validation, metrics, and missing-modality dispatch
- `utils/loss_func.py`: discrete-time survival loss

## Run

Use `Survival/` as the working directory:

```bash
cd Survival
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --max_epochs 20 \
  --seed 1
```

The default `main.py` loops over:

```text
coadread, brca, hnsc, stad, blca
```

Edit the list at the bottom of `main.py` or adapt the script if you want to run one cohort only.

## Missing-Modality Evaluation

Use `--test_missing_mode` for inference-time missingness:

```bash
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --test_missing_mode C
```

Supported values:

```text
''    full modality
P     pathology missing
O     omics missing
C     clinical missing
PO    pathology + omics missing
PC    pathology + clinical missing
OC    omics + clinical missing
```

During inference, HuMP uses HGS completion for missing modalities. During training, missing-modality simulation can skip the missing modality rather than imputing it:

```bash
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --train_missing_o_prob 0.2 \
  --train_skip_imputation
```

## Outputs

Results are written to `results_{cohort}/`, including per-fold pickle files and `summary_kfold.csv`.
