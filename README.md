# HuMP

HuMP is a hyperbolic multimodal computational pathology framework for pathology WSI, molecular/omics, and clinical features. This repository contains the code used for two downstream settings:

- survival prediction: `Survival/`
- MIL classification: `MIL/`

The two folders are kept as separate runnable entry points because each downstream task has its own data layout and evaluation protocol. Shared hyperbolic components used by HuMP are included under `Survival/models/`.

## Repository Layout

```text
HuMP/
├── Survival/                   # Survival prediction pipeline
│   ├── main.py                 # Survival training entry point
│   ├── datasets/               # Survival datasets and split handling
│   ├── datasets_csv/           # Metadata, clinical tables, pathway compositions
│   ├── models/                 # HuMP survival model and hyperbolic layers
│   ├── scripts/                # Data preparation, external validation, plotting
│   └── utils/                  # Training loops, losses, metrics
├── MIL/                        # MIL classification pipeline
│   ├── train_tcga.py           # Classification training entry point
│   ├── preprocess_gene.py      # Transcriptomic preprocessing for classification
│   ├── abmil.py                # HuMP/HypABMIL classification model
│   ├── hyperpath.py            # Additional local comparison implementation
│   └── wikg.py                 # Additional local comparison implementation
├── docs/
│   ├── data.md                 # Data layout and files excluded from Git
│   ├── survival.md             # Survival usage
│   └── classification.md       # MIL classification usage
├── requirements.txt
└── README.md
```

## Installation

Create a Python environment with PyTorch and install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Some packages, such as `torch`, `torch-geometric`, and `scikit-survival`, may require installation commands matched to your CUDA/Python version. Install those following their official instructions if the generic `pip install` fails.

## Data

Large WSI features, raw transcriptomic matrices, generated gene features, checkpoints, and experiment outputs are intentionally excluded by `.gitignore`. See `docs/data.md` for the expected layout.

## Survival Prediction

Use `Survival/` as the working directory:

```bash
cd Survival
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --max_epochs 20
```

For inference-time missing-modality evaluation:

```bash
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --test_missing_mode O
```

`P`, `O`, and `C` denote pathology WSI, omics, and clinical modalities. See `docs/survival.md` for more examples.

## MIL Classification

Use `MIL/` as the working directory. First preprocess transcriptomic features when needed:

```bash
cd MIL
python preprocess_gene.py \
  --dataset all \
  --pathway_comps ../Survival/datasets_csv/pathway_compositions/combine_comps.csv
```

Train HuMP/HypABMIL with pathology, clinical, and gene modalities:

```bash
python train_tcga.py \
  --dataset NSCLC \
  --model hyp_a \
  --fusion_modalities clinical+gene \
  --num_classes 2 \
  --feats_size 512 \
  --table_dim 512 \
  --num_gene_pathways 331
```

For gene-missing or clinical-missing evaluation:

```bash
python train_tcga.py \
  --dataset NSCLC \
  --model hyp_a \
  --fusion_modalities clinical+gene \
  --test_missing gene \
  --num_classes 2 \
  --feats_size 512 \
  --table_dim 512 \
  --num_gene_pathways 331
```

See `docs/classification.md` for CCRCC commands and data notes.

## Main HuMP Components

- `Survival/models/model_HGNN.py`: survival HuMP model (`MHSurv`)
- `Survival/models/layers/lhyperbolic.py`: hyperbolic entailment loss, HGS completion, prototype clustering, and hyperbolic fusion layers
- `MIL/abmil.py`: classification HuMP model (`HypABMIL`) with WSI, clinical, and gene branches
- `MIL/preprocess_gene.py`: pathway-grouped transcriptomic preprocessing for classification

## Citation

If you use this code, please cite the HuMP manuscript.
