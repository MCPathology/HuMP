# HuMP

<details open>
<summary>
  <b>Unified Multimodal Computational Pathology with Missing-Modality Robustness via Riemannian Learning</b>, Information Fusion.
  <br><em>Guang Yang, Mingcheng Qu, Donglin Di, Kai Yi, Hongyan Xu, Tonghua Su, Lei Fan*</em></br>
</summary>

```bibtex
@article{yang2026hump,
  title   = {Unified Multimodal Computational Pathology with Missing-Modality Robustness via Riemannian Learning},
  author  = {Yang, Guang and Qu, Mingcheng and Di, Donglin and Yi, Kai and Xu, Hongyan and Su, Tonghua and Fan, Lei},
  journal = {Information Fusion},
  year    = {2026}
}
```
</details>

**Summary:** HuMP is a hyperbolic unified multimodal pathology framework for molecular profiles, whole-slide pathology images, and clinical data. It models cross-scale biomedical evidence as a directional hierarchy in Riemannian space and reuses the learned hierarchy for hierarchy-guided sampling (HGS), enabling robust prediction when one or more modalities are missing. This repository contains code for both survival prediction and MIL classification experiments.


## Repository Layout

```text
HuMP/
|-- Survival/                  # Survival prediction code
|   |-- main.py                # Survival training entry point
|   |-- datasets/              # Survival datasets and split handling
|   |-- datasets_csv/          # Metadata, clinical tables, pathway compositions
|   |-- models/                # HuMP survival model and hyperbolic layers
|   |-- scripts/               # Data preparation, external validation, plotting
|   `-- utils/                 # Training loops, losses, metrics
|-- MIL/                       # MIL classification code
|   |-- train_tcga.py          # Classification training entry point
|   |-- preprocess_gene.py     # Transcriptomic preprocessing for classification
|   |-- abmil.py               # HuMP/HypABMIL classification model
|   |-- hyperpath.py
|   `-- wikg.py
|-- docs/
|   |-- data.md                # Data layout and ignored files
|   |-- survival.md            # Survival usage
|   `-- classification.md      # MIL classification usage
|-- requirements.txt
`-- README.md
```

## Installation

We recommend using Linux with an NVIDIA GPU. The code was developed with Python 3.10 and PyTorch.

```bash
conda create -n hump python=3.10 -y
conda activate hump
pip install -r requirements.txt
```

Packages such as `torch`, `torch-geometric`, and `scikit-survival` can be CUDA/Python-version sensitive. If the generic installation fails, install those packages following their official instructions and then rerun `pip install -r requirements.txt`.

## Downloading Data

Diagnostic WSIs, molecular profiles, and clinical metadata can be obtained from public cancer genomics resources, including:

- [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/)
- [cBioPortal](https://www.cbioportal.org/)
- [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/)

The clinical variables used in HuMP are also derived from the GDC database and aligned at the patient/case level with the corresponding WSI and molecular profiles.

The repository does not track large raw data files or generated features. See `docs/data.md` for the expected data layout.

## Processing Whole-Slide Images

HuMP uses pre-extracted WSI patch features as input. A typical preprocessing workflow follows CLAM-style processing:

1. Segment tissue regions from each WSI.
2. Extract non-overlapping patches from tissue regions.
3. Encode patches with a pretrained pathology encoder, such as ResNet, CONCH, or another foundation model.
4. Save the patch-level features as tensors or HDF5 files for downstream training.

For survival prediction, place WSI features under `Survival/WSIdata/`. For MIL classification, place WSI feature CSVs and feature tensors according to the paths expected by `MIL/train_tcga.py`.

## Molecular and Clinical Features

HuMP supports molecular/omics features and clinical features.

- Survival code uses pathway-composition files under `Survival/datasets_csv/pathway_compositions/`.
- MIL classification code can preprocess transcriptomic count matrices into pathway-level gene tokens with `MIL/preprocess_gene.py`.
- The default classification gene token shape is `331 x 512` per patient when using `combine_comps.csv`.

Generate classification gene features:

```bash
cd MIL
python preprocess_gene.py \
  --dataset all \
  --pathway_comps ../Survival/datasets_csv/pathway_compositions/combine_comps.csv \
  --output_root ../gene_features
```

## Training-Validation Splits

For survival prediction, HuMP uses patient-level cross-validation splits and prevents slides from the same case from being distributed across training and validation sets. The split loading logic is implemented in `Survival/datasets/dataset_survival.py`.

For classification, NSCLC and CPTAC-CCRCC splits are read by `MIL/train_tcga.py` from the corresponding MIL data CSVs. CCRCC classification should use patient-level CSVs to avoid slide-level leakage.

## Running Survival Experiments

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

Inference-time missing-modality evaluation:

```bash
python main.py \
  --task survival \
  --modality hgnn \
  --type_of_path combine \
  --bag_loss nll_surv \
  --test_missing_mode O
```

In survival code, `P`, `O`, and `C` denote pathology WSI, omics, and clinical modalities. Supported missing settings include `P`, `O`, `C`, `PO`, `PC`, and `OC`.

## Running MIL Classification Experiments

Use `MIL/` as the working directory.

NSCLC:

```bash
cd MIL
python train_tcga.py \
  --dataset NSCLC \
  --model hyp_a \
  --fusion_modalities clinical+gene \
  --num_classes 2 \
  --feats_size 512 \
  --table_dim 512 \
  --num_gene_pathways 331 \
  --seed 1
```

CPTAC-CCRCC:

```bash
python train_tcga.py \
  --dataset CCRCC_patient \
  --model hyp_a \
  --fusion_modalities clinical+gene \
  --num_classes 4 \
  --feats_size 512 \
  --table_dim 512 \
  --num_gene_pathways 331 \
  --seed 1
```

Gene-missing evaluation:

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

Clinical-missing evaluation:

```bash
python train_tcga.py \
  --dataset NSCLC \
  --model hyp_a \
  --fusion_modalities clinical+gene \
  --test_missing clinical \
  --num_classes 2 \
  --feats_size 512 \
  --table_dim 512 \
  --num_gene_pathways 331
```

## Main HuMP Components

- `Survival/models/model_HGNN.py`: HuMP survival model (`MHSurv`)
- `Survival/models/layers/lhyperbolic.py`: hyperbolic entailment loss, HGS completion, prototype clustering, and hyperbolic fusion layers
- `MIL/abmil.py`: HuMP classification model (`HypABMIL`)
- `MIL/preprocess_gene.py`: pathway-grouped transcriptomic preprocessing for classification

## License

The code is released for academic research use. Please check dataset-specific terms before downloading or redistributing any public clinical, pathology, or molecular data.

## Citation

If you use this repository, please cite:

```bibtex
@article{yang2026hump,
  title   = {Unified Multimodal Computational Pathology with Missing-Modality Robustness via Riemannian Learning},
  author  = {Yang, Guang and Qu, Mingcheng and Di, Donglin and Yi, Kai and Xu, Hongyan and Su, Tonghua and Fan, Lei},
  journal = {Information Fusion},
  year    = {2026}
}
```
