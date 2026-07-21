# MIL Classification

This folder contains the MIL classification downstream code for HuMP/HypABMIL.

## Main Files

- `train_tcga.py`: training and evaluation entry point
- `abmil.py`: HuMP/HypABMIL classification model
- `preprocess_gene.py`: pathway-grouped transcriptomic preprocessing

The public HuMP model path is `--model hyp_a`.

## Gene Feature Preprocessing

Use the pathway grouping from the survival code:

```bash
cd MIL
python preprocess_gene.py \
  --dataset all \
  --pathway_comps ../Survival/datasets_csv/pathway_compositions/combine_comps.csv \
  --output_root ../gene_features
```

This writes per-patient features under:

```text
../gene_features/NSCLC/samples/
../gene_features/CPTAC-CCRCC/samples/
```

Each patient is represented as pathway-level gene tokens with default shape:

```text
331 x 512
```

## Three-Modality HuMP

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

CPTAC-CCRCC patient-level setting:

```bash
cd MIL
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

## Missing-Modality Evaluation

Clinical missing:

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

Gene/molecular missing:

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

`train_tcga.py` skips genuinely missing gene files during training, while inference-time `--test_missing gene` evaluates HGS completion from the observed WSI and clinical modalities.
