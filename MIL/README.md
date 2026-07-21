# HuMP MIL Classification Code

This directory contains the MIL classification pipeline. The HuMP classification model is selected with `--model hyp_a`.

Preprocess gene features:

```bash
python preprocess_gene.py \
  --dataset all \
  --pathway_comps ../Survival/datasets_csv/pathway_compositions/combine_comps.csv
```

Run three-modality HuMP:

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

More details are in `../docs/classification.md`.
