#!/bin/bash

DATA_ROOT_DIR='/PATH/TO/WSIdata/'  # where are the TCGA features stored?
TYPE_OF_PATH="combine"  # what type of pathways?
MODEL="hump"  # what type of model do you want to train?
FUSION="concat"
DIM1=8
DIM2=16
LRS=(0.00005 0.0001 0.0005 0.001)
DECAYS=(0.00001 0.0001 0.001 0.01)
# List of datasets
STUDIES=("blca")

for STUDY in "${STUDIES[@]}"; do
    CUDA_VISIBLE_DEVICES=5 /mnt/pfs-gv8sxa/tts/dhg/yg/envs/hyper/bin/python -u main.py --study tcga_${STUDY} --task survival --split_dir splits --which_splits "5foldcv" --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir results_${STUDY} --batch_size 1 --lr 0.0001 --opt radam --reg 0.0001 --alpha_surv 0.5 --weighted_sample --max_epochs 30 --encoding_dim 1024 --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 --fusion $FUSION
done
