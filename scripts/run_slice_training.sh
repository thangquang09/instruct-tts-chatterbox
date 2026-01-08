#!/bin/bash
set -e

# CONFIGURATION

TRAIN_CACHE_DIR="./cache/t3_train"
VAL_CACHE_DIR="./cache/t3_val"
OUTPUT_DIR="./checkpoints/mapper_flow"
SCRIPT_DIR="train_mapper_slice.py"

# Training settings
EPOCHS=100
BATCH_SIZE=512           # Per GPU
LR="1e-4"

# Early stopping
PATIENCE=15

# Number of GPUs
NUM_GPUS=4
GPU_IDS="0,1,2,3"

# WandB settings
WANDB_PROJECT="instruct-tts-mapper"
WANDB_RUN_NAME="slice-cached-v4"

echo "source code"
cat ${SCRIPT_DIR}

# RUN TRAINING
echo "============================================================"
echo "FLOW MATCHING TRAINING"
echo "Train cache: ${TRAIN_CACHE_DIR}"
echo "Val cache:   ${VAL_CACHE_DIR}"
echo "Output:      ${OUTPUT_DIR}"
echo "============================================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run with DDP
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29504 \
    ${SCRIPT_DIR} \
    --train_cache_dir ${TRAIN_CACHE_DIR} \
    --val_cache_dir ${VAL_CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --early_stopping_patience ${PATIENCE} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --compile

echo "============================================================"
echo "DONE!"
echo "============================================================"
