#!/bin/bash
# ============================================================================
# Scheduled Sampling Training for Instruction Mapper
# ============================================================================
# This script runs End-to-End training with Scheduled Sampling.
# Instead of manual 2-phase training, it smoothly transitions from 
# using Ground Truth (100%) to Predicted reconstruction (0%) over epochs.
#
# Key benefits:
# - Single training run (no manual phase switching)
# - Smooth transition avoids distribution shift
# - Supports multiple decay schedules (cosine, linear, exponential)
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data.txt"

OUTPUT_DIR="./checkpoints/mapper_scheduled_sampling"

# Training settings
EPOCHS=30
BATCH_SIZE=170
LR="1e-4"
RECON_WEIGHT=0.5

# Scheduled Sampling settings
WARMUP_EPOCHS=3          # Number of epochs with 100% GT before decay
SCHEDULE="cosine"        # Options: linear, cosine, exponential

# Early stopping
PATIENCE=10

# Number of GPUs
NUM_GPUS=3
GPU_IDS="1,2,3"

# WandB settings (optional)
WANDB_PROJECT="instruct-tts-mapper"
WANDB_RUN_NAME="scheduled-${SCHEDULE}-warmup${WARMUP_EPOCHS}"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "============================================================"
echo "Scheduled Sampling Training"
echo "============================================================"
echo "Schedule: ${SCHEDULE}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "Total Epochs: ${EPOCHS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS} total)"
echo "Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))"
echo "============================================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run with DDP
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train_mapper_ddp_scheduled_sampling.py \
    --manifest ${MANIFEST} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --recon_weight ${RECON_WEIGHT} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --schedule ${SCHEDULE} \
    --early_stopping_patience ${PATIENCE} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --compile

echo "============================================================"
echo "Training Complete!"
echo "Best model saved to: ${OUTPUT_DIR}/best_model.pt"
echo "============================================================"
