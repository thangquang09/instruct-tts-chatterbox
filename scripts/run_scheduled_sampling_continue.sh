#!/bin/bash
# ============================================================================
# RESUME TRAINING SCRIPT (Target: 100 Epochs)
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data.txt"
OUTPUT_DIR="./checkpoints/mapper_scheduled_sampling"

# Resume Settings
RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint_epoch30.pt"

# Training settings
EPOCHS=100
BATCH_SIZE=170           
LR="1e-4"               
RECON_WEIGHT=0.5

# Scheduled Sampling settings
WARMUP_EPOCHS=3          
SCHEDULE="cosine"        

# Early stopping
PATIENCE=20              

# Number of GPUs
NUM_GPUS=3
GPU_IDS="1,2,3"

# WandB settings
WANDB_PROJECT="instruct-tts-mapper"
WANDB_RUN_NAME="resume-epoch30-target100"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "============================================================"
echo "RESUMING TRAINING -> TARGET 100 EPOCHS"
echo "From Checkpoint: ${RESUME_CHECKPOINT}"
echo "============================================================"

# Kiểm tra file resume
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found at $RESUME_CHECKPOINT"
    exit 1
fi

# Run with DDP
CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29503 \
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
    --resume ${RESUME_CHECKPOINT} \
    --compile

echo "============================================================"
echo "DONE!"
echo "============================================================"