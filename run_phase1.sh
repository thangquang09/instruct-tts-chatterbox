#!/bin/bash
# ============================================================
# Phase 1: Train Flow Matching Backbone with GT Reconstruction
# ============================================================
# In this phase, we train:
# - InstructionEncoder (query + attention pooling)
# - InstructionMapper backbone
# 
# Using GT x_1 for reconstruction loss (stable training for heads)
# Training stops automatically via early stopping when converged.
#
# Output: ./checkpoints/mapper_phase1/best_model.pt
# ============================================================

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1


CUDA_VISIBLE_DEVICES=0 python train_mapper.py \
    --manifest final_data.txt \
    --output_dir ./checkpoints/mapper_phase1 \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-4 \
    --num_workers 8 \
    --compile \
    --recon_weight 0.5 \
    --early_stopping_patience 15 \
    --wandb_project instruct-tts-mapper \
    --wandb_run_name "phase1-gt-recon" \
    --device cuda   

echo "Phase 1 training complete!"
echo "Checkpoint saved at: ./checkpoints/mapper_phase1/best_model.pt"
echo "Now run: bash run_phase2.sh"
