#!/bin/bash
# ============================================================
# Phase 2: End-to-End Fine-tuning with Predicted Reconstruction
# ============================================================
# In this phase, we:
# - Load Phase 1 checkpoint
# - Continue training with predicted x_1 for reconstruction
# - Use lower LR for fine-tuning
# 
# This tightens the coupling between backbone and output heads.
# Training stops via early stopping when converged.
#
# Input:  ./checkpoints/mapper_phase1/best_model.pt (from Phase 1)
# Output: ./checkpoints/mapper_phase2/best_model.pt
# ============================================================

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python train_mapper.py \
    --manifest final_data.txt \
    --output_dir ./checkpoints/mapper_phase2 \
    --epochs 100 \
    --batch_size 512 \
    --lr 5e-5 \
    --num_workers 8 \
    --compile \
    --recon_weight 0.5 \
    --use_predicted_recon \
    --resume ./checkpoints/mapper_phase1/best_model.pt \
    --early_stopping_patience 15 \
    --wandb_project instruct-tts-mapper \
    --wandb_run_name "phase2-predicted-recon" \
    --device cuda

echo "Phase 2 training complete!"
echo "Final checkpoint saved at: ./checkpoints/mapper_phase2/best_model.pt"
echo "This checkpoint is ready for Stage 2 (T3 fine-tuning)"
