#!/bin/bash
# Phase 2 Training - DDP Version (3x A100 40GB)
# Predicted Reconstruction for end-to-end optimization

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1,2,3

# Use torchrun for distributed training
torchrun --nproc_per_node=3 train_mapper_ddp.py \
    --manifest final_data.txt \
    --output_dir ./checkpoints/mapper_phase2 \
    --epochs 100 \
    --batch_size 170 \
    --lr 5e-5 \
    --num_workers 8 \
    --recon_weight 0.5 \
    --use_predicted_recon \
    --resume ./checkpoints/mapper_phase1/best_model.pt \
    --reset_best_cos \
    --early_stopping_patience 15 \
    --wandb_project instruct-tts-mapper \
    --wandb_run_name "phase2-predicted-recon-ddp" \
    --save_every 5 
