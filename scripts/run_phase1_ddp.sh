#!/bin/bash
# Phase 1 Training - DDP Version (3x A100 40GB)
# GT Reconstruction for stable head training

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2

# Use torchrun for distributed training
torchrun --nproc_per_node=3 train_mapper_ddp.py \
    --manifest final_data.txt \
    --output_dir ./checkpoints/mapper_phase1 \
    --epochs 100 \
    --batch_size 170 \
    --lr 1e-4 \
    --num_workers 8 \
    --recon_weight 0.5 \
    --early_stopping_patience 10 \
    --wandb_project instruct-tts-mapper \
    --wandb_run_name "phase1-gt-recon-ddp" \
    --save_every 5 \
    2>&1 | tee phase1_ddp.log
