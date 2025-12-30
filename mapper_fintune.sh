#!/bin/bash

# Activate environment if needed
# source .venv/bin/activate

# Optimize settings
# Set to 1 because we use num_workers=8. Using more threads per worker causes CPU contention/thrashing.
export OMP_NUM_THREADS=1

# Run training
CUDA_VISIBLE_DEVICES=0 uv run train_mapper.py \
    --manifest captts_sft_expresso.txt \
    --output_dir ./checkpoints/mapper \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_workers 8 \
    --compile \
    --device cuda