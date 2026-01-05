#!/bin/bash
# ==============================================================================
# T3 Finetuning Script for Instruct-TTS
# ==============================================================================
# Purpose: Finetune T3 model with 100% Instruction Mode
# 
# Prerequisites:
#   1. InstructionMapper checkpoint (mapper.pt) from Phase 1 training
#   2. Training data file (final_data.txt) with format: path|text|instruction
#   3. Validation data file (final_data_val.txt) with same format
#   4. Pretrained Chatterbox model directory
#
# Usage:
#   bash scripts/t3_finetune.sh
# ==============================================================================

set -e

MOCK_MODE=false
MAX_STEPS=-1

if [[ "$1" == "--mock" ]]; then
    MOCK_MODE=true
    MAX_STEPS=10
fi

# =================== Configuration ===================

# Data paths
TRAIN_MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data.txt"
VAL_MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data_val.txt"

# Model paths
MODEL_NAME="ResembleAI/chatterbox"
MAPPER_CKPT="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/checkpoints/mapper_slice_v2/best_model.pt"

# Output
OUTPUT_DIR="./checkpoints/t3_instruct"

# Training settings (Optimized for A100 40GB)
NUM_EPOCHS=5
BATCH_SIZE=4                 
LEARNING_RATE=1e-4
WARMUP_RATIO=0.1
GRAD_ACCUM=4                    
MAX_GRAD_NORM=1.0

# Mixed Precision (A100 supports bf16 natively)
USE_BF16=true
USE_TF32=true

# Instruction Mode (1.0 = 100% instruction, no audio reference)
INSTRUCTION_DROPOUT_PROB=1.0

# Logging
LOGGING_STEPS=50
SAVE_TOTAL_LIMIT=3

# Cache settings (for faster data loading - run scripts/preprocess_training_cache.sh first)
# Set USE_CACHE=true after running preprocessing
USE_CACHE=false
TRAIN_CACHE_DIR="./cache/t3_train"
VAL_CACHE_DIR="./cache/t3_val"

# =================== Validation ===================

echo "=============================================="
echo "  T3 Finetuning for Instruct-TTS"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Train data: $TRAIN_MANIFEST"
echo "  - Val data: $VAL_MANIFEST"
echo "  - Model: $MODEL_NAME (from HuggingFace)"
echo "  - Mapper: $MAPPER_CKPT"
echo "  - Output: $OUTPUT_DIR"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Batch size: $BATCH_SIZE (x$GRAD_ACCUM grad accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective)"
echo "  - LR: $LEARNING_RATE"
echo "  - Instruction Dropout: $INSTRUCTION_DROPOUT_PROB (1.0 = 100% instruction mode)"
echo "  - Mixed Precision: bf16=$USE_BF16, tf32=$USE_TF32"
echo ""

if [ "$MOCK_MODE" = true ]; then
    echo "⚠️  MODE: MOCK / DEBUGGING"
    echo "⚠️  Max Steps set to: $MAX_STEPS"
    OUTPUT_DIR="${OUTPUT_DIR}_mock"
else
    echo "🚀 MODE: FULL TRAINING"
    echo "🚀 Max Steps: Unlimited (Based on $NUM_EPOCHS Epochs)"
fi

# Check required files exist
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "ERROR: Training manifest not found: $TRAIN_MANIFEST"
    exit 1
fi

if [ ! -f "$VAL_MANIFEST" ]; then
    echo "ERROR: Validation manifest not found: $VAL_MANIFEST"
    exit 1
fi

if [ ! -f "$MAPPER_CKPT" ]; then
    echo "ERROR: Mapper checkpoint not found: $MAPPER_CKPT"
    exit 1
fi


# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log counts
TRAIN_COUNT=$(wc -l < "$TRAIN_MANIFEST")
VAL_COUNT=$(wc -l < "$VAL_MANIFEST")
echo "Dataset sizes:"
echo "  - Train: $TRAIN_COUNT samples"
echo "  - Val: $VAL_COUNT samples"
echo ""
echo "Starting training..."
echo "=============================================="

# =================== Launch Training ===================

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=29999 \
    src/finetune_t3.py \
    --metadata_file "$TRAIN_MANIFEST" \
    --eval_metadata_file "$VAL_MANIFEST" \
    --model_name_or_path "$MODEL_NAME" \
    --mapper_ckpt_path "$MAPPER_CKPT" \
    --instruction_dropout_prob $INSTRUCTION_DROPOUT_PROB \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --logging_steps $LOGGING_STEPS \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --dataloader_num_workers 12 \
    --dataloader_pin_memory \
    --ignore_verifications \
    --max_steps $MAX_STEPS \
    --remove_unused_columns false \
    --save_safetensors false \
    --report_to tensorboard \
    --bf16 \
    --tf32 True \
    --use_cache $USE_CACHE \
    --cache_dir "$TRAIN_CACHE_DIR" \
    --eval_cache_dir "$VAL_CACHE_DIR"

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo "Output saved to: $OUTPUT_DIR"
echo "  - t3_cfg.safetensors: Finetuned T3 weights"
echo "  - mapper.pt: Copied mapper checkpoint"
echo ""
