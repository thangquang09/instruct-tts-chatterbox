#!/bin/bash
# T3 Finetuning Script with Instruction Mapper Integration
# Usage: 
#   bash t3_finetune.sh          # Full training
#   bash t3_finetune.sh --mock   # Quick test (20 steps)

set -e

# ============ Parse Arguments ============
MOCK_MODE=false
for arg in "$@"; do
    case $arg in
        --mock)
            MOCK_MODE=true
            shift
            ;;
    esac
done

# ============ Configuration ============
OUTPUT_DIR="./checkpoints/t3_instruct_v1"
MAPPER_CKPT="./checkpoints/mapper_phase2/best_model.pt"
METADATA_FILE="./captts_sft_expresso.txt"

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_EPOCHS=3
SAVE_STEPS=500
LOGGING_STEPS=10

# Mixing strategy (0.5 = 50% instruction-only, 50% audio-only)
INSTRUCTION_DROPOUT_PROB=0.5

# Mock mode settings
if [ "$MOCK_MODE" = true ]; then
    echo "🧪 MOCK MODE: Running quick test (20 steps)"
    MAX_STEPS_ARG="--max_steps 20"
    LOGGING_STEPS=1
    OUTPUT_DIR="./checkpoints/t3_instruct_mock"
else
    MAX_STEPS_ARG=""
fi

# ============ Run Training ============
echo "Starting T3 Finetuning..."
echo "Output: $OUTPUT_DIR"
echo "Mapper: $MAPPER_CKPT"
echo "Data: $METADATA_FILE"

CUDA_VISIBLE_DEVICES=1 python src/finetune_t3.py \
    --do_train \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "ResembleAI/Chatterbox" \
    --mapper_ckpt_path "$MAPPER_CKPT" \
    --instruction_dropout_prob $INSTRUCTION_DROPOUT_PROB \
    --metadata_file "$METADATA_FILE" \
    --instruction_column_name "caption" \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --freeze_voice_encoder True \
    --freeze_s3gen True \
    --dataloader_num_workers 8 \
    --save_safetensors False \
    --report_to none \
    $MAX_STEPS_ARG

echo "Training completed! Checkpoint saved to: $OUTPUT_DIR"
