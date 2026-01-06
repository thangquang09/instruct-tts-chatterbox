#!/bin/bash
# ==============================================================================
# Preprocess and Cache T3 Training Data
# ==============================================================================
# Run this BEFORE t3_finetune.sh to generate cached data.
# Caching speeds up data loading by 10-100x.
#
# Usage:
#   bash scripts/preprocess_training_cache.sh
# ==============================================================================

set -e

# =================== Configuration ===================

# Source metadata files
TRAIN_MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data.txt"
VAL_MANIFEST="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/final_data_val.txt"

# Cache output directories
TRAIN_CACHE_DIR="./cache/t3_train"
VAL_CACHE_DIR="./cache/t3_val"

# GPU for preprocessing (uses GPU for S3 Tokenizer)
export CUDA_VISIBLE_DEVICES=0

# =================== Run Preprocessing ===================

echo "=============================================="
echo "  Preprocessing T3 Training Data"
echo "=============================================="
echo ""

# Create cache directories
mkdir -p "$TRAIN_CACHE_DIR"
mkdir -p "$VAL_CACHE_DIR"

# Process training data
echo "📦 Caching TRAINING data..."
echo "   Source: $TRAIN_MANIFEST"
echo "   Output: $TRAIN_CACHE_DIR"
echo ""

uv run python scripts/preprocess_cache.py \
    --metadata_file "$TRAIN_MANIFEST" \
    --output_dir "$TRAIN_CACHE_DIR" \
    --batch_save_size 1000

echo ""
echo "✅ Training cache complete!"
echo ""

# Process validation data
echo "📦 Caching VALIDATION data..."
echo "   Source: $VAL_MANIFEST"
echo "   Output: $VAL_CACHE_DIR"
echo ""

uv run python scripts/preprocess_cache.py \
    --metadata_file "$VAL_MANIFEST" \
    --output_dir "$VAL_CACHE_DIR" \
    --batch_save_size 1000

echo ""
echo "=============================================="
echo "  Preprocessing Complete!"
echo "=============================================="
echo ""
echo "Cache locations:"
echo "  - Train: $TRAIN_CACHE_DIR"
echo "  - Val:   $VAL_CACHE_DIR"
echo ""
echo "Now run t3_finetune.sh with cache enabled:"
echo "  bash scripts/t3_finetune.sh"
echo ""
