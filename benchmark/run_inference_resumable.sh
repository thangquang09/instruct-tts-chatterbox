#!/bin/bash
# ==============================================================================
# Resumable Benchmark Inference - Flexible GPU Configuration
# ==============================================================================
# This script runs inference on specified GPUs in parallel, with resumable support.
# Each GPU will run 3 processes (parts) automatically.
# If crashed, just re-run this script and it will continue from where it stopped.
#
# Usage:
#   bash benchmark/run_inference_resumable.sh                      # Default: GPUs 0,1 
#   bash benchmark/run_inference_resumable.sh 0,1,2                # Use GPUs 0, 1, 2
#   bash benchmark/run_inference_resumable.sh 0,1 my_output_dir    # Specify output dir
#   bash benchmark/run_inference_resumable.sh 0,1 my_output checkpoints/my_ckpt  # Full config
#
# Examples:
#   bash benchmark/run_inference_resumable.sh 0,1      # 2 GPUs x 3 parts = 6 parallel jobs  
#   bash benchmark/run_inference_resumable.sh 0,1,2,3  # 4 GPUs x 3 parts = 12 parallel jobs
# ==============================================================================

set -e
cd /data1/speech/nhandt23/06_thang/instruct-tts-chatterbox

# ===================== CONFIGURATION =====================
# Parse GPU list from first argument (default: "0,1")
GPU_LIST="${1:-0,1}"
OUTPUT_SUBDIR="${2:-2query_t3_freeze}"
T3_CKPT_DIR="${3:-checkpoints/t3_instruct_ddp_2query}"
PARTS_PER_GPU=4

# Convert comma-separated GPU list to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
TOTAL_PARTS=$((NUM_GPUS * PARTS_PER_GPU))

# Create logs directory
mkdir -p benchmark/logs

echo "=============================================="
echo "  Resumable Benchmark Inference"
echo "  ${NUM_GPUS} GPUs x ${PARTS_PER_GPU} processes = ${TOTAL_PARTS} parts"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - GPUs:          ${GPU_LIST} (${NUM_GPUS} GPUs)"
echo "  - Output subdir: benchmark/output_wavs/${OUTPUT_SUBDIR}/"
echo "  - T3 checkpoint: ${T3_CKPT_DIR}"
echo "  - Test data:     data/full_test.txt"
echo "  - Parts per GPU: ${PARTS_PER_GPU}"
echo "  - Total parts:   ${TOTAL_PARTS}"
echo ""

# Check if test data exists
if [ ! -f "data/full_test.txt" ]; then
    echo "ERROR: data/full_test.txt not found!"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "${T3_CKPT_DIR}" ]; then
    echo "ERROR: T3 checkpoint directory not found: ${T3_CKPT_DIR}"
    exit 1
fi

# Create output directory if not exists
mkdir -p "benchmark/output_wavs/${OUTPUT_SUBDIR}"

# Show current progress (TXT file has no header, so count all lines)
TOTAL=$(wc -l < data/full_test.txt)
DONE=$(ls -1 benchmark/output_wavs/${OUTPUT_SUBDIR}/*.wav 2>/dev/null | wc -l || echo "0")
echo "Progress: ${DONE} / ${TOTAL} samples processed"
echo ""

echo "Starting ${TOTAL_PARTS} parallel inference jobs (${PARTS_PER_GPU} per GPU)..."
echo "Logs will be saved to benchmark/logs/"
echo ""

# Array to store PIDs for each GPU
declare -a ALL_PIDS

# Launch processes for each GPU
PART_INDEX=0
for GPU_ID in "${GPUS[@]}"; do
    echo "=================== GPU ${GPU_ID}: Parts ${PART_INDEX}, $((PART_INDEX+1)), $((PART_INDEX+2)) ==================="
    
    for ((i=0; i<PARTS_PER_GPU; i++)); do
        CURRENT_PART=$((PART_INDEX + i))
        
        nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU_ID} python benchmark/inference_resumable.py \
            --part ${CURRENT_PART} \
            --total_parts ${TOTAL_PARTS} \
            --output_subdir ${OUTPUT_SUBDIR} \
            --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part${CURRENT_PART}.log 2>&1 &
        
        PID=$!
        ALL_PIDS+=($PID)
        echo "  Part ${CURRENT_PART}: PID=${PID}"
    done
    
    PART_INDEX=$((PART_INDEX + PARTS_PER_GPU))
    echo ""
done

echo "=============================================="
echo "All ${TOTAL_PARTS} jobs started!"
echo "=============================================="
echo ""
echo "PIDs by GPU:"
PART_INDEX=0
for GPU_ID in "${GPUS[@]}"; do
    echo "  GPU ${GPU_ID}: Part $PART_INDEX=${ALL_PIDS[$PART_INDEX]}, Part $((PART_INDEX+1))=${ALL_PIDS[$((PART_INDEX+1))]}, Part $((PART_INDEX+2))=${ALL_PIDS[$((PART_INDEX+2))]}"
    PART_INDEX=$((PART_INDEX + PARTS_PER_GPU))
done
echo ""
echo "Monitor logs:"
echo "  tail -f benchmark/logs/part*.log"
echo ""
echo "Check progress:"
echo "  ls -1 benchmark/output_wavs/${OUTPUT_SUBDIR}/*.wav 2>/dev/null | wc -l"
echo ""
echo "Check running processes:"
echo "  ps aux | grep inference_resumable.py | grep -v grep"
echo ""
echo "Kill all jobs (if needed):"
echo "  pkill -f 'inference_resumable.py'"