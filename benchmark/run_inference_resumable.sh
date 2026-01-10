#!/bin/bash
# ==============================================================================
# Resumable Benchmark Inference - 3 GPUs (0, 1, 4) x 3 processes each = 9 parts
# ==============================================================================
# This script runs inference on 3 GPUs in parallel, with resumable support.
# If crashed, just re-run this script and it will continue from where it stopped.
#
# Usage:
#   bash benchmark/run_inference_resumable.sh
#   bash benchmark/run_inference_resumable.sh 2query_t3_freeze
#   bash benchmark/run_inference_resumable.sh 2query_t3_freeze checkpoints/t3_instruct_ddp_2query
# ==============================================================================

set -e
cd /data1/speech/nhandt23/06_thang/instruct-tts-chatterbox

# Default parameters
OUTPUT_SUBDIR="${1:-2query_t3_freeze}"
T3_CKPT_DIR="${2:-checkpoints/t3_instruct_ddp_2query}"
TOTAL_PARTS=9

# Create logs directory
mkdir -p benchmark/logs

echo "=============================================="
echo "  Resumable Benchmark Inference"
echo "  3 GPUs (0, 1, 4) x 3 processes = 9 parts"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Output subdir: benchmark/output_wavs/${OUTPUT_SUBDIR}/"
echo "  - T3 checkpoint: ${T3_CKPT_DIR}"
echo "  - CSV metadata:  benchmark/test_metadata.csv"
echo "  - Total parts:   ${TOTAL_PARTS}"
echo ""

# Check if CSV exists
if [ ! -f "benchmark/test_metadata.csv" ]; then
    echo "ERROR: benchmark/test_metadata.csv not found!"
    echo "Run: python benchmark/create_test_csv.py"
    exit 1
fi

# Show current progress
TOTAL=$(tail -n +2 benchmark/test_metadata.csv | wc -l)
DONE=$(ls -1 benchmark/output_wavs/${OUTPUT_SUBDIR}/*.wav 2>/dev/null | wc -l || echo "0")
echo "Progress: ${DONE} / ${TOTAL} samples processed"
echo ""

echo "Starting 9 parallel inference jobs (3 per GPU)..."
echo "Logs will be saved to benchmark/logs/"
echo ""

# =================== GPU 0: Parts 0, 1, 2 ===================
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python benchmark/inference_resumable.py --part 0 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part0.log 2>&1 &
PID0=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=2 python benchmark/inference_resumable.py --part 1 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part1.log 2>&1 &
PID1=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=2 python benchmark/inference_resumable.py --part 2 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part2.log 2>&1 &
PID2=$!

# =================== GPU 1: Parts 3, 4, 5 ===================
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python benchmark/inference_resumable.py --part 3 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part3.log 2>&1 &
PID3=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python benchmark/inference_resumable.py --part 4 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part4.log 2>&1 &
PID4=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python benchmark/inference_resumable.py --part 5 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part5.log 2>&1 &
PID5=$!

# =================== GPU 4: Parts 6, 7, 8 ===================
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python benchmark/inference_resumable.py --part 6 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part6.log 2>&1 &
PID6=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=3 python benchmark/inference_resumable.py --part 7 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part7.log 2>&1 &
PID7=$!

nohup bash -c "CUDA_VISIBLE_DEVICES=3 python benchmark/inference_resumable.py --part 8 --total_parts ${TOTAL_PARTS} --output_subdir ${OUTPUT_SUBDIR} --t3_ckpt_dir ${T3_CKPT_DIR}" > benchmark/logs/part8.log 2>&1 &
PID8=$!

echo "Jobs started with PIDs:"
echo "  GPU 0: Part 0=$PID0, Part 1=$PID1, Part 2=$PID2"
echo "  GPU 1: Part 3=$PID3, Part 4=$PID4, Part 5=$PID5"
echo "  GPU 3: Part 6=$PID6, Part 7=$PID7, Part 8=$PID8"
echo ""
echo "Monitor logs:"
echo "  tail -f benchmark/logs/part*.log"
echo ""
echo "Check progress:"
echo "  ls -1 benchmark/output_wavs/${OUTPUT_SUBDIR}/*.wav 2>/dev/null | wc -l"
echo ""
echo "Check running processes:"
echo "  ps aux | grep inference_resumable.py | grep -v grep"
