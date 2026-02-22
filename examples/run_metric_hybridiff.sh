#!/bin/bash

# Running HybriDiff metric computation

# Define common variables
export CUDA_VISIBLE_DEVICES=6,7

# Define parameters
GPU_COUNT=2
STRIDE=1
MODEL_N=2

# Define log directory and output file dynamically
LOG_DIR="/Hybrid-Diffusion/logs/metrics"
OUTPUT_FILE="${LOG_DIR}/logging-node-${MODEL_N}-stride-${STRIDE}_metric.log"

# NOTE: This script requires a specific metric computation script
# You may need to create or adapt compute_metric.py for distributed metrics

# Run the command
python examples/compute_metric.py \
    --input_root0 path/to/reference \
    --input_root1 path/to/generated \
    --is_gt \
    --max_length 5000 \
    > "${OUTPUT_FILE}" 2>&1

echo "All runs completed. Logs saved in ${OUTPUT_FILE}."