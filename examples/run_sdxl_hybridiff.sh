#!/bin/bash

# Define common variables
export CUDA_VISIBLE_DEVICES=6,7

# Define parameters
GPU_COUNT=2
STRIDE=1
MODEL_N=2

# Define log directory and output file dynamically
LOG_DIR="/Hybrid-Diffusion/logs/sdxl-hybridiff"
OUTPUT_FILE="${LOG_DIR}/logging-node-${MODEL_N}-stride-${STRIDE}-1024.log"

# Run the command
python -m torch.distributed.run --nproc_per_node=${GPU_COUNT} --master_port=29501 --run_path examples/run_sdxl_hybridiff.py \
    --model_n ${MODEL_N} --stride ${STRIDE} \
    --image_size 1024 \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    > "${OUTPUT_FILE}" 2>&1

echo "All runs completed. Logs saved in ${OUTPUT_FILE}."