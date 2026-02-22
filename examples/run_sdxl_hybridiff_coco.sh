#!/bin/bash

# Running HybriDiff COCO generation

# Define common variables
export CUDA_VISIBLE_DEVICES=6,7

# Define parameters
GPU_COUNT=2
STRIDE=1
MODEL_N=2

LOG_DIR="/Hybrid-Diffusion/logs/MS-COCO-sdxl-hybridiff"
OUTPUT_FILE="${LOG_DIR}/logging-coco-${GPU_COUNT}GPU-${MODEL_N}-stride-${STRIDE}.log"


# Run the command

python -m torch.distributed.run --nproc_per_node=${GPU_COUNT} --master_port=29500 --run_path examples/generate_hybridiff_coco.py \
    --model_n ${MODEL_N} --stride ${STRIDE} \
    > "${OUTPUT_FILE}" 2>&1

echo "All runs completed. Logs saved in ${LOG_DIR}."