#!/usr/bin/env bash
set -euo pipefail

: "${M3_HF_HOME:?Set M3_HF_HOME to a persistent project or scratch directory}"

M3_MODEL_ID="${M3_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
M3_SERVED_MODEL_NAME="${M3_SERVED_MODEL_NAME:-qwen2.5-7b}"
M3_VLLM_HOST="${M3_VLLM_HOST:-127.0.0.1}"
M3_VLLM_PORT="${M3_VLLM_PORT:-8000}"
M3_MAX_MODEL_LEN="${M3_MAX_MODEL_LEN:-4096}"
M3_GPU_MEMORY_UTILIZATION="${M3_GPU_MEMORY_UTILIZATION:-0.85}"

export HF_HOME="$M3_HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HUGGINGFACE_HUB_CACHE"

exec vllm serve "$M3_MODEL_ID" \
  --served-model-name "$M3_SERVED_MODEL_NAME" \
  --host "$M3_VLLM_HOST" \
  --port "$M3_VLLM_PORT" \
  --dtype bfloat16 \
  --max-model-len "$M3_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$M3_GPU_MEMORY_UTILIZATION" \
  --generation-config vllm \
  --seed 42