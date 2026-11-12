#!/usr/bin/env bash
# Start vLLM serving Qwen2.5-7B-Instruct-AWQ (4-bit) for a memory-limited GPU.
#
# Requirements:
#   P4_ENV   path to a Python 3.12 venv with vllm + project deps installed
#   CC_DIR   optional directory containing a C compiler (gcc) for Triton JIT
#   HF_HOME  optional Hugging Face cache directory (default ~/hf-cache)
#
# Memory-related flags default to values suitable for memory-limited GPUs; override via
# MAX_MODEL_LEN and GPU_MEMORY_UTILIZATION if your device differs.
set -euo pipefail

P4_ENV="${P4_ENV:?set P4_ENV to the venv with vllm installed}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [[ -n "${CC_DIR:-}" ]]; then
  export PATH="$CC_DIR:$PATH"
fi
export CC="${CC:-gcc}"

HF_HOME="${HF_HOME:-$HOME/hf-cache}"
mkdir -p "$HF_HOME"
export HF_HOME
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0

exec "$P4_ENV/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b \
  --quantization awq \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 127.0.0.1 \
  --port 8000
