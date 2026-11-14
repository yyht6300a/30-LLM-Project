#!/bin/bash -l
# End-to-end ablation in one job: start bf16 vLLM, wait until ready, run
# evaluation.run_ablation, stop vLLM. Submit from the repository root:
#
#     sbatch evaluation/run_ablation.slurm
#
# Needs .venv1 (vllm) and .venv-eval (openhands-sdk, mcp, openai, pandas,
# scikit-learn, matplotlib, regex) in the repo root, and reference/Popular-but-Wrong
# with res/ materialized. Model cache: $M3_HF_HOME, default ~/hf-cache (~15 GB).

#SBATCH --job-name=p5-ablation
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=evaluation/output/slurm-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export M3_HF_HOME="${M3_HF_HOME:-$HOME/hf-cache}"
VLLM_LOG="evaluation/output/vllm-${SLURM_JOB_ID}.log"

module load python312
module load cuda/12.8.0

source .venv1/bin/activate
model_runtime/vllm/start_qwen25_7b.sh >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# first run downloads the model, so allow up to 30 minutes
for _ in {1..180}; do
  curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null && break
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready"; tail -n 50 "$VLLM_LOG"; exit 1
  fi
  sleep 10
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null || { echo "Timed out waiting for vLLM"; tail -n 50 "$VLLM_LOG"; exit 1; }
echo "vLLM ready"
deactivate

source .venv-eval/bin/activate
export PYTHONPATH="$PWD" OPENHANDS_SUPPRESS_BANNER=1
python -m evaluation.run_ablation
echo "P5 ablation finished"
