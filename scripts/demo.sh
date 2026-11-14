#!/bin/bash -l
# End-to-end demo (docs/03-development-guide.md, M5 step 4): the OpenHands agent answers a
# question through calibrated_qa -> MCP calibrate -> answer, or an in-tool knowledge-base lookup.
# Run on a TC2 GPU node from an interactive shell and record the terminal:
#
#   srun --partition=MGPU-TC2 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash -l
#   bash -l scripts/demo.sh
#
# Needs the environments from scripts/reproduce.sh (.venv1, .venv-eval). Same bf16 server as
# model_runtime/vllm/start_qwen25_7b.sh, plus the tool-calling flags the agent needs.
set -euo pipefail
cd "$(dirname "$0")/.."

module load python312 cuda/12.8.0
export HF_HOME="${M3_HF_HOME:-$HOME/hf-cache}"
mkdir -p logs
VLLM_LOG="logs/vllm-demo-$(date +%Y%m%d-%H%M%S).log"

.venv1/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --generation-config vllm --seed 42 \
  --enable-auto-tool-choice --tool-call-parser hermes >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "starting vLLM (log: $VLLM_LOG) ..."
until curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited"; tail -n 50 "$VLLM_LOG"; exit 1; }
  sleep 5
done

source .venv-eval/bin/activate
export PYTHONPATH="$PWD" OPENHANDS_SUPPRESS_BANNER=1
# Ten P2 test questions (scripts/demo_questions.txt), one conversation each.
mapfile -t QUESTIONS < scripts/demo_questions.txt
python agent_integration/run_calibrated_agent.py "${QUESTIONS[@]}"
