#!/bin/bash -l


#SBATCH --job-name=m3-e2e
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=m3-e2e-%j.out
#SBATCH --error=m3-e2e-%j.err

set -euo pipefail

: "${M3_PROJECT_ROOT:?Set M3_PROJECT_ROOT to the repository root}"

PROJECT_ROOT="$(
  cd "$M3_PROJECT_ROOT"
  pwd
)"

if [[ ! -f "$PROJECT_ROOT/mcp_server/server.py" ]]; then
  echo "Invalid M3_PROJECT_ROOT: $PROJECT_ROOT"
  echo "Expected: $PROJECT_ROOT/mcp_server/server.py"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

: "${M3_HF_HOME:?Submit with M3_HF_HOME set to a persistent cache directory}"

module load python312
module load cuda/12.8.0

source "$PROJECT_ROOT/.venv1/bin/activate"

VLLM_LOG="$PROJECT_ROOT/logs/vllm-${SLURM_JOB_ID}.log"

echo "node=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "project_root=${PROJECT_ROOT}"
echo "hf_home=${M3_HF_HOME}"

nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv

"$PROJECT_ROOT/model_runtime/vllm/start_qwen25_7b.sh" \
  >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

READY=0

for attempt in {1..120}
do
  if curl \
    --fail \
    --silent \
    "http://127.0.0.1:8000/v1/models" \
    >/dev/null
  then
    READY=1
    echo "vLLM ready after attempt ${attempt}"
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready"
    tail -n 100 "$VLLM_LOG"
    exit 1
  fi

  sleep 5
done

if [[ "$READY" != "1" ]]; then
  echo "Timed out waiting for vLLM"
  tail -n 100 "$VLLM_LOG"
  exit 1
fi

deactivate

source "$PROJECT_ROOT/.venv-m3/bin/activate"

echo "Running live vLLM-to-MCP smoke test"
python -m mcp_server.e2e_smoke_test

echo "Running M3 unit and protocol tests"
pytest -q mcp_server/tests

echo "M3 Slurm acceptance passed"