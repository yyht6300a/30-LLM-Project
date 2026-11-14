#!/usr/bin/env bash
# One-command reproduction (docs/03-development-guide.md, M5). Linux / TC2, Python >= 3.12
# (on TC2 run `module load python312` first).
#
#   bash scripts/reproduce.sh        tests + CPU results: paper statistics (P1), calibrator
#                                    evaluation and weights (P2), offline coverage-accuracy curve
#   bash scripts/reproduce.sh test   environment, paper data and tests only (the CI gate)
#   bash scripts/reproduce.sh gpu    TC2 login node: submit the end-to-end ablation job
#
# Regenerated P2 outputs go to reproduce_output/, so the committed weights used by the
# end-to-end evaluation are never overwritten; compare them with calibrator/.
set -euo pipefail

STAGE="${1:-cpu}"
case "$STAGE" in test | cpu | gpu) ;; *) echo "usage: $0 [test|cpu|gpu]" >&2; exit 2 ;; esac
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
PAPER_REPO=reference/Popular-but-Wrong
PAPER_COMMIT=1b1f250

"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
  || { echo "Python >= 3.12 required (TC2: module load python312)" >&2; exit 1; }

echo "== CPU environment (.venv-eval)"
[[ -x .venv-eval/bin/python ]] || "$PY" -m venv .venv-eval
source .venv-eval/bin/activate
pip install -q -r scripts/requirements.txt
export PYTHONPATH="$PWD" OPENHANDS_SUPPRESS_BANNER=1

echo "== paper repository at $PAPER_COMMIT"
[[ -d "$PAPER_REPO/.git" ]] \
  || git clone -q https://github.com/Trustworthy-Information-Access/Popular-but-Wrong.git "$PAPER_REPO"
git -C "$PAPER_REPO" checkout -q "$PAPER_COMMIT"
[[ -d "$PAPER_REPO/res" ]] || (cd "$PAPER_REPO" && python scripts/materialize_compact_data.py)

if [[ "$STAGE" == gpu ]]; then
  if [[ ! -x .venv1/bin/vllm ]]; then
    echo "== vLLM environment (.venv1)"
    "$PY" -m venv .venv1
    .venv1/bin/pip install -q -r model_runtime/vllm/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128
  fi
  sbatch evaluation/run_ablation.slurm
  exit 0
fi

echo "== tests"
pytest -q calibrator mcp_server evaluation
if [[ "$STAGE" == test ]]; then exit 0; fi

echo "== P1: paper statistics"
(cd "$PAPER_REPO" && python scripts/reproduce.py --stage core)

echo "== P2: calibrator evaluation and weights -> reproduce_output/"
python -m calibrator.eval --output-dir reproduce_output/calibrator_eval
python -m calibrator.training --artifacts-dir reproduce_output/artifacts

echo "== P5: offline coverage-accuracy curve -> evaluation/output/"
python -m evaluation.coverage_accuracy

echo "CPU reproduction finished. End-to-end ablation on TC2: bash scripts/reproduce.sh gpu"
