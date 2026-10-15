#!/bin/bash
# Serves one model with vLLM on a GPU node and runs the harness against it in
# the same job, then shuts the server down.
#
#   sbatch scripts/run_vllm.sbatch configs/qwen3-8b.yaml            # full run
#   sbatch scripts/run_vllm.sbatch configs/qwen3-8b.yaml --limit 1  # smoke test
#   sbatch scripts/run_vllm.sbatch configs/diagnostics/*.yaml --limit 1
#
# Several configs share one server, so they must all name the same model and
# revision. Arguments after the configs are passed to every harness run.
#
#SBATCH --job-name=pb-vllm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIGS=()
while [[ $# -gt 0 && "$1" == *.yaml ]]; do CONFIGS+=("$1"); shift; done
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
# Weights are pre-downloaded on the login node; offline mode guarantees the job
# serves exactly the cached revision instead of resolving anything at run time.
export HF_HUB_OFFLINE=1
# vLLM's helper tools (e.g. ninja) live in its venv's bin, which must be on PATH.
export PATH="$PB_SCRATCH/vllm-venv/bin:$PATH"
# FlashInfer's sampler JIT-compiles CUDA kernels at startup, which needs an nvcc
# matching vLLM's CUDA 13 build; PACE only has CUDA 12.x. Use PyTorch sampling.
export VLLM_USE_FLASHINFER_SAMPLER=0

read -r MODEL REVISION GEN_CONFIG < <(uv run --no-sync python -c "
import sys
from harness.config import load_config
models = {(c.model.hf_id, c.model.revision, c.model.generation_config) for c in map(load_config, sys.argv[1:])}
assert len(models) == 1, f'configs name different models: {models}'
print(*models.pop())" "${CONFIGS[@]}")

PORT=$((8000 + SLURM_JOB_ID % 1000))
BASE_URL="http://127.0.0.1:$PORT/v1"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --seed fixes the engine's sampler state; per-request seeds come from the harness.
"$PB_SCRATCH/vllm-venv/bin/vllm" serve "$MODEL" \
    --revision "$REVISION" \
    --generation-config "$GEN_CONFIG" \
    --host 127.0.0.1 --port "$PORT" \
    --seed 42 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM exited; see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }

for CONFIG in "${CONFIGS[@]}"; do
    uv run --no-sync python -m harness.run "$CONFIG" --base-url "$BASE_URL" "$@"
done
