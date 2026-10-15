#!/bin/bash
# Serves one model with vLLM on a GPU node and runs the harness against it in
# the same job, then shuts the server down.
#
#   sbatch scripts/run_vllm.sbatch configs/qwen3-8b.yaml            # full run
#   sbatch scripts/run_vllm.sbatch configs/qwen3-8b.yaml --limit 1  # smoke test
#
#SBATCH --job-name=pb-vllm
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
CONFIG="$1"; shift
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh

read -r MODEL REVISION < <(uv run --no-sync python -c "
from harness.config import load_config
c = load_config('$CONFIG'); print(c.model.hf_id, c.model.revision)")

PORT=$((8000 + SLURM_JOB_ID % 1000))
BASE_URL="http://127.0.0.1:$PORT/v1"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --seed fixes the engine's sampler state; per-request seeds come from the harness.
"$PB_SCRATCH/vllm-venv/bin/vllm" serve "$MODEL" \
    --revision "$REVISION" \
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

uv run --no-sync python -m harness.run "$CONFIG" --base-url "$BASE_URL" "$@"
