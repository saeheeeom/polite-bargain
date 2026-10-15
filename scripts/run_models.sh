#!/bin/bash
# Runs several model configs one after another from the login node: download the
# weights, run the full design in a GPU job, then delete the weights. Scratch
# can't hold every model at once, and weights can always be re-downloaded;
# transcripts under results/runs/ are kept.
#
#   bash scripts/run_models.sh configs/qwen3-8b.yaml configs/kanana-1.5-8b.yaml
#
# Arguments after the configs (e.g. --limit 1) are passed to each harness run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

CONFIGS=()
while [[ $# -gt 0 && "$1" == *.yaml ]]; do CONFIGS+=("$1"); shift; done

for CONFIG in "${CONFIGS[@]}"; do
    read -r MODEL REVISION < <(uv run --no-sync python -c "
from harness.config import load_config
c = load_config('$CONFIG'); print(c.model.hf_id, c.model.revision)" 2>/dev/null)
    echo "=== $CONFIG ($MODEL @ ${REVISION:0:8}) $(date -Is)"

    # xet's parallel downloader gets killed on PACE login nodes; plain HTTP works.
    if ! HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$PB_SCRATCH/vllm-venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', revision='$REVISION', max_workers=2)"; then
        echo "download failed for $MODEL; skipping"; continue
    fi

    sbatch --wait scripts/run_vllm.sbatch "$CONFIG" "$@"
    echo "job exit code: $? $(date -Is)"

    rm -rf "$HF_HOME/hub/models--${MODEL//\//--}"
done
