#!/bin/bash
# Downloads one model's weights into the scratch HF cache on a CPU node.
# Login nodes kill long-running downloads, so this runs as a job instead.
#
#   sbatch scripts/download_model.sbatch configs/qwen3-8b.yaml
#
#SBATCH --job-name=pb-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh

read -r MODEL REVISION < <(uv run --no-sync python -c "
from harness.config import load_config
c = load_config('$1'); print(c.model.hf_id, c.model.revision)")

# Plain HTTP instead of xet's parallel chunk downloader, which got killed for
# memory on shared nodes; a couple of workers is plenty for a 16 GB model.
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 "$PB_SCRATCH/vllm-venv/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', revision='$REVISION', max_workers=2))"
du -shL "$HF_HOME/hub/models--${MODEL//\//--}"
