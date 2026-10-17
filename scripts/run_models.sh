#!/bin/bash
# Queues a chain of Slurm jobs that runs several models one after another:
# for each model, a CPU job downloads the weights, then a GPU job runs the
# paper-faithful config and its configs/fixed/ counterpart against one vLLM
# server and, if everything succeeded, deletes the weights. Each model waits
# for the previous one, so only one model's weights are on scratch at a time.
#
#   bash scripts/run_models.sh configs/qwen3-8b.yaml configs/kanana-1.5-8b.yaml
#
# The script only submits jobs and exits, so nothing has to keep running on
# the login node. Arguments after the configs (e.g. --limit 1) are passed to
# every harness run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CONFIGS=()
while [[ $# -gt 0 && "$1" == *.yaml ]]; do CONFIGS+=("$1"); shift; done

PREVIOUS=""
for CONFIG in "${CONFIGS[@]}"; do
    RUN=("$CONFIG")
    FIXED="configs/fixed/$(basename "$CONFIG")"
    [[ -f "$FIXED" ]] && RUN+=("$FIXED")

    # afterany: one model failing shouldn't stop the rest of the chain.
    DEPEND=()
    [[ -n "$PREVIOUS" ]] && DEPEND=(--dependency="afterany:$PREVIOUS")

    DOWNLOAD=$(sbatch --parsable "${DEPEND[@]}" scripts/download_model.sbatch "$CONFIG")
    PREVIOUS=$(sbatch --parsable --dependency="afterok:$DOWNLOAD" --kill-on-invalid-dep=yes \
        --export=ALL,PB_DELETE_WEIGHTS=1 scripts/run_vllm.sbatch "${RUN[@]}" "$@")
    echo "$CONFIG: download job $DOWNLOAD -> GPU job $PREVIOUS (${RUN[*]})"
done
