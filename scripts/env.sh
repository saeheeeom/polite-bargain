# Source this before working on the harness:  source scripts/env.sh
#
# PACE home directories have a 30 GB quota, so everything large (the Python
# environment, uv's download cache, Hugging Face model weights) lives on scratch.
# Scratch is not backed up and old files can be purged; nothing here is
# irreplaceable — it can all be re-downloaded.

export PB_SCRATCH="${PB_SCRATCH:-$HOME/scratch/polite-bargain}"

export UV_CACHE_DIR="$PB_SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PB_SCRATCH/uv-python"
export UV_PROJECT_ENVIRONMENT="$PB_SCRATCH/venv"
export HF_HOME="$PB_SCRATCH/hf"
# vLLM's torch.compile and CUDA-graph caches (hundreds of MB per model).
export VLLM_CACHE_ROOT="$PB_SCRATCH/vllm-cache"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$VLLM_CACHE_ROOT"
