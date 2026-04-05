#!/usr/bin/env bash
# =============================================================================
# setup_env_hpc.sh — Full environment setup for HPC (Linux + A100 + CUDA 12.6)
#
# This is the ONLY setup script needed on HPC. It handles:
#   1. Conda environment creation (Python 3.12)
#   2. PyTorch 2.9.1 + torchvision 0.24.1 (CUDA 12.6 wheels)
#   3. Flash Attention 2.8.3 (precompiled wheel — user must provide path)
#   4. All model dependencies (Qwen2.5-VL, InternVL3, Gemma 3, LLaVA-OneVision)
#   5. AI2-THOR + headless rendering deps
#   6. Analysis & utility packages
#   7. Data directory structure
#
# Usage (from repo root):
#   bash scripts/setup_env_hpc.sh
#   bash scripts/setup_env_hpc.sh --flash-attn /path/to/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
#
# After setup:
#   conda activate viewpoint
# =============================================================================

set -e  # exit on any error

CONDA_ENV="viewpoint"
PYTHON_VERSION="3.12"
FLASH_ATTN_WHL=""

# Parse optional --flash-attn argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --flash-attn)
            FLASH_ATTN_WHL="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo " HPC Environment Setup"
echo " Python $PYTHON_VERSION | CUDA 12.6 | torch 2.9.1"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 1/7: Creating conda environment..."
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "[setup] Active env: $(which python)"
echo "[setup] Python version: $(python --version)"

# ---------------------------------------------------------------------------
# 2. PyTorch + torchvision (CUDA 12.6)
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 2/7: Installing PyTorch 2.9.1 (CUDA 12.6)..."
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# Verify CUDA is visible
python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA ver:   {torch.version.cuda}')
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
"

# ---------------------------------------------------------------------------
# 3. Flash Attention (precompiled wheel for A100)
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 3/7: Flash Attention..."
if [ -n "$FLASH_ATTN_WHL" ]; then
    if [ -f "$FLASH_ATTN_WHL" ]; then
        echo "[setup] Installing flash-attn from: $FLASH_ATTN_WHL"
        pip install "$FLASH_ATTN_WHL"
    else
        echo "[setup] ERROR: Wheel not found at $FLASH_ATTN_WHL"
        echo "[setup] Skipping flash-attn — install manually later."
    fi
else
    echo "[setup] No --flash-attn wheel provided. Skipping."
    echo "[setup] To install later:"
    echo "  pip install /path/to/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
fi

# ---------------------------------------------------------------------------
# 4. Core requirements (transformers, accelerate, ai2thor, analysis, etc.)
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 4/7: Installing core requirements..."
pip install -r env/requirements.txt

# ---------------------------------------------------------------------------
# 5. Model-specific dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 5/7: Installing model-specific packages..."
# Qwen2.5-VL
pip install qwen-vl-utils
# InternVL3
pip install einops timm
# Gemma 3 & LLaVA-OneVision: use native transformers classes, no extra deps

# ---------------------------------------------------------------------------
# 6. HuggingFace login (needed for gated Gemma 3 model)
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 6/7: HuggingFace authentication..."
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "[setup] HF token already exists. Skipping login."
else
    echo "[setup] Gemma 3 requires HuggingFace authentication."
    echo "[setup] Accept the license at: https://huggingface.co/google/gemma-3-12b-it"
    echo "[setup] Then run: huggingface-cli login"
    echo "[setup] (Skipping for now — run it manually before Phase 1)"
fi

# ---------------------------------------------------------------------------
# 7. Data directories + headless display check
# ---------------------------------------------------------------------------
echo ""
echo "[setup] Step 7/7: Creating data directories and checking display..."
mkdir -p data/alfred_episodes
mkdir -p data/rendered_frames
mkdir -p data/logs/raw
mkdir -p data/logs/aggregated
mkdir -p results

# AI2-THOR uses CloudRendering (Vulkan) on Linux — no Xvfb needed.
echo "[setup] AI2-THOR will use CloudRendering (Vulkan) for headless rendering."
echo "[setup] No X11 or Xvfb required."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo " Installed packages:"
python -c "
import torch, transformers, accelerate, ai2thor
print(f'  torch:        {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  transformers: {transformers.__version__}')
print(f'  accelerate:   {accelerate.__version__}')
print(f'  ai2thor:      {ai2thor.__version__}')
try:
    import flash_attn
    print(f'  flash-attn:   {flash_attn.__version__}')
except ImportError:
    print('  flash-attn:   NOT INSTALLED (install precompiled wheel manually)')
"
echo ""
echo " Next steps:"
echo "   1. Install flash-attn (if not done):"
echo "      pip install /path/to/flash_attn-*.whl"
echo ""
echo "   2. Download ALFRED data:"
echo "      python scripts/download_alfred.py"
echo ""
echo "   3. Build candidate episode list:"
echo "      python scripts/build_candidate_list.py --alfred_data ../datasets/json_2.1.0"
echo ""
echo "   4. HuggingFace login (for Gemma 3):"
echo "      huggingface-cli login"
echo ""
echo "   5. Test AI2-THOR rendering (CloudRendering, no Xvfb needed):"
echo "      python scripts/test_render_one.py"
echo ""
echo "   6. Test VLM loading:"
echo "      python -c \"from src.models.registry import load_model; m = load_model('qwen25vl', use_full=True); print('OK')\""
echo ""
echo "========================================"
