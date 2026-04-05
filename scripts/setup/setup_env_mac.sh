#!/usr/bin/env bash
# =============================================================================
# setup_env_mac.sh — Create conda environment on macOS (Intel or Apple Silicon)
#
# Apple Silicon (M1/M2/M3) runs AI2-THOR via Rosetta 2.
# VLM inference is NOT expected to work on Mac — this setup is for testing
# the AI2-THOR rendering pipeline and ALFRED data loading only.
#
# Run from the repo root:
#   chmod +x scripts/setup_env_mac.sh
#   bash scripts/setup_env_mac.sh
# =============================================================================

set -e

CONDA_ENV="viewpoint"
PYTHON_VERSION="3.10"

echo "========================================"
echo " Phase 0: Environment Setup (macOS)"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
echo "[setup] Creating conda environment: $CONDA_ENV (Python $PYTHON_VERSION)"
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# 2. Install PyTorch (CPU-only for Mac — no CUDA)
# ---------------------------------------------------------------------------
echo "[setup] Installing PyTorch (CPU)..."
pip install torch torchvision

# ---------------------------------------------------------------------------
# 3. Install core requirements
# ---------------------------------------------------------------------------
echo "[setup] Installing requirements..."
pip install -r env/requirements.txt

# ---------------------------------------------------------------------------
# 4. Install AI2-THOR
# ---------------------------------------------------------------------------
echo "[setup] Installing ai2thor..."
pip install ai2thor

# ---------------------------------------------------------------------------
# 5. Install ALFRED download dependencies
# ---------------------------------------------------------------------------
echo "[setup] Installing download tools..."
pip install requests py7zr

# ---------------------------------------------------------------------------
# 6. Install model-specific dependencies (for import compatibility)
# ---------------------------------------------------------------------------
echo "[setup] Installing qwen-vl-utils, einops, timm..."
pip install qwen-vl-utils einops timm

# ---------------------------------------------------------------------------
# 7. Create data directories
# ---------------------------------------------------------------------------
echo "[setup] Creating data directories..."
mkdir -p data/alfred_episodes
mkdir -p data/rendered_frames
mkdir -p data/logs/raw
mkdir -p data/logs/aggregated
mkdir -p results

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo " Activate environment:  conda activate $CONDA_ENV"
echo ""
echo " Next steps:"
echo "   1. Download ALFRED data:"
echo "      python scripts/download_alfred.py"
echo ""
echo "   2. Build candidate list:"
echo "      python scripts/build_candidate_list.py --alfred_data ../datasets/json_2.1.0"
echo ""
echo "   3. Test AI2-THOR rendering:"
echo "      python -c \\"
echo "        \"from ai2thor.controller import Controller; \\"
echo "         c = Controller(width=300, height=300); \\"
echo "         c.reset(scene='FloorPlan1'); \\"
echo "         print('Frame shape:', c.last_event.frame.shape); \\"
echo "         c.stop()\""
echo ""
echo "   4. Test rendering a single episode frame:"
echo "      python scripts/test_render_one.py"
echo "========================================"
