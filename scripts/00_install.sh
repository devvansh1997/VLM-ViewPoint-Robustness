#!/usr/bin/env bash
# =============================================================================
# 00_install.sh — Create environment and install all dependencies
#
# Run once from the repo root:
#   bash scripts/00_install.sh
#
# After this script completes, run the sanity checks at the bottom
# before proceeding to Phase 1.
# =============================================================================

set -e  # exit immediately on any error

CONDA_ENV="viewpoint"
PYTHON_VERSION="3.10"
ALFRED_REPO="https://github.com/askforalfred/alfred.git"

echo "========================================"
echo " Phase 0: Environment Setup"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
echo "[00_install] Creating conda environment: $CONDA_ENV (Python $PYTHON_VERSION)"
conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------------------
echo "[00_install] Installing Python dependencies from env/requirements.txt"
pip install -r env/requirements.txt

# ---------------------------------------------------------------------------
# 3. Install AI2-THOR (headless support)
# ---------------------------------------------------------------------------
echo "[00_install] Installing ai2thor"
pip install ai2thor

# On Linux HPC: install Xvfb and set up virtual display
if [[ "$(uname -s)" == "Linux" ]]; then
    echo "[00_install] Linux detected — checking for Xvfb"
    if ! command -v Xvfb &> /dev/null; then
        echo "[00_install] WARNING: Xvfb not found. On HPC, install via:"
        echo "  sudo apt-get install xvfb  or  module load xvfb"
        echo "  Then run: Xvfb :1 -screen 0 1024x768x24 &"
        echo "            export DISPLAY=:1"
    else
        echo "[00_install] Xvfb found."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Clone and install ALFRED
# ---------------------------------------------------------------------------
echo "[00_install] Cloning ALFRED"
if [ ! -d "alfred" ]; then
    git clone "$ALFRED_REPO" alfred
else
    echo "[00_install] alfred/ already exists — skipping clone"
fi

cd alfred
pip install -r requirements.txt
cd ..

# ---------------------------------------------------------------------------
# 5. Create data directories (in case they don't exist)
# ---------------------------------------------------------------------------
echo "[00_install] Creating data directories"
mkdir -p data/alfred_episodes
mkdir -p data/rendered_frames
mkdir -p data/logs/raw
mkdir -p data/logs/aggregated
mkdir -p results

echo ""
echo "========================================"
echo " Sanity Checks (run manually after setup)"
echo "========================================"
echo ""
echo "  1. AI2-THOR headless render:"
echo "     python -c \""
echo "       from ai2thor.controller import Controller"
echo "       c = Controller(width=300, height=300)"
echo "       c.reset(scene='FloorPlan1')"
echo "       print('Frame shape:', c.last_event.frame.shape)"
echo "       c.stop()"
echo "     \""
echo ""
echo "  2. ALFRED episode load:"
echo "     python -c \""
echo "       from src.simulator.alfred_loader import load_all_episodes"
echo "       eps = load_all_episodes('path/to/alfred_data')"
echo "       print('Loaded', len(eps), 'episodes')"
echo "     \""
echo ""
echo "  3. Smallest VLM load + predict:"
echo "     python -c \""
echo "       from src.models.registry import load_model"
echo "       from PIL import Image; import numpy as np"
echo "       m = load_model('llava_onevision', use_full=False)"
echo "       img = Image.fromarray(np.zeros((300,300,3), dtype='uint8'))"
echo "       print('Response:', m.predict(img, 'A) MoveAhead\nReply with only the letter.'))"
echo "     \""
echo ""
echo "Do NOT proceed to Phase 1 until all three checks pass."
echo "========================================"
