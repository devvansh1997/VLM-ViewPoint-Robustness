#!/usr/bin/env bash
# =============================================================================
# 02_run_perturbation.sh — Phase 2: Core rotation perturbation study
#
# Runs all 4 models (full size) across all yaw/pitch offsets on locked episodes.
# Frames are pre-rendered on Mac — this script only does VLM inference.
#
# Usage:
#   bash scripts/hpc/02_run_perturbation.sh                  # all 4 models
#   bash scripts/hpc/02_run_perturbation.sh qwen25vl         # single model
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"

if [ ! -f "$EPISODES" ]; then
    echo "ERROR: $EPISODES not found."
    echo "Run Phase 1 baseline + filter_episodes.py first."
    exit 1
fi

if [[ -n "$1" ]]; then
    MODELS=("$1")
else
    MODELS=("qwen25vl" "internvl3" "gemma3" "llava_onevision")
fi

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo " Phase 2: Core Perturbation — $MODEL"
    echo "========================================"

    python src/inference/run_inference.py \
        --model        "$MODEL" \
        --phase        core \
        --episodes     "$EPISODES" \
        --frames_dir   "$FRAMES_DIR" \
        --output_dir   "$OUTPUT_DIR" \
        --use_full_model

    echo "[perturbation] $MODEL done."
done

echo ""
echo "========================================"
echo " All perturbation runs complete."
echo " Next: aggregate logs"
echo "   python src/analysis/aggregate_logs.py"
echo "========================================"
