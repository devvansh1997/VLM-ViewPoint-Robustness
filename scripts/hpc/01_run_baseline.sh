#!/usr/bin/env bash
# =============================================================================
# 01_run_baseline.sh — Phase 1: Baseline evaluation at original pose (yaw=0, pitch=0)
#
# PREREQ: Rendered frames must already exist in data/rendered_frames/
#         (rendered on Mac, transferred via zip)
#
# Runs all 4 models (full size) on original-pose frames.
# After this completes, filter episodes to lock selected_episodes.json.
#
# Usage:
#   bash scripts/hpc/01_run_baseline.sh              # all 4 models
#   bash scripts/hpc/01_run_baseline.sh qwen25vl     # single model
# =============================================================================

set -e

EPISODES="data/alfred_episodes/candidate_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"

# Verify frames exist
FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.png" 2>/dev/null | wc -l)
if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No frames found in $FRAMES_DIR"
    echo "Transfer rendered frames from Mac first:"
    echo "  scp data/rendered_frames.zip <hpc>:~/Independent-Study/VLM-ViewPoint-Robustness/data/"
    echo "  cd data && unzip rendered_frames.zip"
    exit 1
fi
echo "[baseline] Found $FRAME_COUNT frames in $FRAMES_DIR"

# Select models
if [[ -n "$1" ]]; then
    MODELS=("$1")
else
    MODELS=("qwen25vl" "internvl3" "gemma3" "llava_onevision")
fi

echo "========================================"
echo " Phase 1: Baseline Inference"
echo " Models: ${MODELS[*]}"
echo " Frames: $FRAME_COUNT"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Model: $MODEL ---"

    python src/inference/run_inference.py \
        --model        "$MODEL" \
        --phase        baseline \
        --episodes     "$EPISODES" \
        --frames_dir   "$FRAMES_DIR" \
        --output_dir   "$OUTPUT_DIR" \
        --use_full_model \
        --skip_action_check  # action validation runs on Mac (Pass 2)

    echo "[baseline] $MODEL done."
done

echo ""
echo "========================================"
echo " Baseline complete."
echo ""
echo " Next: lock the episode list"
echo "   python src/analysis/filter_episodes.py \\"
echo "     --logs_dir $OUTPUT_DIR \\"
echo "     --episodes $EPISODES \\"
echo "     --output   data/alfred_episodes/selected_episodes.json"
echo "========================================"
