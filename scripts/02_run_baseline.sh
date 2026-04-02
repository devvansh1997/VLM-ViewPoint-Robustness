#!/usr/bin/env bash
# =============================================================================
# 02_run_baseline.sh — Phase 1: Run all models at original pose (yaw=0, pitch=0)
#
# This establishes per-model baseline accuracy and identifies which episodes
# to keep (filter out floor-effect episodes where no model exceeds 30% success).
#
# After running, execute the filter step to lock selected_episodes.json:
#   python src/analysis/filter_episodes.py
#
# On HPC, submit as separate SLURM jobs (one per model):
#   sbatch --gres=gpu:a100:1 scripts/02_run_baseline.sh  (with MODEL set)
# =============================================================================

set -e

EPISODES="data/alfred_episodes/candidate_episodes.json"  # full candidate set
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"
HEADLESS_FLAG=""

if [[ "$(uname -s)" == "Linux" ]]; then
    HEADLESS_FLAG="--headless"
fi

# ---------------------------------------------------------------------------
# Run baseline for all models
# Local: uses small model variants (no --use_full_model flag)
# HPC:   add --use_full_model to each command
# ---------------------------------------------------------------------------

MODELS=("qwen25vl" "internvl3" "gemma3" "llava_onevision")

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo " Baseline: $MODEL"
    echo "========================================"

    python src/inference/run_inference.py \
        --model        "$MODEL" \
        --phase        baseline \
        --episodes     "$EPISODES" \
        --frames_dir   "$FRAMES_DIR" \
        --output_dir   "$OUTPUT_DIR" \
        $HEADLESS_FLAG
        # Add --use_full_model on HPC

    echo "[02_baseline] $MODEL done."
done

echo ""
echo "========================================"
echo " All baseline runs complete."
echo " Next: filter episodes and lock the list."
echo ""
echo " Run: python src/analysis/filter_episodes.py \\"
echo "        --logs_dir $OUTPUT_DIR \\"
echo "        --episodes $EPISODES \\"
echo "        --output data/alfred_episodes/selected_episodes.json"
echo "========================================"
