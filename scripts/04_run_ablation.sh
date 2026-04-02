#!/usr/bin/env bash
# =============================================================================
# 04_run_ablation.sh — Phase 4: Prompt augmentation ablation
#
# Runs both ablation variants (exact + qualitative) on the best and worst
# performing models identified from Phase 2.
#
# The pre-rendered frames from Phase 2 are reused directly — no re-rendering.
#
# Usage:
#   bash scripts/04_run_ablation.sh <best_model> <worst_model>
#
# Example:
#   bash scripts/04_run_ablation.sh qwen25vl llava_onevision
#
# If models are not passed, you must edit BEST_MODEL and WORST_MODEL below.
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"
HEADLESS_FLAG=""

if [[ "$(uname -s)" == "Linux" ]]; then
    HEADLESS_FLAG="--headless"
fi

# Set best/worst model from args or defaults
BEST_MODEL="${1:-qwen25vl}"       # replace after Phase 2 results
WORST_MODEL="${2:-llava_onevision}"  # replace after Phase 2 results

echo "========================================"
echo " Phase 4: Ablation"
echo " Best model:  $BEST_MODEL"
echo " Worst model: $WORST_MODEL"
echo "========================================"

for MODEL in "$BEST_MODEL" "$WORST_MODEL"; do
    for VARIANT in "exact" "qualitative"; do
        echo "--- Model: $MODEL | Variant: $VARIANT ---"

        python src/inference/run_inference.py \
            --model             "$MODEL" \
            --phase             ablation \
            --episodes          "$EPISODES" \
            --frames_dir        "$FRAMES_DIR" \
            --output_dir        "$OUTPUT_DIR" \
            --ablation_variant  "$VARIANT" \
            $HEADLESS_FLAG
            # Add --use_full_model on HPC

        echo "[04_ablation] $MODEL ($VARIANT) done."
    done
done

echo ""
echo "========================================"
echo " Ablation runs complete."
echo " Next: re-aggregate logs to include ablation phase."
echo "   python src/analysis/aggregate_logs.py"
echo ""
echo " Then run ablation analysis:"
echo "   python src/analysis/ablation.py"
echo "========================================"
