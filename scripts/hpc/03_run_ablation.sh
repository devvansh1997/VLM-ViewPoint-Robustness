#!/usr/bin/env bash
# =============================================================================
# 03_run_ablation.sh — Phase 4: Prompt augmentation ablation
#
# Runs both ablation variants (exact + qualitative) on best and worst models.
# Reuses pre-rendered frames — inference only.
#
# Usage:
#   bash scripts/hpc/03_run_ablation.sh <best_model> <worst_model>
#   bash scripts/hpc/03_run_ablation.sh qwen25vl llava_onevision
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"

BEST_MODEL="${1:-qwen25vl}"
WORST_MODEL="${2:-llava_onevision}"

echo "========================================"
echo " Phase 4: Ablation"
echo " Best model:  $BEST_MODEL"
echo " Worst model: $WORST_MODEL"
echo "========================================"

for MODEL in "$BEST_MODEL" "$WORST_MODEL"; do
    for VARIANT in "exact" "qualitative"; do
        echo ""
        echo "--- $MODEL | variant: $VARIANT ---"

        python src/inference/run_inference.py \
            --model             "$MODEL" \
            --phase             ablation \
            --episodes          "$EPISODES" \
            --frames_dir        "$FRAMES_DIR" \
            --output_dir        "$OUTPUT_DIR" \
            --ablation_variant  "$VARIANT" \
            --use_full_model \
            --skip_action_check  # action validation runs on Mac (Pass 2)

        echo "[ablation] $MODEL ($VARIANT) done."
    done
done

echo ""
echo "========================================"
echo " Ablation complete."
echo " Next: re-aggregate and analyze"
echo "   python src/analysis/aggregate_logs.py"
echo "   python src/analysis/ablation.py"
echo "========================================"
