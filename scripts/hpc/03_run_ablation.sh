#!/usr/bin/env bash
# =============================================================================
# 03_run_ablation.sh — Phase 4: Prompt augmentation ablation
#
# Runs both ablation variants (exact + qualitative) on best and worst models.
# Reuses pre-rendered frames — inference only.
#
# Usage:
#   bash scripts/hpc/03_run_ablation.sh <model1> [model2] [model3] ...
#   bash scripts/hpc/03_run_ablation.sh qwen25vl llava_onevision
#   bash scripts/hpc/03_run_ablation.sh qwen25vl internvl3 gemma3 llava_onevision
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"

MODELS=("${@:-qwen25vl llava_onevision}")
if [ $# -eq 0 ]; then
    MODELS=(qwen25vl llava_onevision)
fi

echo "========================================"
echo " Phase 4: Ablation"
echo " Models: ${MODELS[*]}"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
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
