#!/usr/bin/env bash
# =============================================================================
# 03_run_perturbation.sh — Phase 2: Core rotation perturbation study
#
# Runs all 4 models across all rotation levels on the locked episode list.
# On HPC, submit as an array job — one job per model.
#
# Local run (small models, all offsets):
#   bash scripts/03_run_perturbation.sh
#
# HPC array job:
#   for MODEL in qwen25vl internvl3 spatial_mllm llava_onevision; do
#     sbatch --gres=gpu:a100:1 scripts/03_run_perturbation.sh $MODEL  # qwen25vl internvl3 gemma3 llava_onevision
#   done
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"
HEADLESS_FLAG=""

if [[ "$(uname -s)" == "Linux" ]]; then
    HEADLESS_FLAG="--headless"
fi

# If a model name is passed as arg (for HPC array jobs), run only that model
if [[ -n "$1" ]]; then
    MODELS=("$1")
else
    MODELS=("qwen25vl" "internvl3" "gemma3" "llava_onevision")
fi

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo " Core Perturbation: $MODEL"
    echo "========================================"

    python src/inference/run_inference.py \
        --model        "$MODEL" \
        --phase        core \
        --episodes     "$EPISODES" \
        --frames_dir   "$FRAMES_DIR" \
        --output_dir   "$OUTPUT_DIR" \
        $HEADLESS_FLAG
        # Add --use_full_model on HPC

    echo "[03_perturbation] $MODEL done."
done

echo ""
echo "========================================"
echo " All core perturbation runs complete."
echo " Next: validate log completeness."
echo ""
echo " Run: python - <<EOF"
echo "   import json, os"
echo "   from src.simulator.renderer import YAW_OFFSETS, PITCH_OFFSETS"
echo "   eps = json.load(open('$EPISODES'))"
echo "   expected = len(eps) * len(YAW_OFFSETS) * len(PITCH_OFFSETS) * 4"
echo "   print('Expected:', expected, 'log entries')"
echo " EOF"
echo ""
echo " Then aggregate:"
echo "   python src/analysis/aggregate_logs.py"
echo "========================================"
