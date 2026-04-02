#!/usr/bin/env bash
# =============================================================================
# 02_run_baseline.sh — Phase 1: Baseline evaluation at original pose (yaw=0, pitch=0)
#
# Steps:
#   1. Render original-pose frames for all candidate episodes
#   2. Run all 4 models on those frames
#   3. Print next-step instructions for filtering
#
# After this completes, lock the episode list:
#   python src/analysis/filter_episodes.py \
#       --logs_dir data/logs/raw \
#       --episodes data/alfred_episodes/candidate_episodes.json \
#       --output   data/alfred_episodes/selected_episodes.json
#
# On HPC, submit as separate SLURM jobs (one per model):
#   sbatch --gres=gpu:a100:1 scripts/02_run_baseline.sh <model_name>
# =============================================================================

set -e

EPISODES="data/alfred_episodes/candidate_episodes.json"
FRAMES_DIR="data/rendered_frames"
OUTPUT_DIR="data/logs/raw"
HEADLESS_FLAG=""

if [[ "$(uname -s)" == "Linux" ]]; then
    HEADLESS_FLAG="--headless"
    echo "[02_baseline] Linux detected — using headless mode"
fi

# ---------------------------------------------------------------------------
# Step 1 — Render original-pose frames (yaw=0, pitch=0) for all candidates
# Only renders if the file doesn't already exist (overwrite=False default)
# ---------------------------------------------------------------------------
echo "========================================"
echo " Step 1: Rendering original-pose frames"
echo "========================================"

python - <<EOF
import sys
sys.path.insert(0, ".")
import json
from tqdm import tqdm
from src.simulator.alfred_loader import load_episode_list
from src.simulator.renderer import render_original_pose

headless = "$HEADLESS_FLAG" == "--headless"
episodes = load_episode_list("$EPISODES")

print(f"Rendering original-pose frames for {len(episodes)} episodes...")
for ep in tqdm(episodes, desc="Rendering"):
    render_original_pose(ep, output_dir="$FRAMES_DIR", headless=headless, overwrite=False)

print("Done.")
EOF

echo "[02_baseline] Original-pose frames ready."

# ---------------------------------------------------------------------------
# Step 2 — Run all 4 models on original-pose frames
# If a specific model is passed as arg (for HPC), run only that one
# ---------------------------------------------------------------------------

if [[ -n "$1" ]]; then
    MODELS=("$1")
else
    MODELS=("qwen25vl" "internvl3" "gemma3" "llava_onevision")
fi

echo ""
echo "========================================"
echo " Step 2: Running baseline inference"
echo " Models: ${MODELS[*]}"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    echo "--- Model: $MODEL ---"

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

# ---------------------------------------------------------------------------
# Step 3 — Instructions to lock the episode list
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo " All baseline runs complete."
echo ""
echo " Exit criterion check:"
python - <<EOF
import sys, json, os
from collections import Counter

log_dir = "$OUTPUT_DIR"
counts = Counter()
for f in os.listdir(log_dir):
    if "baseline" in f and f.endswith(".jsonl"):
        with open(os.path.join(log_dir, f)) as fh:
            for line in fh:
                import json as j
                try:
                    e = j.loads(line)
                    counts[e["model"]] += 1
                except: pass

print("  Baseline log counts per model:")
for model, n in sorted(counts.items()):
    print(f"    {model}: {n} episodes")
EOF

echo ""
echo " Next: lock the episode list"
echo "   python src/analysis/filter_episodes.py \\"
echo "     --logs_dir $OUTPUT_DIR \\"
echo "     --episodes $EPISODES \\"
echo "     --output   data/alfred_episodes/selected_episodes.json"
echo "========================================"
