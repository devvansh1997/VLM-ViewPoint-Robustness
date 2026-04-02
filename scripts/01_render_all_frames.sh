#!/usr/bin/env bash
# =============================================================================
# 01_render_all_frames.sh — Pre-render all perturbed frames to disk
#
# Run AFTER locking selected_episodes.json (end of Phase 1).
# This decouples the simulator from inference — if a model crashes,
# you re-run inference only, not the simulator.
#
# Usage:
#   bash scripts/01_render_all_frames.sh
#
# On Linux HPC, set DISPLAY before running:
#   Xvfb :1 -screen 0 1024x768x24 &
#   export DISPLAY=:1
#   bash scripts/01_render_all_frames.sh
# =============================================================================

set -e

EPISODES="data/alfred_episodes/selected_episodes.json"
FRAMES_DIR="data/rendered_frames"
HEADLESS_FLAG=""

# Auto-detect Linux and set headless flag
if [[ "$(uname -s)" == "Linux" ]]; then
    HEADLESS_FLAG="--headless"
    echo "[01_render] Linux detected — using headless mode"
fi

echo "[01_render] Rendering all frames for episodes in: $EPISODES"
echo "[01_render] Output directory: $FRAMES_DIR"

python - <<EOF
import sys
sys.path.insert(0, ".")
import json
from tqdm import tqdm
from src.simulator.alfred_loader import load_episode_list
from src.simulator.renderer import render_episode_all_offsets

episodes = load_episode_list("$EPISODES")
headless = "$HEADLESS_FLAG" == "--headless"

print(f"Rendering {len(episodes)} episodes × 7 yaw × 5 pitch = {len(episodes)*35} frames")

for ep in tqdm(episodes, desc="Episodes"):
    render_episode_all_offsets(
        episode=ep,
        output_dir="$FRAMES_DIR",
        headless=headless,
        overwrite=False,
    )

print("Done. All frames saved to $FRAMES_DIR")
EOF

echo "[01_render] Frame rendering complete."
echo "[01_render] Verify frame count:"
python -c "
import os
n = len([f for f in os.listdir('$FRAMES_DIR') if f.endswith('.png')])
print(f'  Found {n} PNG files in $FRAMES_DIR')
"
