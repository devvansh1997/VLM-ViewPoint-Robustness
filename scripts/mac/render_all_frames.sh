#!/usr/bin/env bash
# =============================================================================
# render_all_frames.sh — Render ALL frames on Mac via AI2-THOR
#
# Renders 240 episodes x 35 viewpoints = 8,400 frames.
# Estimated time on M1 Air: ~5-7 hours (runs overnight).
#
# After rendering, zip the frames for transfer to HPC:
#   bash scripts/mac/package_frames.sh
#
# Usage (from repo root):
#   bash scripts/mac/render_all_frames.sh
#   bash scripts/mac/render_all_frames.sh --episodes data/alfred_episodes/candidate_episodes.json
# =============================================================================

set -e

EPISODES="${1:-data/alfred_episodes/candidate_episodes.json}"
FRAMES_DIR="data/rendered_frames"

echo "========================================"
echo " Mac Frame Rendering"
echo " Episodes: $EPISODES"
echo " Output:   $FRAMES_DIR"
echo "========================================"

python - <<'PYEOF'
import sys, json, time
sys.path.insert(0, ".")

from src.simulator.alfred_loader import load_episode_list
from src.simulator.renderer import (
    render_frame, _build_controller,
    YAW_OFFSETS, PITCH_OFFSETS, frame_path,
)

EPISODES = sys.argv[1] if len(sys.argv) > 1 else "data/alfred_episodes/candidate_episodes.json"
FRAMES_DIR = "data/rendered_frames"

episodes = load_episode_list(EPISODES)
total_combos = len(YAW_OFFSETS) * len(PITCH_OFFSETS)
total_frames = len(episodes) * total_combos

print(f"Episodes:   {len(episodes)}")
print(f"Viewpoints: {total_combos} per episode (7 yaw x 5 pitch)")
print(f"Total:      {total_frames} frames to render")
print()

# Count already rendered frames (for resume)
import os
existing = set(os.listdir(FRAMES_DIR)) if os.path.isdir(FRAMES_DIR) else set()
skipped = 0

start = time.time()
rendered = 0

for i, ep in enumerate(episodes):
    # Check how many frames this episode already has
    ep_existing = sum(
        1 for y in YAW_OFFSETS for p in PITCH_OFFSETS
        if os.path.basename(frame_path(FRAMES_DIR, ep["episode_id"], y, p)) in existing
    )
    if ep_existing == total_combos:
        skipped += total_combos
        continue

    # Open one controller per episode to avoid memory leaks
    controller = _build_controller(headless=False)
    try:
        for yaw in YAW_OFFSETS:
            for pitch in PITCH_OFFSETS:
                fname = os.path.basename(frame_path(FRAMES_DIR, ep["episode_id"], yaw, pitch))
                if fname in existing:
                    skipped += 1
                    continue

                render_frame(controller, ep, yaw, pitch, FRAMES_DIR)
                rendered += 1

                total_done = rendered + skipped
                elapsed = time.time() - start
                rate = rendered / elapsed if elapsed > 0 else 0
                remaining = (total_frames - total_done) / rate if rate > 0 else 0

                print(
                    f"\r  [{total_done}/{total_frames}] "
                    f"Episode {i+1}/{len(episodes)} "
                    f"yaw={yaw:+d} pitch={pitch:+d} | "
                    f"{rate:.1f} frames/sec | "
                    f"ETA: {remaining/3600:.1f}h",
                    end="", flush=True
                )
    finally:
        controller.stop()

elapsed = time.time() - start
print(f"\n\nDone!")
print(f"  Rendered: {rendered} new frames")
print(f"  Skipped:  {skipped} (already existed)")
print(f"  Time:     {elapsed/3600:.1f} hours")
print(f"  Frames:   {FRAMES_DIR}/")
PYEOF

echo ""
echo "========================================"
echo " Rendering complete."
echo " Next: package for HPC transfer"
echo "   bash scripts/mac/package_frames.sh"
echo "========================================"
