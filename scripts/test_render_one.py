"""
Quick smoke test: render a single ALFRED episode frame via AI2-THOR.

This validates the full rendering pipeline:
  1. Load an episode from candidate_episodes.json
  2. Initialize AI2-THOR controller
  3. Reset scene, teleport to start pose
  4. Capture and save a frame

Usage (from repo root):
    python scripts/test_render_one.py
    python scripts/test_render_one.py --episodes data/alfred_episodes/candidate_episodes.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Render a single test frame.")
    parser.add_argument(
        "--episodes",
        default="data/alfred_episodes/candidate_episodes.json",
        help="Path to candidate_episodes.json",
    )
    parser.add_argument(
        "--output_dir",
        default="data/rendered_frames",
        help="Directory to save the test frame",
    )
    args = parser.parse_args()

    # Load episodes
    with open(args.episodes) as f:
        episodes = json.load(f)

    if not episodes:
        print("[test] ERROR: No episodes found.")
        sys.exit(1)

    ep = episodes[0]
    print(f"[test] Episode:     {ep['episode_id']}")
    print(f"[test] Task type:   {ep['task_type']}")
    print(f"[test] Scene:       {ep['scene']}")
    print(f"[test] Instruction: {ep['instruction'][:80]}...")
    print(f"[test] Start pose:  {ep['start_pose']}")

    # Render original pose (yaw=0, pitch=0)
    from src.simulator.renderer import render_original_pose, frame_path

    print("\n[test] Initializing AI2-THOR and rendering frame...")
    path = render_original_pose(ep, args.output_dir, headless=False)
    print(f"[test] Frame saved to: {path}")

    # Verify the file
    from PIL import Image
    img = Image.open(path)
    print(f"[test] Image size: {img.size}, mode: {img.mode}")

    # Render one perturbed frame to test offsets
    from src.simulator.renderer import render_frame, _build_controller

    print("\n[test] Rendering perturbed frame (yaw=+15, pitch=-10)...")
    controller = _build_controller(headless=False)
    try:
        path2 = render_frame(controller, ep, yaw_offset=15, pitch_offset=-10,
                             output_dir=args.output_dir)
        print(f"[test] Perturbed frame saved to: {path2}")
        img2 = Image.open(path2)
        print(f"[test] Image size: {img2.size}, mode: {img2.mode}")
    finally:
        controller.stop()

    print("\n[test] SUCCESS — AI2-THOR rendering pipeline is working!")
    print(f"[test] Check the frames in: {args.output_dir}/")


if __name__ == "__main__":
    main()
