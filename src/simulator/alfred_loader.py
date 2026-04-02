"""
Loads ALFRED episode metadata from traj_data.json files.

ALFRED validation data is organized as:
  alfred_data/
    valid_seen/ and valid_unseen/
      {task_type}-{object}-...-{scene}/
        trial_{id}/
          traj_data.json

Each traj_data.json contains the scene, agent start pose, NL instruction,
high-level sub-goals, and low-level action trajectory.
"""

import json
import os
from pathlib import Path
from typing import Optional


# Maps ALFRED task_type strings to clean category names used in logs
TASK_TYPE_MAP = {
    "pick_and_place_simple":          "pick_and_place",
    "pick_two_obj_and_place":         "pick_and_place",
    "look_at_obj_in_light":           "pick_and_place",
    "pick_and_place_with_movable_recep": "pick_and_place",
    "stack_and_place":                "stack_and_place",
    "pick_heat_then_place_in_recep":  "heat_and_place",
    "pick_cool_then_place_in_recep":  "cool_and_place",
    "pick_clean_then_place_in_recep": "clean_and_place",
}


def load_episode(traj_data_path: str) -> dict:
    """
    Parse a single traj_data.json into a flat episode dict.

    Returns a dict with keys:
        episode_id    — trial folder name (e.g. "trial_T20190909_...")
        task_type     — cleaned category (e.g. "pick_and_place")
        scene         — AI2-THOR floor plan name (e.g. "FloorPlan1")
        instruction   — first annotator's task description
        start_pose    — dict with x, y, z, rotation, horizon
        raw           — original traj_data dict (for any downstream use)
    """
    with open(traj_data_path, "r") as f:
        traj = json.load(f)

    raw_task_type = traj.get("task_type", "")
    task_type = TASK_TYPE_MAP.get(raw_task_type, raw_task_type)

    # First annotator's task description is the NL instruction
    anns = traj.get("turk_annotations", {}).get("anns", [])
    instruction = anns[0]["task_desc"] if anns else ""

    scene_data = traj["scene"]
    init = scene_data["init_action"]

    start_pose = {
        "x":        init["x"],
        "y":        init["y"],
        "z":        init["z"],
        "rotation": init["rotation"],   # yaw in degrees (AI2-THOR y-axis)
        "horizon":  init["horizon"],    # pitch in degrees (0=horizontal, +ve=down)
    }

    # episode_id is the trial folder name
    episode_id = Path(traj_data_path).parent.name

    return {
        "episode_id":  episode_id,
        "task_type":   task_type,
        "scene":       scene_data["floor_plan"],
        "instruction": instruction,
        "start_pose":  start_pose,
        "raw":         traj,
    }


def load_all_episodes(alfred_data_dir: str, splits: Optional[list] = None) -> list[dict]:
    """
    Scan an ALFRED data directory and load all episodes.

    Args:
        alfred_data_dir: Root directory containing valid_seen/, valid_unseen/, etc.
        splits: List of split subfolder names to include. Defaults to ["valid_seen", "valid_unseen"].

    Returns:
        List of episode dicts.
    """
    if splits is None:
        splits = ["valid_seen", "valid_unseen"]

    episodes = []
    root = Path(alfred_data_dir)

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[alfred_loader] Warning: split directory not found: {split_dir}")
            continue

        for traj_file in sorted(split_dir.rglob("traj_data.json")):
            try:
                ep = load_episode(str(traj_file))
                episodes.append(ep)
            except Exception as e:
                print(f"[alfred_loader] Skipping {traj_file}: {e}")

    print(f"[alfred_loader] Loaded {len(episodes)} episodes from {alfred_data_dir}")
    return episodes


def save_episode_list(episodes: list[dict], output_path: str) -> None:
    """
    Save the selected episode list to JSON.
    Only saves the fields needed for inference (not the full raw traj).
    This file becomes the single source of truth for all phases.
    """
    slim = []
    for ep in episodes:
        slim.append({
            "episode_id":  ep["episode_id"],
            "task_type":   ep["task_type"],
            "scene":       ep["scene"],
            "instruction": ep["instruction"],
            "start_pose":  ep["start_pose"],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"[alfred_loader] Saved {len(slim)} episodes to {output_path}")


def load_episode_list(path: str) -> list[dict]:
    """Load the locked selected_episodes.json."""
    with open(path, "r") as f:
        episodes = json.load(f)
    print(f"[alfred_loader] Loaded {len(episodes)} episodes from {path}")
    return episodes
