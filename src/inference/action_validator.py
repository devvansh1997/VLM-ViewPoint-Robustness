"""
Action validator — Pass 2 of the two-pass pipeline.

Reads JSONL inference logs (produced on HPC with --skip_action_check),
replays each predicted action in AI2-THOR, and writes updated logs
with action_success filled in.

Optimizations:
  - Deduplicates on (episode_id, yaw, pitch, mapped_action) — identical
    simulator checks across models/phases are run only once.
  - Groups checks by scene and reuses the AI2-THOR controller within
    each scene batch (avoids restarting Unity per check).
  - Periodic checkpointing for resume support.

This script runs on Mac (where AI2-THOR works). No VLM needed.

Usage:
    python src/inference/action_validator.py \
        --logs_dir    data/logs/raw/ \
        --episodes    data/alfred_episodes/selected_episodes.json \
        --output_dir  data/logs/validated/
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulator.alfred_loader import load_episode_list
from src.simulator.success_checker import _build_action_kwargs


def parse_args():
    p = argparse.ArgumentParser(description="Validate predicted actions in AI2-THOR.")
    p.add_argument("--logs_dir", required=True,
                   help="Directory containing JSONL inference logs from HPC")
    p.add_argument("--episodes", required=True,
                   help="Path to selected_episodes.json (needs scene + start_pose)")
    p.add_argument("--output_dir", required=True,
                   help="Directory for validated JSONL output")
    p.add_argument("--model", default=None,
                   help="Only validate logs for this model (default: all)")
    p.add_argument("--phase", default=None,
                   help="Only validate logs for this phase (default: all)")
    return p.parse_args()


def load_inference_logs(logs_dir, model_filter=None, phase_filter=None):
    entries = []
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if model_filter and not fname.startswith(model_filter + "_"):
            continue
        if phase_filter and phase_filter not in fname:
            continue
        with open(os.path.join(logs_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


def load_completed_results(output_dir):
    """Load already-validated (ep_id, yaw, pitch, action) -> (success, error) results."""
    results = {}
    if not os.path.exists(output_dir):
        return results
    for fname in os.listdir(output_dir):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(output_dir, fname)) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    key = (e["episode_id"], e["yaw_offset"], e["pitch_offset"], e["mapped_action"])
                    if e.get("action_success") is not None:
                        results[key] = (e["action_success"], e.get("error_message", ""))
                except Exception:
                    pass
    return results


def build_episode_lookup(episodes):
    return {ep["episode_id"]: ep for ep in episodes}


def build_controller():
    from ai2thor.controller import Controller
    return Controller(
        width=300, height=300,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )


def run_check(controller, episode, action, yaw_offset, pitch_offset):
    """Execute a single action check using an existing controller."""
    pose = episode["start_pose"]
    raw = episode.get("raw", {})
    scene_data = raw.get("scene", {})

    if scene_data.get("object_poses"):
        controller.step("SetObjectPoses", objectPoses=scene_data["object_poses"])
    if scene_data.get("object_toggles"):
        controller.step("SetObjectToggles", objectToggles=scene_data["object_toggles"])

    target_rotation = (pose["rotation"] + yaw_offset) % 360
    target_horizon = float(pose["horizon"] + pitch_offset)

    controller.step(
        "TeleportFull",
        x=pose["x"], y=pose["y"], z=pose["z"],
        rotation={"x": 0, "y": target_rotation, "z": 0},
        horizon=target_horizon,
        standing=True,
        forceAction=True,
    )

    action_kwargs = _build_action_kwargs(controller, action)
    event = controller.step(action, **action_kwargs)

    return event.metadata["lastActionSuccess"], event.metadata.get("errorMessage", "")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    episodes = load_episode_list(args.episodes)
    ep_lookup = build_episode_lookup(episodes)

    entries = load_inference_logs(args.logs_dir, args.model, args.phase)
    print(f"[validator] Loaded {len(entries)} inference entries")

    # Load previously completed results for resume
    cached_results = load_completed_results(args.output_dir)
    print(f"[validator] Cached results from previous runs: {len(cached_results)}")

    # ---- Deduplication ----
    # Action success depends only on (episode_id, yaw, pitch, action), not on
    # which model/phase produced it. Build unique checks needed.
    unique_checks = {}  # (ep_id, yaw, pitch, action) -> True (needs check)
    no_action_entries = []

    for entry in entries:
        if entry.get("mapped_action") is None:
            no_action_entries.append(entry)
            continue
        key = (entry["episode_id"], entry["yaw_offset"], entry["pitch_offset"], entry["mapped_action"])
        if key not in cached_results:
            unique_checks[key] = True

    print(f"[validator] Unique checks needed: {len(unique_checks)} "
          f"(deduplicated from {len(entries) - len(no_action_entries)} entries)")
    print(f"[validator] Already cached: {len(cached_results)} | No action: {len(no_action_entries)}")

    # ---- Group by scene for controller reuse ----
    scene_groups = defaultdict(list)
    for key in unique_checks:
        ep_id = key[0]
        if ep_id in ep_lookup:
            scene = ep_lookup[ep_id]["scene"]
            scene_groups[scene].append(key)

    print(f"[validator] Scenes to process: {len(scene_groups)}")

    # ---- Run checks, batched by scene ----
    results = dict(cached_results)  # start with cached
    checks_done = 0
    errors = 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"check_results_{timestamp}.jsonl")

    total_checks = len(unique_checks)
    pbar = tqdm(total=total_checks, desc="Validating actions")

    controller = None
    current_scene = None

    try:
        for scene in sorted(scene_groups.keys()):
            checks = scene_groups[scene]

            # Start or reset controller for this scene
            try:
                if controller is None:
                    controller = build_controller()
                controller.reset(scene=scene)
                current_scene = scene
            except Exception as e:
                print(f"\n[validator] Controller error on {scene}: {e}")
                print("[validator] Restarting controller...")
                try:
                    if controller:
                        controller.stop()
                except Exception:
                    pass
                controller = None
                try:
                    controller = build_controller()
                    controller.reset(scene=scene)
                    current_scene = scene
                except Exception as e2:
                    print(f"[validator] Failed to restart for {scene}: {e2}")
                    errors += len(checks)
                    pbar.update(len(checks))
                    continue

            for key in checks:
                ep_id, yaw, pitch, action = key
                episode = ep_lookup[ep_id]

                # Reset scene if episode is in a different scene
                # (shouldn't happen within a batch, but safety check)
                if episode["scene"] != current_scene:
                    controller.reset(scene=episode["scene"])
                    current_scene = episode["scene"]

                try:
                    success, error_msg = run_check(controller, episode, action, yaw, pitch)
                    results[key] = (success, error_msg)
                    checks_done += 1
                except Exception as e:
                    print(f"\n[validator] Check failed {ep_id} yaw={yaw} pitch={pitch}: {e}")
                    results[key] = (None, str(e))
                    errors += 1

                    # Try to recover controller
                    try:
                        controller.reset(scene=current_scene)
                    except Exception:
                        try:
                            controller.stop()
                        except Exception:
                            pass
                        controller = build_controller()
                        controller.reset(scene=current_scene)

                pbar.update(1)

                # Checkpoint every 500 checks
                if (checks_done + errors) % 500 == 0:
                    _save_results_checkpoint(results_path, results, cached_results)

    finally:
        pbar.close()
        if controller:
            try:
                controller.stop()
            except Exception:
                pass

    # Save final results
    _save_results_checkpoint(results_path, results, cached_results)

    print(f"\n[validator] Checks complete: {checks_done} | Errors: {errors}")
    print(f"[validator] Total cached results: {len(results)}")

    # ---- Write validated entries ----
    output_path = os.path.join(args.output_dir, f"validated_{timestamp}.jsonl")
    written = 0

    with open(output_path, "w") as out_f:
        for entry in entries:
            if entry.get("mapped_action") is None:
                entry["action_success"] = False
                entry["error_message"] = entry.get("error_message", "no_action_mapped")
            else:
                key = (entry["episode_id"], entry["yaw_offset"],
                       entry["pitch_offset"], entry["mapped_action"])
                if key in results:
                    entry["action_success"], entry["error_message"] = results[key]

            out_f.write(json.dumps(entry) + "\n")
            written += 1

    print(f"[validator] Wrote {written} validated entries to {output_path}")


def _save_results_checkpoint(path, results, exclude_cached):
    """Save only new results (not previously cached) as checkpoint."""
    with open(path, "w") as f:
        for key, (success, error) in results.items():
            if key in exclude_cached:
                continue
            entry = {
                "episode_id": key[0],
                "yaw_offset": key[1],
                "pitch_offset": key[2],
                "mapped_action": key[3],
                "action_success": success,
                "error_message": error,
            }
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
