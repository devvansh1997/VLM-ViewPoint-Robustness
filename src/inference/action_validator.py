"""
Action validator — Pass 2 of the two-pass pipeline.

Reads JSONL inference logs (produced on HPC with --skip_action_check),
replays each predicted action in AI2-THOR, and writes updated logs
with action_success filled in.

This script runs on Mac (where AI2-THOR works). No VLM needed.

Usage:
    python src/inference/action_validator.py \
        --logs_dir    data/logs/raw/ \
        --episodes    data/alfred_episodes/selected_episodes.json \
        --output_dir  data/logs/validated/

Resume support: already-validated entries are skipped automatically.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulator.alfred_loader import load_episode_list
from src.simulator.success_checker import check_action_success


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


def load_inference_logs(logs_dir: str, model_filter: str = None, phase_filter: str = None) -> list[dict]:
    """Load all JSONL entries from inference logs, optionally filtered."""
    entries = []
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl"):
            continue
        if model_filter and not fname.startswith(model_filter + "_"):
            continue
        if phase_filter and phase_filter not in fname:
            continue

        fpath = os.path.join(logs_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


def load_completed_keys(output_dir: str) -> set:
    """Load already-validated (episode_id, model, yaw, pitch, phase) keys."""
    completed = set()
    if not os.path.exists(output_dir):
        return completed

    for fname in os.listdir(output_dir):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(output_dir, fname)) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    key = (e["episode_id"], e["model"], e["yaw_offset"],
                           e["pitch_offset"], e["phase"])
                    completed.add(key)
                except Exception:
                    pass
    return completed


def build_episode_lookup(episodes: list[dict]) -> dict:
    """Build episode_id -> episode dict for quick lookup."""
    lookup = {}
    for ep in episodes:
        lookup[ep["episode_id"]] = ep
    return lookup


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load episodes (need scene + start_pose for AI2-THOR)
    episodes = load_episode_list(args.episodes)
    ep_lookup = build_episode_lookup(episodes)

    # Load inference logs
    entries = load_inference_logs(args.logs_dir, args.model, args.phase)
    print(f"[validator] Loaded {len(entries)} inference entries")

    # Load already-completed validations for resume
    completed = load_completed_keys(args.output_dir)
    print(f"[validator] Already validated: {len(completed)} entries")

    # Filter to entries that need validation
    to_validate = []
    skipped_no_action = 0
    skipped_completed = 0

    for entry in entries:
        key = (entry["episode_id"], entry["model"], entry["yaw_offset"],
               entry["pitch_offset"], entry["phase"])

        if key in completed:
            skipped_completed += 1
            continue

        if entry.get("mapped_action") is None:
            skipped_no_action += 1
            # Write immediately as failed — no action to validate
            entry["action_success"] = False
            entry["error_message"] = entry.get("error_message", "no_action_mapped")
            to_validate.append(("write_only", entry))
            continue

        if entry["episode_id"] not in ep_lookup:
            print(f"[validator] Episode {entry['episode_id']} not in episodes file — skipping")
            continue

        to_validate.append(("validate", entry))

    print(f"[validator] To validate: {len(to_validate)} | "
          f"Skipped (done): {skipped_completed} | "
          f"Skipped (no action): {skipped_no_action}")

    if not to_validate:
        print("[validator] Nothing to validate. Done.")
        return

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"validated_{timestamp}.jsonl")

    validated_count = 0
    failed_count = 0

    with open(output_path, "w") as out_f:
        for action_type, entry in tqdm(to_validate, desc="Validating actions"):
            if action_type == "write_only":
                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                failed_count += 1
                continue

            episode = ep_lookup[entry["episode_id"]]

            try:
                success, error_msg = check_action_success(
                    episode=episode,
                    action=entry["mapped_action"],
                    yaw_offset=entry["yaw_offset"],
                    pitch_offset=entry["pitch_offset"],
                    headless=False,
                )
                entry["action_success"] = success
                entry["error_message"] = error_msg
            except Exception as e:
                print(f"[validator] Error on {entry['episode_id']} "
                      f"yaw={entry['yaw_offset']} pitch={entry['pitch_offset']}: {e}")
                entry["action_success"] = None
                entry["error_message"] = str(e)

            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            validated_count += 1

    print(f"\n[validator] Done. Validated: {validated_count} | Failed (no action): {failed_count}")
    print(f"[validator] Output: {output_path}")


if __name__ == "__main__":
    main()
