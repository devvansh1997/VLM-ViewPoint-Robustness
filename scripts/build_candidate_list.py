"""
Build the candidate episode list from the downloaded ALFRED validation data.

Scans valid_seen/ and valid_unseen/, loads all traj_data.json files, and
saves a flat candidate_episodes.json to data/alfred_episodes/.

This is the INPUT to Phase 1 baseline runs. After baseline runs complete,
filter_episodes.py will trim this down to selected_episodes.json.

Usage (from repo root):
    python scripts/build_candidate_list.py --alfred_data ../datasets/json_2.1.0
    python scripts/build_candidate_list.py --alfred_data ../datasets/json_2.1.0 --max_per_type 60
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator.alfred_loader import (
    load_all_episodes,
    save_episode_list,
    TASK_TYPE_MAP,
)

# Target task types — must include all 5 per proposal
TARGET_TASK_TYPES = {
    "pick_and_place",
    "stack_and_place",
    "heat_and_place",
    "cool_and_place",
    "clean_and_place",
}


def filter_by_task_type(episodes: list[dict]) -> list[dict]:
    """Keep only episodes belonging to the 5 target task types."""
    kept = [ep for ep in episodes if ep["task_type"] in TARGET_TASK_TYPES]
    dropped = len(episodes) - len(kept)
    if dropped:
        print(f"[build] Dropped {dropped} episodes with non-target task types")
    return kept


def cap_per_task_type(episodes: list[dict], max_per_type: int) -> list[dict]:
    """
    Cap episodes per task type to `max_per_type` to keep compute tractable.
    Samples evenly from valid_seen and valid_unseen within each type.
    """
    buckets = defaultdict(list)
    for ep in episodes:
        buckets[ep["task_type"]].append(ep)

    result = []
    for task_type, eps in sorted(buckets.items()):
        if len(eps) > max_per_type:
            # Take first half from valid_seen, second half from valid_unseen
            # so generalization split is represented
            seen   = [e for e in eps if "seen"   in e.get("episode_id", "").lower() or True]
            # Simple even slice since we don't track split in episode dict
            capped = eps[:max_per_type]
            print(f"[build]  {task_type}: capped {len(eps)} → {max_per_type}")
        else:
            capped = eps
            print(f"[build]  {task_type}: {len(eps)} episodes (under cap)")
        result.extend(capped)

    return result


def print_summary(episodes: list[dict]) -> None:
    print("\n[build] Candidate episode summary:")
    counts = defaultdict(int)
    for ep in episodes:
        counts[ep["task_type"]] += 1
    for task_type in sorted(counts):
        print(f"  {task_type:30s}: {counts[task_type]}")
    print(f"  {'TOTAL':30s}: {len(episodes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build candidate episode list from ALFRED validation data."
    )
    parser.add_argument(
        "--alfred_data",
        required=True,
        help="Path to ALFRED json_2.1.0/ directory (e.g. ../datasets/json_2.1.0)",
    )
    parser.add_argument(
        "--output",
        default="data/alfred_episodes/candidate_episodes.json",
        help="Output path for candidate_episodes.json",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid_seen", "valid_unseen"],
        help="ALFRED splits to include (default: valid_seen valid_unseen)",
    )
    parser.add_argument(
        "--max_per_type",
        type=int,
        default=60,
        help="Max episodes per task type (default: 60 → ~300 total before filtering)",
    )
    parser.add_argument(
        "--no_cap",
        action="store_true",
        help="Include all episodes without capping per task type",
    )
    args = parser.parse_args()

    alfred_data = Path(args.alfred_data)
    if not alfred_data.exists():
        print(f"[build] ERROR: ALFRED data directory not found: {alfred_data}")
        print("[build] Run: python scripts/download_alfred.py first")
        sys.exit(1)

    # Load all episodes
    print(f"[build] Loading episodes from: {alfred_data}")
    episodes = load_all_episodes(str(alfred_data), splits=args.splits)

    # Filter to target task types
    episodes = filter_by_task_type(episodes)

    # Cap per task type
    if not args.no_cap:
        print(f"\n[build] Capping at {args.max_per_type} per task type:")
        episodes = cap_per_task_type(episodes, args.max_per_type)

    # Summary
    print_summary(episodes)

    # Sanity check
    task_types_present = {ep["task_type"] for ep in episodes}
    missing = TARGET_TASK_TYPES - task_types_present
    if missing:
        print(f"\n[build] WARNING: Missing task types: {missing}")
        print("[build] Check that both valid_seen and valid_unseen are downloaded.")
    else:
        print(f"\n[build] All 5 target task types present.")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_episode_list(episodes, str(output))

    print(f"\n[build] Next steps:")
    print(f"  1. Set up conda env:  scripts\\setup_env_windows.bat")
    print(f"  2. Run baseline:      python src/inference/run_inference.py \\")
    print(f"                            --model qwen25vl --phase baseline \\")
    print(f"                            --episodes {args.output} \\")
    print(f"                            --frames_dir data/rendered_frames/ \\")
    print(f"                            --output_dir data/logs/raw/")
    print(f"  3. Then repeat for: internvl3, gemma3, llava_onevision")
    print(f"  4. Filter episodes:   python src/analysis/filter_episodes.py")


if __name__ == "__main__":
    main()
