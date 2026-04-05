"""
Filter episode candidates after Phase 1 baseline run and lock the episode list.

Two-pass pipeline: at this stage action_success is null (validated later on Mac).
Instead, filter on whether models produced a valid mapped action at original pose.
Episodes where fewer than --threshold fraction of models return a valid action
are dropped (the VLMs can't interpret the scene at all).

The output selected_episodes.json becomes the single source of truth for
all subsequent phases. Do not change it after this step.

Usage:
    python src/analysis/filter_episodes.py \
        --logs_dir  data/logs/raw \
        --episodes  data/alfred_episodes/candidate_episodes.json \
        --output    data/alfred_episodes/selected_episodes.json \
        --threshold 0.50
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path


def load_baseline_logs(logs_dir: str) -> pd.DataFrame:
    records = []
    for fname in os.listdir(logs_dir):
        if "baseline" in fname and fname.endswith(".jsonl"):
            with open(os.path.join(logs_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return pd.DataFrame(records)


def filter_episodes(logs_dir: str, episodes_path: str, output_path: str, threshold: float = 0.50):
    df = load_baseline_logs(logs_dir)

    if df.empty:
        print("[filter] No baseline logs found. Run 01_run_baseline.sh first.")
        return

    # For each episode, compute fraction of models that returned a valid action
    df["has_valid_action"] = df["mapped_action"].notna()
    episode_valid_rate = (
        df.groupby("episode_id")["has_valid_action"]
        .mean()
    )

    valid_ids = set(episode_valid_rate[episode_valid_rate >= threshold].index.tolist())

    print(f"[filter] Total candidates: {len(episode_valid_rate)}")
    print(f"[filter] Passing threshold ({threshold:.0%} models produced valid action): {len(valid_ids)}")
    print(f"[filter] Dropped (unparseable): {len(episode_valid_rate) - len(valid_ids)}")

    # Per task type breakdown
    df_valid = df[df["episode_id"].isin(valid_ids)]
    print("\n[filter] Task type distribution in selected episodes:")
    print(df_valid.drop_duplicates("episode_id")["task_type"].value_counts().to_string())

    # Load original episode metadata and filter
    with open(episodes_path) as f:
        all_episodes = json.load(f)

    selected = [ep for ep in all_episodes if ep["episode_id"] in valid_ids]

    if len(selected) < 150:
        print(f"\nWARNING: Only {len(selected)} episodes selected. Target is ≥150.")
        print("Consider lowering --threshold or expanding the candidate set.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n[filter] Locked episode list saved to {output_path}")
    print(f"[filter] {len(selected)} episodes — DO NOT change this file after this step.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir",  default="data/logs/raw")
    p.add_argument("--episodes",  default="data/alfred_episodes/candidate_episodes.json")
    p.add_argument("--output",    default="data/alfred_episodes/selected_episodes.json")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Min fraction of models that must produce a valid action")
    args = p.parse_args()
    filter_episodes(args.logs_dir, args.episodes, args.output, args.threshold)


if __name__ == "__main__":
    main()
