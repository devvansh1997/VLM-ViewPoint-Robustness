"""
Merge all raw JSONL log files into a single all_results.csv.

Run after each phase's inference is complete.
The phase column allows all results to coexist in one file.

Usage:
    python src/analysis/aggregate_logs.py
    python src/analysis/aggregate_logs.py --logs_dir data/logs/raw --output data/logs/aggregated/all_results.csv
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path


def load_all_logs(logs_dir: str) -> pd.DataFrame:
    records = []
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(logs_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[aggregate] Skipping malformed line in {fname}: {e}")
    return pd.DataFrame(records)


def validate_completeness(df: pd.DataFrame, episodes_path: str) -> None:
    """Print a completeness report — how many entries exist vs. expected."""
    import json
    from src.simulator.renderer import YAW_OFFSETS, PITCH_OFFSETS

    with open(episodes_path) as f:
        episodes = json.load(f)

    n_episodes = len(episodes)
    n_yaw      = len(YAW_OFFSETS)
    n_pitch    = len(PITCH_OFFSETS)
    n_models   = df[df["phase"] == "core"]["model"].nunique()
    expected   = n_episodes * n_yaw * n_pitch * n_models

    core_count = len(df[df["phase"] == "core"])
    print(f"[aggregate] Completeness check (core phase):")
    print(f"  Episodes: {n_episodes} | Yaw: {n_yaw} | Pitch: {n_pitch} | Models: {n_models}")
    print(f"  Expected: {expected} | Actual: {core_count}")
    if core_count < expected:
        print(f"  WARNING: Missing {expected - core_count} entries — re-run inference for incomplete models.")
    else:
        print(f"  OK: All core entries present.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default="data/logs/raw")
    p.add_argument("--output",   default="data/logs/aggregated/all_results.csv")
    p.add_argument("--episodes", default="data/alfred_episodes/selected_episodes.json",
                   help="Path to selected_episodes.json for completeness check")
    args = p.parse_args()

    print(f"[aggregate] Loading logs from {args.logs_dir}")
    df = load_all_logs(args.logs_dir)
    print(f"[aggregate] Total records: {len(df)}")
    print(f"[aggregate] Phases:  {df['phase'].value_counts().to_dict()}")
    print(f"[aggregate] Models:  {df['model'].unique().tolist()}")

    if os.path.exists(args.episodes):
        validate_completeness(df, args.episodes)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[aggregate] Saved to {args.output}")


if __name__ == "__main__":
    main()
