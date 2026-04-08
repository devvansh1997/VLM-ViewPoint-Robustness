"""
Phase 4: Prompt Augmentation Ablation Analysis

Goal: Quantify how much providing a verbal viewpoint description in the prompt
      recovers task success compared to the core (no description) condition.

recovery_delta = ablation_success_rate - core_success_rate
  Positive delta → the prompt description helped
  Negative delta → the description confused the model

Input:  data/logs/aggregated/all_results.csv  (contains both core and ablation phases)
Output: results/ablation_recovery.csv, recovery delta plots (via plots.py)
"""

import pandas as pd
import numpy as np
from pathlib import Path


YAW_OFFSETS = [-45, -30, -15, 0, 15, 30, 45]


def load_ablation_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and separate core vs. ablation records.

    Returns:
        (core_df, ablation_df)
    """
    df = pd.read_csv(csv_path)
    core     = df[df["phase"] == "core"].copy()
    ablation = df[df["phase"] == "ablation"].copy()
    return core, ablation


def compute_recovery_delta(core_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, yaw_offset, pitch_offset, ablation_variant) compute:
      core_success:      mean action_success in core phase
      ablation_success:  mean action_success in ablation phase
      recovery_delta:    ablation_success - core_success
    """
    # Assign variant label to every ablation row
    # Exact entries (written before the fix) have ablation_variant=NaN — detect from viewpoint_context
    ablation_df = ablation_df.copy()
    ablation_df["variant_label"] = ablation_df.apply(
        lambda r: r["ablation_variant"] if pd.notna(r.get("ablation_variant")) else _detect_variant(str(r.get("viewpoint_context", ""))),
        axis=1,
    )

    rows = []

    for model in ablation_df["model"].unique():
        abl_model = ablation_df[ablation_df["model"] == model]
        core_model = core_df[core_df["model"] == model]

        for variant in abl_model["variant_label"].unique():
            abl_variant = abl_model[abl_model["variant_label"] == variant]

            for yaw in YAW_OFFSETS:
                for pitch in abl_variant["pitch_offset"].unique():
                    core_slice = core_model[
                        (core_model["yaw_offset"] == yaw) &
                        (core_model["pitch_offset"] == pitch)
                    ]
                    abl_slice = abl_variant[
                        (abl_variant["yaw_offset"] == yaw) &
                        (abl_variant["pitch_offset"] == pitch)
                    ]

                    if core_slice.empty or abl_slice.empty:
                        continue

                    core_rate = core_slice["action_success"].mean()
                    abl_rate  = abl_slice["action_success"].mean()

                    rows.append({
                        "model":            model,
                        "yaw_offset":       yaw,
                        "pitch_offset":     pitch,
                        "ablation_variant": variant,
                        "core_success":     round(core_rate, 4),
                        "ablation_success": round(abl_rate, 4),
                        "recovery_delta":   round(abl_rate - core_rate, 4),
                        "n_episodes":       len(core_slice),
                    })

    return pd.DataFrame(rows)


def _detect_variant(context: str) -> str:
    """Infer ablation variant from the viewpoint_context string."""
    if not context:
        return "none"
    if any(c.isdigit() for c in context):
        return "exact"
    return "qualitative"


def summarize_recovery(recovery_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate recovery delta across all rotation levels per model per variant.
    Used for the paper's summary table.
    """
    return (
        recovery_df
        .groupby(["model", "ablation_variant"])["recovery_delta"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .reset_index()
        .rename(columns={"mean": "mean_delta", "std": "std_delta",
                          "min": "min_delta",  "max": "max_delta"})
    )


def run_ablation_analysis(csv_path: str, output_dir: str) -> pd.DataFrame:
    """
    Full ablation analysis pipeline.

    Returns recovery_df — also saves to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    core_df, ablation_df = load_ablation_data(csv_path)
    print(f"[ablation] Core records: {len(core_df)} | Ablation records: {len(ablation_df)}")

    recovery_df = compute_recovery_delta(core_df, ablation_df)
    summary_df  = summarize_recovery(recovery_df)

    recovery_path = Path(output_dir) / "ablation_recovery.csv"
    summary_path  = Path(output_dir) / "ablation_summary.csv"

    recovery_df.to_csv(recovery_path, index=False)
    summary_df.to_csv(summary_path,   index=False)

    print(f"[ablation] Recovery deltas saved to {recovery_path}")
    print(f"\nRecovery Summary (mean delta > 0 = prompt helped):")
    print(summary_df.to_string(index=False))

    return recovery_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default="data/logs/aggregated/all_results.csv")
    p.add_argument("--output_dir", default="results/")
    args = p.parse_args()
    run_ablation_analysis(args.csv, args.output_dir)
