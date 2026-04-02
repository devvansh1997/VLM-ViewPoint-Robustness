"""
Phase 3: Symmetry Analysis

Goal: Determine whether accuracy degradation is symmetric.
  - Does rotating +30° cause the same drop as rotating -30°?
  - Asymmetry suggests a directional bias in the model's spatial representations
    (likely left-to-right reading bias from pretraining on web images).

Input:  data/logs/aggregated/all_results.csv  (from Phase 2)
Output: results/symmetry_ratio.csv, symmetry plots (via plots.py)

No new inference or rendering is needed — this is pure analysis on the Phase 2 logs.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path


THETA_LEVELS = [15, 30, 45]


def load_core_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["phase"] == "core"].copy()


def compute_symmetry_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, |theta|) pair, compute:
      pos_rate: mean action_success at +theta
      neg_rate: mean action_success at -theta
      symmetry_ratio: pos_rate / neg_rate  (1.0 = perfectly symmetric)
      delta: pos_rate - neg_rate

    Returns a DataFrame with one row per (model, theta).
    """
    rows = []
    for model in df["model"].unique():
        m_df = df[df["model"] == model]
        for theta in THETA_LEVELS:
            pos = m_df[m_df["yaw_offset"] == +theta]["action_success"].mean()
            neg = m_df[m_df["yaw_offset"] == -theta]["action_success"].mean()
            ratio = (pos / neg) if neg > 0 else float("nan")
            rows.append({
                "model":          model,
                "theta":          theta,
                "pos_rate":       round(pos, 4),
                "neg_rate":       round(neg, 4),
                "symmetry_ratio": round(ratio, 4),
                "delta":          round(pos - neg, 4),
            })
    return pd.DataFrame(rows)


def run_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired Wilcoxon signed-rank test for each (model, theta):
    H0: success rates at +theta and -theta come from the same distribution.

    Returns a DataFrame with p-values.
    """
    rows = []
    for model in df["model"].unique():
        m_df = df[df["model"] == model]
        for theta in THETA_LEVELS:
            pos_scores = m_df[m_df["yaw_offset"] == +theta].sort_values("episode_id")["action_success"].values
            neg_scores = m_df[m_df["yaw_offset"] == -theta].sort_values("episode_id")["action_success"].values

            # Wilcoxon requires matched pairs and at least some variation
            if len(pos_scores) != len(neg_scores) or len(pos_scores) < 5:
                p_value = float("nan")
                stat = float("nan")
            else:
                try:
                    stat, p_value = wilcoxon(pos_scores, neg_scores, zero_method="wilcox")
                except ValueError:
                    # All differences zero — perfectly symmetric
                    stat, p_value = 0.0, 1.0

            rows.append({
                "model":    model,
                "theta":    theta,
                "statistic": round(stat, 4) if not np.isnan(stat) else stat,
                "p_value":   round(p_value, 4) if not np.isnan(p_value) else p_value,
                "significant": (p_value < 0.05) if not np.isnan(p_value) else False,
            })
    return pd.DataFrame(rows)


def run_symmetry_analysis(csv_path: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full symmetry analysis pipeline.

    Returns:
        (symmetry_df, wilcoxon_df) — also saves both to output_dir.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_core_results(csv_path)
    print(f"[symmetry] Loaded {len(df)} core inference records.")

    sym_df = compute_symmetry_ratios(df)
    wil_df = run_wilcoxon_tests(df)

    sym_path = Path(output_dir) / "symmetry_ratios.csv"
    wil_path = Path(output_dir) / "wilcoxon_results.csv"

    sym_df.to_csv(sym_path, index=False)
    wil_df.to_csv(wil_path, index=False)

    print(f"[symmetry] Symmetry ratios saved to {sym_path}")
    print(f"[symmetry] Wilcoxon results saved to {wil_path}")
    print("\nSymmetry Ratios (1.0 = perfectly symmetric):")
    print(sym_df.to_string(index=False))
    print("\nWilcoxon Tests (p < 0.05 = significant asymmetry):")
    print(wil_df.to_string(index=False))

    return sym_df, wil_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default="data/logs/aggregated/all_results.csv")
    p.add_argument("--output_dir", default="results/")
    args = p.parse_args()
    run_symmetry_analysis(args.csv, args.output_dir)
