"""
All visualization code for the paper.

All figures are generated from the aggregated CSV — no manual number entry anywhere.
Figures are saved to results/ as both PNG (for the repo) and PDF (for the paper).

Figure inventory (matches plan Section 5.1):
  Fig 1 — Accuracy vs. yaw offset curves, one line per model         [§3 Results]
  Fig 2 — Accuracy vs. pitch offset curves, one line per model       [§3 Results]
  Fig 3 — Heatmap: robustness drop Δθ (models × offset level)        [§3 Results]
  Fig 4 — Mirrored bar chart: +θ vs −θ symmetry per model            [§4.1 Symmetry]
  Fig 5 — Recovery delta curves: core vs. ablation A vs. ablation B  [§4.2 Ablation]
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Consistent color palette across all figures
MODEL_COLORS = {
    "qwen25vl":        "#2196F3",   # blue
    "internvl3":       "#4CAF50",   # green
    "spatial_mllm":    "#FF9800",   # orange
    "llava_onevision": "#9C27B0",   # purple
}

MODEL_LABELS = {
    "qwen25vl":        "Qwen2.5-VL",
    "internvl3":       "InternVL3",
    "spatial_mllm":    "Spatial-MLLM",
    "llava_onevision": "LLaVA-OneVision",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def _save(fig, output_dir: str, name: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plots] Saved {name}.png / .pdf to {output_dir}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 — Accuracy vs. yaw offset
# ---------------------------------------------------------------------------

def plot_yaw_curves(df: pd.DataFrame, output_dir: str) -> None:
    core = df[df["phase"] == "core"]
    yaw_acc = (
        core.groupby(["model", "yaw_offset"])["action_success"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in yaw_acc["model"].unique():
        m = yaw_acc[yaw_acc["model"] == model].sort_values("yaw_offset")
        ax.plot(
            m["yaw_offset"], m["action_success"],
            marker="o", label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Yaw Offset (degrees)")
    ax.set_ylabel("Action Success Rate")
    ax.set_title("Fig 1: Action Success vs. Yaw Offset")
    ax.set_xticks(sorted(yaw_acc["yaw_offset"].unique()))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower center", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig1_yaw_curves")


# ---------------------------------------------------------------------------
# Fig 2 — Accuracy vs. pitch offset
# ---------------------------------------------------------------------------

def plot_pitch_curves(df: pd.DataFrame, output_dir: str) -> None:
    core = df[df["phase"] == "core"]
    pitch_acc = (
        core.groupby(["model", "pitch_offset"])["action_success"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    for model in pitch_acc["model"].unique():
        m = pitch_acc[pitch_acc["model"] == model].sort_values("pitch_offset")
        ax.plot(
            m["pitch_offset"], m["action_success"],
            marker="s", label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pitch Offset (degrees, + = looking down)")
    ax.set_ylabel("Action Success Rate")
    ax.set_title("Fig 2: Action Success vs. Pitch Offset")
    ax.set_xticks(sorted(pitch_acc["pitch_offset"].unique()))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower center", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig2_pitch_curves")


# ---------------------------------------------------------------------------
# Fig 3 — Robustness drop heatmap
# ---------------------------------------------------------------------------

def plot_robustness_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """
    Δθ = baseline_success - success_at_offset, per (model, |yaw_offset|).
    """
    core = df[df["phase"] == "core"]

    baseline = (
        core[core["is_original_pose"] == True]
        .groupby("model")["action_success"]
        .mean()
        .rename("baseline")
    )

    yaw_acc = (
        core.groupby(["model", "yaw_offset"])["action_success"]
        .mean()
        .reset_index()
        .join(baseline, on="model")
    )
    yaw_acc["delta"] = yaw_acc["baseline"] - yaw_acc["action_success"]

    pivot = yaw_acc.pivot(index="model", columns="yaw_offset", values="delta")
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f", cmap="Reds",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Robustness Drop Δθ"},
    )
    ax.set_title("Fig 3: Robustness Drop Δθ (models × yaw offset)")
    ax.set_xlabel("Yaw Offset (degrees)")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, output_dir, "fig3_robustness_heatmap")


# ---------------------------------------------------------------------------
# Fig 4 — Mirrored bar chart (symmetry)
# ---------------------------------------------------------------------------

def plot_symmetry_bars(df: pd.DataFrame, output_dir: str) -> None:
    """
    For each model, plot +θ bars on the right and -θ bars on the left.
    Identical bar heights = perfect symmetry.
    """
    core = df[df["phase"] == "core"]
    models = core["model"].unique()
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    theta_levels = [15, 30, 45]

    for ax, model in zip(axes, models):
        pos_rates = [
            core[(core["model"] == model) & (core["yaw_offset"] == +t)]["action_success"].mean()
            for t in theta_levels
        ]
        neg_rates = [
            core[(core["model"] == model) & (core["yaw_offset"] == -t)]["action_success"].mean()
            for t in theta_levels
        ]

        y = np.arange(len(theta_levels))
        color = MODEL_COLORS.get(model, "steelblue")

        ax.barh(y,  pos_rates, color=color,       alpha=0.8, label="+θ (right)")
        ax.barh(y, [-r for r in neg_rates], color=color, alpha=0.4, label="−θ (left)")

        ax.set_yticks(y)
        ax.set_yticklabels([f"|θ|={t}°" for t in theta_levels])
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Success Rate")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(x):.0%}"))

        if ax == axes[0]:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Fig 4: Symmetry — +θ (right) vs −θ (left)", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "fig4_symmetry_bars")


# ---------------------------------------------------------------------------
# Fig 5 — Recovery delta (ablation)
# ---------------------------------------------------------------------------

def plot_recovery_curves(df: pd.DataFrame, output_dir: str) -> None:
    """
    For each ablation model: plot core vs. ablation-A vs. ablation-B success
    as a function of yaw offset.
    """
    core     = df[df["phase"] == "core"]
    ablation = df[df["phase"] == "ablation"]

    if ablation.empty:
        print("[plots] No ablation data found — skipping Fig 5.")
        return

    ablation_models = ablation["model"].unique()

    fig, axes = plt.subplots(1, len(ablation_models), figsize=(6 * len(ablation_models), 4), squeeze=False)

    for ax, model in zip(axes[0], ablation_models):
        core_m = core[core["model"] == model].groupby("yaw_offset")["action_success"].mean()
        color  = MODEL_COLORS.get(model, "steelblue")

        ax.plot(core_m.index, core_m.values, "o-", color=color, label="Core (no hint)", linewidth=2)

        for variant, style in [("exact", "--s"), ("qualitative", ":^")]:
            abl_v = ablation[
                (ablation["model"] == model) &
                (ablation["viewpoint_context"].apply(
                    lambda c: ("degree" in c.lower()) if variant == "exact" else ("degree" not in c.lower() and c != "")
                ))
            ]
            if abl_v.empty:
                continue
            abl_acc = abl_v.groupby("yaw_offset")["action_success"].mean()
            label = f"Ablation {'A (exact)' if variant == 'exact' else 'B (qualitative)'}"
            ax.plot(abl_acc.index, abl_acc.values, style, color=color, alpha=0.6, label=label)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Yaw Offset (degrees)")
        ax.set_ylabel("Action Success Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 5: Recovery from Prompt Augmentation", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, "fig5_recovery_curves")


# ---------------------------------------------------------------------------
# Entry point — generate all figures at once
# ---------------------------------------------------------------------------

def generate_all_plots(csv_path: str, output_dir: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"[plots] Loaded {len(df)} records from {csv_path}")

    plot_yaw_curves(df, output_dir)
    plot_pitch_curves(df, output_dir)
    plot_robustness_heatmap(df, output_dir)
    plot_symmetry_bars(df, output_dir)
    plot_recovery_curves(df, output_dir)

    print(f"[plots] All figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default="data/logs/aggregated/all_results.csv")
    p.add_argument("--output_dir", default="results/")
    args = p.parse_args()
    generate_all_plots(args.csv, args.output_dir)
