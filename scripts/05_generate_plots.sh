#!/usr/bin/env bash
# =============================================================================
# 05_generate_plots.sh — Phase 5: Generate all paper figures and run analyses
#
# Requires: data/logs/aggregated/all_results.csv to exist.
# Output:   results/fig1_yaw_curves.{png,pdf}
#           results/fig2_pitch_curves.{png,pdf}
#           results/fig3_robustness_heatmap.{png,pdf}
#           results/fig4_symmetry_bars.{png,pdf}
#           results/fig5_recovery_curves.{png,pdf}
#           results/symmetry_ratios.csv
#           results/wilcoxon_results.csv
#           results/ablation_recovery.csv
#           results/ablation_summary.csv
# =============================================================================

set -e

CSV="data/logs/aggregated/all_results.csv"
RESULTS_DIR="results"

if [ ! -f "$CSV" ]; then
    echo "ERROR: $CSV not found."
    echo "Run aggregate_logs.py first:"
    echo "  python src/analysis/aggregate_logs.py"
    exit 1
fi

echo "========================================"
echo " Phase 5: Analysis & Visualization"
echo "========================================"

# ---------------------------------------------------------------------------
# Symmetry analysis (Phase 3)
# ---------------------------------------------------------------------------
echo "[05_plots] Running symmetry analysis..."
python src/analysis/symmetry.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Ablation analysis (Phase 4)
# ---------------------------------------------------------------------------
echo "[05_plots] Running ablation analysis..."
python src/analysis/ablation.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# All figures
# ---------------------------------------------------------------------------
echo "[05_plots] Generating all figures..."
python src/analysis/plots.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

echo ""
echo "========================================"
echo " Done. All outputs in $RESULTS_DIR/"
ls "$RESULTS_DIR"
echo "========================================"
