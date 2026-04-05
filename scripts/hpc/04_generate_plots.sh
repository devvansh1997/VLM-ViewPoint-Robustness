#!/usr/bin/env bash
# =============================================================================
# 04_generate_plots.sh — Phase 5: Generate all figures and analysis
#
# Requires: data/logs/aggregated/all_results.csv
# Output:   results/ (figures + CSV tables)
# =============================================================================

set -e

CSV="data/logs/aggregated/all_results.csv"
RESULTS_DIR="results"

if [ ! -f "$CSV" ]; then
    echo "ERROR: $CSV not found."
    echo "Run: python src/analysis/aggregate_logs.py"
    exit 1
fi

echo "========================================"
echo " Phase 5: Analysis & Visualization"
echo "========================================"

echo "[plots] Running symmetry analysis..."
python src/analysis/symmetry.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

echo "[plots] Running ablation analysis..."
python src/analysis/ablation.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

echo "[plots] Generating all figures..."
python src/analysis/plots.py \
    --csv        "$CSV" \
    --output_dir "$RESULTS_DIR"

echo ""
echo "========================================"
echo " Done. All outputs in $RESULTS_DIR/"
ls "$RESULTS_DIR"
echo "========================================"
