#!/usr/bin/env bash
# =============================================================================
# run_action_validation.sh — Pass 2: Validate predicted actions in AI2-THOR
#
# Reads JSONL inference logs from HPC (Pass 1) and replays each predicted
# action in AI2-THOR to record action_success.
#
# PREREQ: Transfer inference logs from HPC to data/logs/raw/
#   scp user@hpc:~/Independent-Study/VLM-ViewPoint-Robustness/data/logs/raw/*.jsonl \
#       data/logs/raw/
#
# Usage:
#   bash scripts/mac/run_action_validation.sh                    # all models
#   bash scripts/mac/run_action_validation.sh --model qwen25vl   # one model
# =============================================================================

set -e

LOGS_DIR="data/logs/raw"
EPISODES="data/alfred_episodes/selected_episodes.json"
OUTPUT_DIR="data/logs/validated"

# Check prerequisites
if [ ! -d "$LOGS_DIR" ] || [ -z "$(ls "$LOGS_DIR"/*.jsonl 2>/dev/null)" ]; then
    echo "ERROR: No JSONL logs found in $LOGS_DIR"
    echo "Transfer inference logs from HPC first:"
    echo "  scp user@hpc:~/path/data/logs/raw/*.jsonl data/logs/raw/"
    exit 1
fi

if [ ! -f "$EPISODES" ]; then
    echo "ERROR: $EPISODES not found."
    exit 1
fi

LOG_COUNT=$(ls "$LOGS_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "========================================"
echo " Pass 2: Action Validation (Mac)"
echo " Logs:     $LOG_COUNT JSONL files"
echo " Episodes: $EPISODES"
echo " Output:   $OUTPUT_DIR"
echo "========================================"

python src/inference/action_validator.py \
    --logs_dir   "$LOGS_DIR" \
    --episodes   "$EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "========================================"
echo " Validation complete."
echo " Next: aggregate all logs (validated + raw)"
echo "   python src/analysis/aggregate_logs.py \\"
echo "     --logs_dir $OUTPUT_DIR"
echo "========================================"
