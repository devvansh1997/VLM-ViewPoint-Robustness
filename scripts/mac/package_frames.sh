#!/usr/bin/env bash
# =============================================================================
# package_frames.sh — Zip rendered frames for transfer to HPC
#
# Creates: data/rendered_frames.zip (~2-3 GB)
#
# Transfer to HPC:
#   scp data/rendered_frames.zip <user>@<hpc>:~/Independent-Study/VLM-ViewPoint-Robustness/data/
#
# On HPC, unzip:
#   cd ~/Independent-Study/VLM-ViewPoint-Robustness/data
#   unzip rendered_frames.zip
#
# Usage (from repo root):
#   bash scripts/mac/package_frames.sh
# =============================================================================

set -e

FRAMES_DIR="data/rendered_frames"
ZIP_FILE="data/rendered_frames.zip"

# Count frames
FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.png" | wc -l | tr -d ' ')

if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No PNG frames found in $FRAMES_DIR"
    echo "Run render_all_frames.sh first."
    exit 1
fi

echo "========================================"
echo " Packaging Rendered Frames"
echo " Frames: $FRAME_COUNT PNGs"
echo "========================================"

# Remove old zip if exists
[ -f "$ZIP_FILE" ] && rm "$ZIP_FILE"

cd data
zip -r rendered_frames.zip rendered_frames/*.png
cd ..

ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo ""
echo "========================================"
echo " Done!"
echo " File:   $ZIP_FILE"
echo " Size:   $ZIP_SIZE"
echo " Frames: $FRAME_COUNT"
echo ""
echo " Transfer to HPC:"
echo "   scp $ZIP_FILE <user>@<hpc>:~/Independent-Study/VLM-ViewPoint-Robustness/data/"
echo ""
echo " On HPC:"
echo "   cd ~/Independent-Study/VLM-ViewPoint-Robustness/data"
echo "   unzip rendered_frames.zip"
echo "========================================"
