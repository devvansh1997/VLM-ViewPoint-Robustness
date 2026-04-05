#!/usr/bin/env bash
# Quick smoke test for AI2-THOR rendering on Mac
# Usage: bash scripts/mac/test_render_one.sh

set -e
cd "$(dirname "$0")/../.."
python scripts/test_render_one.py
