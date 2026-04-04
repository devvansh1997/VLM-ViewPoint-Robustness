#!/usr/bin/env bash
# =============================================================================
# 00_install.sh — Redirects to setup_env_hpc.sh
#
# This script exists for backward compatibility with the numbered script
# convention. The actual setup logic lives in setup_env_hpc.sh.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[00_install] Running setup_env_hpc.sh..."
bash "$SCRIPT_DIR/setup_env_hpc.sh"
