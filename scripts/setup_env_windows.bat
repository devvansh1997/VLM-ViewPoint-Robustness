@echo off
REM =============================================================================
REM setup_env_windows.bat -- Create conda environment on Windows native
REM
REM Run from the repo root in Anaconda Prompt or any terminal with conda:
REM   scripts\setup_env_windows.bat
REM =============================================================================

SET CONDA_ENV=viewpoint
SET PYTHON_VERSION=3.10

echo ========================================
echo  Phase 0: Environment Setup (Windows)
echo ========================================

echo [setup] Creating conda environment: %CONDA_ENV% (Python %PYTHON_VERSION%)
call conda create -n %CONDA_ENV% python=%PYTHON_VERSION% -y
if errorlevel 1 (
    echo [setup] ERROR: conda create failed. Make sure Anaconda/Miniconda is installed.
    exit /b 1
)

echo [setup] Activating environment...
call conda activate %CONDA_ENV%

echo [setup] Installing PyTorch (CUDA 12.1 for A100 / CPU fallback for local)
REM For local CPU/GPU testing:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REM On HPC with A100, use the HPC module system instead (e.g. module load pytorch)

echo [setup] Installing requirements...
pip install -r env\requirements.txt

echo [setup] Installing ai2thor...
pip install ai2thor

echo [setup] Installing ALFRED download dependency...
pip install requests py7zr

echo [setup] Installing qwen-vl-utils (for Qwen2.5-VL)...
pip install qwen-vl-utils

echo.
echo ========================================
echo  Sanity Checks
echo  Run these one at a time to verify setup
echo ========================================
echo.
echo  1. AI2-THOR (open a new Anaconda Prompt, activate viewpoint, then run):
echo     python -c "from ai2thor.controller import Controller; c = Controller(width=300, height=300); c.reset(scene='FloorPlan1'); print('Frame shape:', c.last_event.frame.shape); c.stop()"
echo.
echo  2. ALFRED data load (after downloading data):
echo     python -c "from src.simulator.alfred_loader import load_all_episodes; eps = load_all_episodes('../datasets/json_2.1.0'); print('Episodes:', len(eps))"
echo.
echo  3. Smallest VLM (LLaVA-OneVision 0.5B):
echo     python -c "from src.models.registry import load_model; from PIL import Image; import numpy as np; m = load_model('llava_onevision'); img = Image.fromarray(np.zeros((300,300,3), dtype='uint8')); print('Response:', m.predict(img, 'Reply with only A.'))"
echo.
echo  Do NOT proceed to Phase 1 until all three pass.
echo ========================================
echo.
echo [setup] Done. Activate with: conda activate %CONDA_ENV%
