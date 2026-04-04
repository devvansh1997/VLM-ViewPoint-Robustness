# Setup Guide

## Prerequisites

- **Miniconda or Anaconda** installed
- **Git** installed
- **Python 3.10** (installed automatically via conda)

## Platform Support

| Platform | VLM Inference | AI2-THOR Rendering | Notes |
|----------|:---:|:---:|-------|
| Windows (NVIDIA GPU) | Yes | No | AI2-THOR has no Windows build |
| macOS (Intel/Apple Silicon) | No | Yes | M1+ runs via Rosetta 2, no CUDA for VLMs |
| Linux + NVIDIA GPU (HPC) | Yes | Yes | Full pipeline, use CloudRendering for headless |

## 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/VLM-ViewPoint-Robustness.git
cd VLM-ViewPoint-Robustness
```

## 2. Environment Setup

### Windows
```cmd
scripts\setup_env_windows.bat
conda activate viewpoint
```

### macOS
```bash
chmod +x scripts/setup_env_mac.sh
bash scripts/setup_env_mac.sh
conda activate viewpoint
```

### Linux / HPC
```bash
bash scripts/00_install.sh
conda activate viewpoint
```

## 3. Download ALFRED Data

The project uses ALFRED validation episodes (JSON metadata only, no RGB frames).
We render our own frames via AI2-THOR.

```bash
# Downloads ~2.7GB, extracts valid_seen + valid_unseen (~300MB)
python scripts/download_alfred.py
```

This places data at `../datasets/json_2.1.0/` (one level above the repo).

**Requires:** `pip install requests py7zr` (included in setup scripts).

### Verify the download

```bash
python scripts/download_alfred.py  # re-running shows episode counts if already extracted
```

Expected output:
```
[download] Data already extracted at .../datasets/json_2.1.0
  valid_seen: 410 episodes
  valid_unseen: 96 episodes
  Total: 506 episodes
```

## 4. Build Candidate Episode List

Scans the ALFRED data, filters to target task types, and caps at 60 episodes per type.

```bash
python scripts/build_candidate_list.py --alfred_data ../datasets/json_2.1.0
```

This creates `data/alfred_episodes/candidate_episodes.json`.

### Verify

```bash
python -c "import json; eps=json.load(open('data/alfred_episodes/candidate_episodes.json')); print(f'Episodes: {len(eps)}')"
```

## 5. Sanity Checks

### AI2-THOR rendering (macOS / Linux only)

```bash
python scripts/test_render_one.py
```

This renders two frames (original pose + one perturbed) and saves them to `data/rendered_frames/`.

### VLM loading (Windows / Linux with GPU only)

```bash
# Quick test — Qwen2.5-VL (no gated access required)
python -c "
from src.models.registry import load_model
from PIL import Image
import numpy as np
m = load_model('qwen25vl')
img = Image.fromarray(np.zeros((300,300,3), dtype='uint8'))
print('Response:', m.predict(img, 'Reply with only A.'))
"
```

### Gemma 3 (gated model — one-time setup)

1. Go to https://huggingface.co/google/gemma-3-4b-it and accept the license
2. Run `huggingface-cli login` and paste your HF token
3. Then test:
```bash
python -c "from src.models.registry import load_model; from PIL import Image; import numpy as np; m = load_model('gemma3'); print('Response:', m.predict(Image.fromarray(np.zeros((300,300,3), dtype='uint8')), 'Reply with only A.'))"
```

## 6. Data Directory Structure

After setup, your directory tree should look like:

```
Independent-Study/
├── datasets/
│   └── json_2.1.0/           # ALFRED episode JSONs (downloaded)
│       ├── valid_seen/
│       └── valid_unseen/
│
└── VLM-ViewPoint-Robustness/  # This repo
    └── data/
        ├── alfred_episodes/
        │   └── candidate_episodes.json   # Built from ALFRED data
        ├── rendered_frames/              # AI2-THOR frames (generated)
        └── logs/
            ├── raw/                      # Per-model JSONL logs
            └── aggregated/               # Merged CSV
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `conda activate` fails | Run `conda init powershell` (Windows) or `conda init bash` (Mac/Linux), restart terminal |
| `No module named torch` | Reactivate env: `conda activate viewpoint`, then `pip install torch torchvision` |
| AI2-THOR `ValueError: no build exists for arch=Windows` | AI2-THOR doesn't support Windows. Use Mac or Linux for rendering. |
| Gemma 3 `401 Unauthorized` / `GatedRepoError` | Run `huggingface-cli login` and accept license at HuggingFace |
| `No module named einops` or `timm` | Run `pip install einops timm` |
