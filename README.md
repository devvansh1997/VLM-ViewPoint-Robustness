# VLM-ViewPoint-Robustness

**Viewpoint Robustness of Vision-Language Models for Embodied Action in Simulation**
CAP 6908 — Directed Independent Study | Devansh Sharma

---

## Research Question

> Does action success in ALFRED degrade as camera rotation increases from the original episode viewpoint, and which VLMs are most robust?

---

## Overview

This project evaluates four open-source VLMs under systematic camera rotation perturbations using the [ALFRED](https://github.com/askforalfred/alfred) benchmark on [AI2-THOR](https://ai2thor.allenai.org/). For each episode, we re-render the agent's initial observation at several yaw and pitch offsets, prompt a VLM to select the next action, and measure whether that action physically succeeds in the simulator.

**Models evaluated (zero-shot):**

| Model | Size | Notes |
|-------|------|-------|
| Qwen2.5-VL | 7B (full) / 3B (local) | Strong general VLM baseline |
| InternVL3 | 8B (full) / 2B (local) | Competitive open-source |
| Gemma 3 | 12B (full) / 4B (local) | Google 2025 SOTA, SigLIP2 vision encoder |
| LLaVA-OneVision | 7B (full) / 0.5B (local) | Widely used embodied AI baseline |

**Rotation offsets evaluated:**
- Yaw (horizontal): −45°, −30°, −15°, 0° (original), +15°, +30°, +45°
- Pitch (vertical): −20°, −10°, 0° (original), +10°, +20°

---

## Repository Structure

```
VLM-ViewPoint-Robustness/
├── env/
│   └── requirements.txt
├── data/
│   ├── alfred_episodes/         # ALFRED validation JSON metadata
│   │   ├── candidate_episodes.json
│   │   └── selected_episodes.json   ← single source of truth for all phases
│   ├── rendered_frames/         # ep_{id}_yaw_{deg}_pitch_{deg}.png
│   └── logs/
│       ├── raw/                 # one JSONL file per model per run
│       └── aggregated/          # all_results.csv
├── src/
│   ├── simulator/
│   │   ├── alfred_loader.py     # load ALFRED episode metadata
│   │   ├── renderer.py          # AI2-THOR rotation + frame saving
│   │   └── success_checker.py   # execute action + check success
│   ├── models/
│   │   ├── base_vlm.py          # abstract base class
│   │   ├── qwen25vl.py
│   │   ├── internvl3.py
│   │   ├── spatial_mllm.py
│   │   ├── llava_onevision.py
│   │   └── registry.py          # load_model("name", use_full=False)
│   ├── inference/
│   │   ├── prompt_builder.py    # single template, ablation-ready
│   │   ├── action_mapper.py     # VLM text → AI2-THOR action
│   │   └── run_inference.py     # main inference loop (CLI)
│   └── analysis/
│       ├── aggregate_logs.py    # merge JSONL → all_results.csv
│       ├── filter_episodes.py   # Phase 1 episode selection
│       ├── symmetry.py          # Phase 3 symmetry analysis
│       ├── ablation.py          # Phase 4 ablation analysis
│       └── plots.py             # all paper figures
├── scripts/
│   ├── 00_install.sh
│   ├── 01_render_all_frames.sh
│   ├── 02_run_baseline.sh
│   ├── 03_run_perturbation.sh
│   ├── 04_run_ablation.sh
│   └── 05_generate_plots.sh
└── results/                     # final figures and tables
```

---

## Reproducing Results

### Phase 0 — Setup

```bash
bash scripts/00_install.sh
```

Verify all three sanity checks pass before proceeding:
1. AI2-THOR launches headlessly and returns a frame
2. ALFRED episode JSON loads correctly
3. Smallest VLM (LLaVA-OneVision 0.5B) loads and returns a response

### Phase 1 — Baseline (Original Pose)

```bash
bash scripts/02_run_baseline.sh
python src/analysis/filter_episodes.py   # locks selected_episodes.json
```

### Phase 2 — Core Perturbation Study

```bash
bash scripts/01_render_all_frames.sh     # pre-render 8,750 frames
bash scripts/03_run_perturbation.sh      # run all 4 models
python src/analysis/aggregate_logs.py   # merge to all_results.csv
```

### Phase 3 & 4 — Analysis & Ablation

```bash
bash scripts/04_run_ablation.sh <best_model> <worst_model>
bash scripts/05_generate_plots.sh
```

---

## Running a Single Model (Custom)

```bash
python src/inference/run_inference.py \
    --model        qwen25vl \
    --phase        core \
    --episodes     data/alfred_episodes/selected_episodes.json \
    --frames_dir   data/rendered_frames/ \
    --output_dir   data/logs/raw/
    # --use_full_model   (add on HPC)
    # --headless         (add on Linux)
```

---

## Local vs. HPC

| Task | Local | HPC (A100 80GB) |
|------|-------|-----------------|
| Setup, sanity checks | ✓ | — |
| Rendering frames | ✓ (slow) | ✓ (preferred) |
| VLM inference | Small variants only | Full variants |
| Analysis + plots | ✓ | ✓ |

Local uses small model variants (3B/2B/0.5B). HPC uses full variants (7B/8B).
The pipeline is identical — only `--use_full_model` changes.

---

## Log Schema

Every inference call writes one JSON line:

```json
{
  "episode_id":            "trial_00001",
  "task_type":             "pick_and_place",
  "model":                 "qwen25vl",
  "phase":                 "core",
  "yaw_offset":            30,
  "pitch_offset":          0,
  "is_original_pose":      false,
  "viewpoint_context":     "",
  "prompt":                "...",
  "vlm_response":          "C",
  "mapped_action":         "RotateRight",
  "action_success":        true,
  "original_pose_success": true,
  "image_path":            "data/rendered_frames/ep_00001_yaw_30_pitch_0.png"
}
```
