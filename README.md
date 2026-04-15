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

| Model | Full (HPC) | Local (testing) |
|-------|-----------|-----------------|
| Qwen2.5-VL | 7B | 3B |
| InternVL3 | 8B | 2B |
| Gemma 3 | 12B | 4B |
| LLaVA-OneVision | 7B | 0.5B |

**Rotation offsets:**
- Yaw (horizontal): -45, -30, -15, 0 (original), +15, +30, +45
- Pitch (vertical): -20, -10, 0 (original), +10, +20
- Total: 35 viewpoints per episode

---

## Platform Roles

| Platform | Role | What Runs Here |
|----------|------|----------------|
| **Mac (M1 Air)** | Frame rendering + action validation | AI2-THOR simulator, frame capture, action success checks |
| **HPC (A100 Linux)** | VLM inference + analysis | All 4 models, stats, figures |
| Windows (NVIDIA GPU) | Local model testing | VLM sanity checks only |

> **Two-pass pipeline:** AI2-THOR has no Windows build and HPC lacks Vulkan for headless rendering.
> Pass 1 (HPC): VLM inference on pre-rendered frames — produces JSONL logs with predictions.
> Pass 2 (Mac): Replay predicted actions in AI2-THOR — fills in action_success.

---

## Repository Structure

```
VLM-ViewPoint-Robustness/
├── env/
│   └── requirements.txt
│
├── scripts/
│   ├── setup/                          # One-time environment setup
│   │   ├── setup_env_mac.sh
│   │   ├── setup_env_hpc.sh            # Python 3.12, torch 2.9.1, CUDA 12.6
│   │   └── setup_env_windows.bat
│   ├── data/                           # Data preparation
│   │   ├── download_alfred.py          # Download ALFRED JSON metadata
│   │   ├── build_candidate_list.py     # Build 240-episode candidate list
│   │   └── test_render_one.py          # Smoke test AI2-THOR rendering
│   ├── mac/                            # AI2-THOR tasks (Mac only)
│   │   ├── render_all_frames.sh        # Render all 8,400 frames
│   │   ├── package_frames.sh           # Zip for transfer to HPC
│   │   ├── run_action_validation.sh    # Pass 2: validate actions in AI2-THOR
│   │   └── test_render_one.sh
│   └── hpc/                            # Inference + analysis (HPC only)
│       ├── 01_run_baseline.sh          # Phase 1: original pose
│       ├── 02_run_perturbation.sh      # Phase 2: all yaw/pitch combos
│       ├── 03_run_ablation.sh          # Phase 4: prompt augmentation
│       └── 04_generate_plots.sh        # Phase 5: figures + stats
│
├── src/
│   ├── models/
│   │   ├── base_vlm.py                # Abstract base: load() + predict()
│   │   ├── qwen25vl.py
│   │   ├── internvl3.py
│   │   ├── gemma3.py
│   │   ├── llava_onevision.py
│   │   └── registry.py                # load_model("name", use_full=False)
│   ├── simulator/
│   │   ├── alfred_loader.py            # Load ALFRED episode metadata
│   │   ├── renderer.py                 # AI2-THOR rotation + frame capture
│   │   └── success_checker.py          # Execute action + check success
│   ├── inference/
│   │   ├── prompt_builder.py           # Multiple-choice template, ablation-ready
│   │   ├── action_mapper.py            # VLM text -> AI2-THOR action
│   │   ├── run_inference.py            # Pass 1: VLM inference (HPC)
│   │   └── action_validator.py         # Pass 2: action validation (Mac)
│   └── analysis/
│       ├── aggregate_logs.py           # Merge JSONL -> all_results.csv
│       ├── filter_episodes.py          # Phase 1 episode selection
│       ├── symmetry.py                 # Phase 3: Wilcoxon +theta vs -theta
│       ├── ablation.py                 # Phase 4: recovery delta analysis
│       └── plots.py                    # All 5 paper figures
│
├── data/
│   ├── alfred_episodes/                # candidate_episodes.json, selected_episodes.json
│   ├── rendered_frames/                # 8,400 PNGs (rendered on Mac)
│   └── logs/
│       ├── raw/                        # Per-model JSONL logs (Pass 1: HPC)
│       ├── validated/                  # Action-validated logs (Pass 2: Mac)
│       └── aggregated/                 # all_results.csv
│
└── results/                            # Final figures + analysis CSVs
```

---

## Experiment Pipeline

### Phase 0: Setup

**Mac:**
```bash
bash scripts/setup/setup_env_mac.sh && conda activate viewpoint
```

**HPC:**
```bash
bash scripts/setup/setup_env_hpc.sh    # add --flash-attn /path/to/wheel
conda activate viewpoint
huggingface-cli login                   # for Gemma 3 gated access
```

**Data (on either machine):**
```bash
python scripts/data/download_alfred.py
python scripts/data/build_candidate_list.py --alfred_data ../datasets/json_2.1.0
```

### Phase 1A: Render Frames (Mac)

```bash
bash scripts/mac/render_all_frames.sh       # ~5-7 hours on M1 Air
bash scripts/mac/package_frames.sh          # creates data/rendered_frames.zip
scp data/rendered_frames.zip user@hpc:~/Independent-Study/VLM-ViewPoint-Robustness/data/
```

### Phase 1B: Baseline Inference (HPC — Pass 1)

```bash
# On HPC: unzip frames first
cd data && unzip rendered_frames.zip && cd ..

bash scripts/hpc/01_run_baseline.sh
python src/analysis/filter_episodes.py \
    --logs_dir data/logs/raw \
    --episodes data/alfred_episodes/candidate_episodes.json \
    --output   data/alfred_episodes/selected_episodes.json
```

### Phase 2: Core Perturbation (HPC — Pass 1)

```bash
bash scripts/hpc/02_run_perturbation.sh
```

### Phase 3 + 4: Ablation (HPC — Pass 1)

```bash
bash scripts/hpc/03_run_ablation.sh qwen25vl internvl3 gemma3 llava_onevision
```

Runs both ablation variants (exact numeric + qualitative descriptions) for each model.
Each variant produces 8,400 entries per model (240 episodes x 35 viewpoints).

### Action Validation (Mac — Pass 2)

After all HPC inference is done, transfer logs and episode metadata to Mac:

```bash
# On Mac: pull inference logs + episode list from HPC
scp user@hpc:~/Independent-Study/VLM-ViewPoint-Robustness/data/logs/raw/*.jsonl \
    data/logs/raw/
scp user@hpc:~/Independent-Study/VLM-ViewPoint-Robustness/data/alfred_episodes/selected_episodes.json \
    data/alfred_episodes/

# Run action validation in AI2-THOR (uses caffeinate to prevent sleep)
caffeinate -dims bash scripts/mac/run_action_validation.sh

# Aggregate validated logs
python src/analysis/aggregate_logs.py --logs_dir data/logs/validated
```

The validator deduplicates checks on `(episode_id, yaw, pitch, action)` — identical
simulator checks across models/phases run only once. Batches by scene and reuses the
AI2-THOR controller for speed. Checkpoints every 500 checks for resume support.

### Phase 5: Analysis + Figures

```bash
python src/analysis/symmetry.py --csv data/logs/aggregated/all_results.csv --output_dir results
python src/analysis/ablation.py --csv data/logs/aggregated/all_results.csv --output_dir results
python src/analysis/plots.py    --csv data/logs/aggregated/all_results.csv --output_dir results
```

Generates 5 figures (PNG + PDF) and analysis CSVs in `results/`.

---

## Running a Single Model

```bash
python src/inference/run_inference.py \
    --model        qwen25vl \
    --phase        core \
    --episodes     data/alfred_episodes/selected_episodes.json \
    --frames_dir   data/rendered_frames/ \
    --output_dir   data/logs/raw/ \
    --use_full_model
```

---

## Log Schema

Every inference call writes one JSON line:

```json
{
  "episode_id":            "trial_00001",
  "task_type":             "pick_and_place",
  "model":                 "qwen25vl",
  "phase":                 "core",
  "ablation_variant":      null,
  "yaw_offset":            30,
  "pitch_offset":          0,
  "is_original_pose":      false,
  "viewpoint_context":     "",
  "prompt":                "...",
  "vlm_response":          "C",
  "mapped_action":         "RotateRight",
  "action_success":        true,
  "original_pose_success": true,
  "image_path":            "data/rendered_frames/ep_00001_yaw_30_pitch_0.png",
  "error_message":         ""
}
```

- `ablation_variant`: null for baseline/core, "exact" or "qualitative" for ablation phase
- `action_success`: null during Pass 1 (HPC), filled by Pass 2 (Mac validation)

---

## Results Summary

**Dataset:** 240 episodes (60 per task type), 35 viewpoints each, 4 models = 101,760 total evaluations.

**Key Findings:**

| Model | Baseline Success | Yaw Robustness | Pitch Sensitivity |
|-------|-----------------|----------------|-------------------|
| InternVL3 8B | ~85% | Very robust (flat curve) | Moderate drop at extreme pitch |
| LLaVA-OneVision 7B | ~70% | Robust | Stable |
| Qwen2.5-VL 7B | ~46% | Mild degradation | Sharp cliff at +pitch (looking down) |
| Gemma 3 12B | ~15% | Near-random (no signal) | Near-random |

- **Pitch perturbations are more disruptive than yaw** across all models
- **Left/right yaw symmetry confirmed** — no directional bias (Wilcoxon p > 0.05)
- **Qualitative viewpoint hints help more than exact numeric ones** for 3/4 models
- **Model size does not predict robustness** — Gemma 3 (12B) performs worst; InternVL3 (8B) performs best

**Output files in `results/`:**
- `fig1_yaw_curves` — Action success vs. yaw offset per model
- `fig2_pitch_curves` — Action success vs. pitch offset per model
- `fig3_robustness_heatmap` — Robustness drop (model x yaw offset)
- `fig4_symmetry_bars` — Left/right symmetry visualization
- `fig5_recovery_curves` — Ablation recovery (core vs. exact vs. qualitative)
- `symmetry_ratios.csv`, `wilcoxon_results.csv` — Statistical tests
- `ablation_recovery.csv`, `ablation_summary.csv` — Ablation deltas

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `conda activate` fails | `conda init bash` (Mac/Linux) or `conda init powershell` (Windows), restart terminal |
| AI2-THOR `no build exists for arch=Windows` | Expected — render frames on Mac |
| AI2-THOR `vulkaninfo failed` on HPC | Expected — HPC lacks Vulkan; use two-pass pipeline |
| AI2-THOR `TeleportFull collision` on Mac | Already handled — `forceAction=True` in renderer + success_checker |
| AI2-THOR hangs during rendering | Ctrl+C and restart — resume support skips completed frames |
| Gemma 3 `401 Unauthorized` | `huggingface-cli login` + accept license at huggingface.co |
| `No module named einops/timm` | `pip install einops timm` |
| `unzip` not available on HPC | `python -c "import zipfile; zipfile.ZipFile('rendered_frames.zip').extractall('.')"` |
| `Image data of dtype object` in plots | Fixed — `pd.to_numeric(df["action_success"], errors="coerce")` |
| Ablation qualitative all zeros | Ensure `ablation_variant` field is in log entries (fixed in run_inference.py) |
