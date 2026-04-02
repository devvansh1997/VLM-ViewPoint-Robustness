"""
Main inference loop.

For each (episode, yaw_offset, pitch_offset):
  1. Load the pre-rendered frame from disk
  2. Build the prompt
  3. Run VLM inference
  4. Map response to AI2-THOR action
  5. Check action success in the simulator
  6. Write one JSONL log line

Usage:
    python src/inference/run_inference.py \\
        --model qwen25vl \\
        --phase core \\
        --episodes data/alfred_episodes/selected_episodes.json \\
        --frames_dir data/rendered_frames/ \\
        --output_dir data/logs/raw/ \\
        [--use_full_model]        # omit for local small variant
        [--yaw_offsets 0]         # omit to run all offsets
        [--pitch_offsets 0]       # omit to run all offsets
        [--headless]              # add on Linux/HPC
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulator.alfred_loader  import load_episode_list
from src.simulator.renderer       import YAW_OFFSETS, PITCH_OFFSETS, frame_path
from src.simulator.success_checker import check_action_success
from src.models.registry          import load_model
from src.inference.prompt_builder  import (
    build_prompt,
    build_viewpoint_context_exact,
    build_viewpoint_context_qualitative,
)
from src.inference.action_mapper   import map_response


def parse_args():
    p = argparse.ArgumentParser(description="Run VLM inference on ALFRED episodes.")

    p.add_argument("--model",     required=True, choices=["qwen25vl", "internvl3", "gemma3", "llava_onevision"])
    p.add_argument("--phase",     required=True, choices=["baseline", "core", "ablation"])
    p.add_argument("--episodes",  required=True, help="Path to selected_episodes.json")
    p.add_argument("--frames_dir",required=True, help="Directory with pre-rendered frames")
    p.add_argument("--output_dir",required=True, help="Directory for JSONL output logs")

    p.add_argument("--use_full_model", action="store_true",
                   help="Use the full HPC model variant instead of the local small variant")

    p.add_argument("--yaw_offsets",   type=int, nargs="+", default=None,
                   help="Subset of yaw offsets to run (default: all)")
    p.add_argument("--pitch_offsets", type=int, nargs="+", default=None,
                   help="Subset of pitch offsets to run (default: all)")

    # Phase 4 ablation variant
    p.add_argument("--ablation_variant", choices=["exact", "qualitative"], default="exact",
                   help="Viewpoint context style for ablation phase")

    p.add_argument("--headless", action="store_true",
                   help="Run AI2-THOR headlessly (required on Linux HPC)")

    p.add_argument("--overwrite", action="store_true",
                   help="Re-run episodes that already have log entries")

    return p.parse_args()


def get_output_path(output_dir: str, model: str, phase: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{model}_{phase}_{timestamp}.jsonl")


def load_completed_keys(output_dir: str, model: str, phase: str) -> set:
    """
    Load (episode_id, yaw_offset, pitch_offset, phase) keys from existing log files
    so we can skip already-completed entries on resume.
    """
    completed = set()
    pattern = f"{model}_{phase}_"
    for fname in os.listdir(output_dir):
        if fname.startswith(pattern) and fname.endswith(".jsonl"):
            with open(os.path.join(output_dir, fname)) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = (
                            entry["episode_id"],
                            entry["yaw_offset"],
                            entry["pitch_offset"],
                        )
                        completed.add(key)
                    except Exception:
                        pass
    return completed


def build_viewpoint_context(phase: str, variant: str, yaw: int, pitch: int) -> str:
    if phase != "ablation":
        return ""
    if variant == "exact":
        return build_viewpoint_context_exact(yaw, pitch)
    return build_viewpoint_context_qualitative(yaw, pitch)


def main():
    args = parse_args()

    yaw_offsets   = args.yaw_offsets   or (YAW_OFFSETS   if args.phase != "baseline" else [0])
    pitch_offsets = args.pitch_offsets or (PITCH_OFFSETS  if args.phase != "baseline" else [0])

    # Phase 1 baseline: only original pose
    if args.phase == "baseline":
        yaw_offsets   = [0]
        pitch_offsets = [0]

    episodes = load_episode_list(args.episodes)
    model    = load_model(args.model, use_full=args.use_full_model)

    output_path = get_output_path(args.output_dir, args.model, args.phase)
    completed   = set() if args.overwrite else load_completed_keys(
        args.output_dir, args.model, args.phase
    )

    print(f"[run_inference] Model: {args.model} | Phase: {args.phase} | "
          f"Episodes: {len(episodes)} | Yaw: {yaw_offsets} | Pitch: {pitch_offsets}")
    print(f"[run_inference] Output: {output_path}")

    total = len(episodes) * len(yaw_offsets) * len(pitch_offsets)
    skipped = 0

    with open(output_path, "w") as log_file, tqdm(total=total) as pbar:
        for episode in episodes:
            ep_id = episode["episode_id"]

            for yaw in yaw_offsets:
                for pitch in pitch_offsets:
                    pbar.update(1)

                    key = (ep_id, yaw, pitch)
                    if key in completed:
                        skipped += 1
                        continue

                    # Load pre-rendered frame
                    img_path = frame_path(args.frames_dir, ep_id, yaw, pitch)
                    if not os.path.exists(img_path):
                        print(f"[run_inference] Missing frame: {img_path} — skipping")
                        continue

                    image = Image.open(img_path).convert("RGB")

                    # Build prompt
                    viewpoint_context = build_viewpoint_context(
                        args.phase, args.ablation_variant, yaw, pitch
                    )
                    prompt = build_prompt(episode["instruction"], viewpoint_context)

                    # VLM inference
                    vlm_response = model.predict(image, prompt)
                    mapped_action, letter = map_response(vlm_response)

                    # Check action success in simulator
                    action_success = False
                    error_msg = "no_action_mapped"

                    if mapped_action is not None:
                        action_success, error_msg = check_action_success(
                            episode=episode,
                            action=mapped_action,
                            yaw_offset=yaw,
                            pitch_offset=pitch,
                            headless=args.headless,
                        )

                    # Build log entry (matches the schema defined in Phase 0.4)
                    entry = {
                        "episode_id":          ep_id,
                        "task_type":           episode["task_type"],
                        "model":               args.model,
                        "phase":               args.phase,
                        "yaw_offset":          yaw,
                        "pitch_offset":        pitch,
                        "is_original_pose":    (yaw == 0 and pitch == 0),
                        "viewpoint_context":   viewpoint_context,
                        "prompt":              prompt,
                        "vlm_response":        vlm_response,
                        "mapped_action":       mapped_action,
                        "action_success":      action_success,
                        "original_pose_success": None,  # filled in post-processing (see below)
                        "image_path":          img_path,
                        "error_message":       error_msg,
                    }

                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()

    print(f"[run_inference] Done. Skipped {skipped} already-completed entries.")
    print(f"[run_inference] Log written to: {output_path}")

    # Fill original_pose_success for non-baseline phases
    if args.phase != "baseline":
        _backfill_original_pose_success(output_path, args.output_dir, args.model)


def _backfill_original_pose_success(log_path: str, output_dir: str, model: str) -> None:
    """
    After the run, fill in original_pose_success for each entry by joining
    against the baseline log for the same model.

    This avoids needing a separate baseline file at analysis time — the
    relative degradation is self-contained in the aggregated CSV.
    """
    # Load baseline results for this model
    baseline_results = {}
    for fname in os.listdir(output_dir):
        if fname.startswith(f"{model}_baseline") and fname.endswith(".jsonl"):
            with open(os.path.join(output_dir, fname)) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        baseline_results[e["episode_id"]] = e["action_success"]
                    except Exception:
                        pass

    if not baseline_results:
        print(f"[run_inference] No baseline log found for {model} — original_pose_success left as null")
        return

    # Rewrite the log file with original_pose_success filled in
    updated = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            ep_id = entry["episode_id"]
            entry["original_pose_success"] = baseline_results.get(ep_id)
            updated.append(entry)

    with open(log_path, "w") as f:
        for entry in updated:
            f.write(json.dumps(entry) + "\n")

    print(f"[run_inference] Backfilled original_pose_success for {len(updated)} entries.")


if __name__ == "__main__":
    main()
