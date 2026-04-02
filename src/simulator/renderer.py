"""
Handles AI2-THOR scene initialization and frame rendering.

Design principles:
- Headless-first: works identically on local Windows and Linux HPC.
  On Linux, set DISPLAY env var before calling (Xvfb or :0).
- Decoupled from inference: renders and saves frames to disk.
  Inference reads from disk — simulator never needs to run again.
- Naming convention: ep_{episode_id}_yaw_{yaw_deg}_pitch_{pitch_deg}.png
"""

import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Rotation offset grid (matches proposal)
# ---------------------------------------------------------------------------
YAW_OFFSETS   = [-45, -30, -15, 0, 15, 30, 45]
PITCH_OFFSETS = [-20, -10, 0, 10, 20]


def _build_controller(headless: bool = True):
    """
    Initialize an AI2-THOR controller.

    On Linux (HPC), uses the DISPLAY env var set by the caller (via Xvfb).
    On Windows, runs with the default display — headless flag is ignored
    because Windows doesn't use Xvfb.
    """
    from ai2thor.controller import Controller

    kwargs = dict(
        width=300,
        height=300,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    if headless and platform.system() == "Linux":
        display = os.environ.get("DISPLAY", ":0")
        kwargs["x_display"] = display

    return Controller(**kwargs)


def frame_filename(episode_id: str, yaw_deg: int, pitch_deg: int) -> str:
    """Canonical filename for a rendered frame."""
    return f"ep_{episode_id}_yaw_{yaw_deg}_pitch_{pitch_deg}.png"


def frame_path(output_dir: str, episode_id: str, yaw_deg: int, pitch_deg: int) -> str:
    """Full path for a rendered frame."""
    return os.path.join(output_dir, frame_filename(episode_id, yaw_deg, pitch_deg))


def render_frame(
    controller,
    episode: dict,
    yaw_offset: int,
    pitch_offset: int,
    output_dir: str,
    overwrite: bool = False,
) -> str:
    """
    Teleport the agent to the episode's start pose + rotation offset,
    capture a frame, and save it to disk.

    Args:
        controller:   An active AI2-THOR Controller instance.
        episode:      Episode dict with keys: scene, start_pose, episode_id.
        yaw_offset:   Degrees to add to the start pose's yaw (rotation y-axis).
        pitch_offset: Degrees to add to the start pose's horizon (camera pitch).
        output_dir:   Directory to save the PNG.
        overwrite:    Skip rendering if the file already exists.

    Returns:
        Path to the saved PNG.
    """
    out_path = frame_path(output_dir, episode["episode_id"], yaw_offset, pitch_offset)

    if not overwrite and os.path.exists(out_path):
        return out_path

    os.makedirs(output_dir, exist_ok=True)

    pose = episode["start_pose"]
    scene = episode["scene"]

    # Initialize the scene
    controller.reset(scene=scene)

    # Apply object poses and toggles from the original episode if available
    raw = episode.get("raw", {})
    scene_data = raw.get("scene", {})

    if scene_data.get("object_poses"):
        controller.step(
            "SetObjectPoses",
            objectPoses=scene_data["object_poses"],
        )

    if scene_data.get("object_toggles"):
        controller.step(
            "SetObjectToggles",
            objectToggles=scene_data["object_toggles"],
        )

    # Teleport to start position with rotation offset applied
    target_rotation = (pose["rotation"] + yaw_offset) % 360
    target_horizon  = float(pose["horizon"] + pitch_offset)

    event = controller.step(
        "TeleportFull",
        x=pose["x"],
        y=pose["y"],
        z=pose["z"],
        rotation={"x": 0, "y": target_rotation, "z": 0},
        horizon=target_horizon,
        standing=True,
    )

    if not event.metadata["lastActionSuccess"]:
        print(
            f"[renderer] TeleportFull failed for episode {episode['episode_id']} "
            f"yaw={yaw_offset} pitch={pitch_offset}: "
            f"{event.metadata.get('errorMessage', 'unknown error')}"
        )

    frame = event.frame  # numpy array (H, W, 3) uint8
    img = Image.fromarray(frame)
    img.save(out_path)

    return out_path


def render_episode_all_offsets(
    episode: dict,
    output_dir: str,
    headless: bool = True,
    overwrite: bool = False,
) -> dict:
    """
    Render all (yaw, pitch) combinations for a single episode.
    Opens and closes one controller per episode to avoid memory leaks.

    Returns:
        Dict mapping (yaw_offset, pitch_offset) -> saved file path.
    """
    controller = _build_controller(headless=headless)
    results = {}

    try:
        for yaw in YAW_OFFSETS:
            for pitch in PITCH_OFFSETS:
                path = render_frame(
                    controller, episode, yaw, pitch, output_dir, overwrite=overwrite
                )
                results[(yaw, pitch)] = path
    finally:
        controller.stop()

    return results


def render_original_pose(
    episode: dict,
    output_dir: str,
    headless: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Render only the original pose (yaw=0, pitch=0) for Phase 1 baseline.
    Opens and closes its own controller.
    """
    controller = _build_controller(headless=headless)
    try:
        path = render_frame(
            controller, episode, yaw_offset=0, pitch_offset=0,
            output_dir=output_dir, overwrite=overwrite
        )
    finally:
        controller.stop()
    return path
