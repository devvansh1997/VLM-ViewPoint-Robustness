"""
Executes a predicted action in AI2-THOR and checks whether it succeeded.

"Success" here is AI2-THOR's lastActionSuccess — did the physical action
execute without error? Examples:
  - PickupObject succeeds if an interactable object is in reach
  - MoveAhead succeeds if the agent is not blocked by a wall
  - RotateLeft/Right always succeeds mechanically

This is a single-step probe: the agent takes exactly one action from the
start position and we record whether it was physically executable.
The same controller state used to render the frame is recreated here so
the VLM input (pre-rendered frame) matches the simulator state.
"""

import platform
import os


def _build_controller(headless: bool = True):
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


def check_action_success(
    episode: dict,
    action: str,
    yaw_offset: int,
    pitch_offset: int,
    headless: bool = True,
) -> tuple[bool, str]:
    """
    Initialize AI2-THOR at the episode's start pose + rotation offsets,
    execute the given action, and return whether it succeeded.

    Args:
        episode:      Episode dict with scene, start_pose, raw.
        action:       AI2-THOR action string (e.g. "PickupObject", "MoveAhead").
        yaw_offset:   Yaw offset applied during rendering (must match).
        pitch_offset: Pitch offset applied during rendering (must match).
        headless:     Whether to run headlessly (Linux HPC).

    Returns:
        (success: bool, error_message: str)
    """
    controller = _build_controller(headless=headless)

    try:
        pose = episode["start_pose"]
        scene = episode["scene"]
        raw = episode.get("raw", {})
        scene_data = raw.get("scene", {})

        controller.reset(scene=scene)

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

        target_rotation = (pose["rotation"] + yaw_offset) % 360
        target_horizon  = float(pose["horizon"] + pitch_offset)

        controller.step(
            "TeleportFull",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation={"x": 0, "y": target_rotation, "z": 0},
            horizon=target_horizon,
            standing=True,
        )

        # Build action kwargs — object interaction actions need a target
        action_kwargs = _build_action_kwargs(controller, action)
        event = controller.step(action, **action_kwargs)

        success = event.metadata["lastActionSuccess"]
        error   = event.metadata.get("errorMessage", "")
        return success, error

    finally:
        controller.stop()


def _build_action_kwargs(controller, action: str) -> dict:
    """
    For object interaction actions (PickupObject, PutObject, OpenObject),
    automatically target the nearest visible interactable object.
    Navigation actions (MoveAhead, RotateLeft, etc.) need no extra kwargs.
    """
    interaction_actions = {"PickupObject", "PutObject", "OpenObject", "CloseObject"}

    if action not in interaction_actions:
        return {}

    # Find the nearest visible, interactable object
    objects = controller.last_event.metadata.get("objects", [])
    candidates = [
        obj for obj in objects
        if obj.get("visible") and obj.get("isInteractable")
    ]

    if not candidates:
        return {}

    # Pick the closest one
    nearest = min(candidates, key=lambda o: o.get("distance", float("inf")))
    return {"objectId": nearest["objectId"]}
