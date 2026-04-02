"""
Single prompt template, ablation-ready from day one.

The {viewpoint_context} slot is:
  - Empty string ""  in Phases 1-3  (core experiment — no viewpoint info)
  - Filled string    in Phase 4     (ablation — explicit viewpoint description)

The inference code never changes between phases. Only what's passed into
{viewpoint_context} changes. This eliminates any confound between prompt
wording changes and the ablation effect.
"""

PROMPT_TEMPLATE = """\
You are an embodied agent in a household environment.
{viewpoint_context}
Task instruction: {instruction}

Based on the image, select the next best action to make progress on the task.

Options:
A) MoveAhead
B) RotateLeft
C) RotateRight
D) LookUp
E) LookDown
F) PickupObject
G) PutObject
H) OpenObject

Reply with only the letter of your chosen action.\
"""


def build_prompt(instruction: str, viewpoint_context: str = "") -> str:
    """
    Build the full prompt string.

    Args:
        instruction:       ALFRED task instruction (e.g. "Pick up the mug and place it in the sink").
        viewpoint_context: Viewpoint description string, or "" for core experiment.

    Returns:
        Formatted prompt string ready to pass to the VLM.
    """
    # Strip trailing whitespace on the viewpoint_context line when it's empty
    # so the prompt doesn't have a dangling blank line between the header and instruction.
    context_line = viewpoint_context.strip()
    if context_line:
        context_line = context_line + "\n"

    return PROMPT_TEMPLATE.format(
        viewpoint_context=context_line,
        instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Phase 4 ablation helpers — two variants as defined in the plan
# ---------------------------------------------------------------------------

def build_viewpoint_context_exact(yaw_deg: int, pitch_deg: int) -> str:
    """
    Variant A — Exact numeric description.
    e.g. "Note: The camera has rotated 30 degrees to the right and
          10 degrees upward from the original position."
    """
    if yaw_deg == 0 and pitch_deg == 0:
        return ""

    parts = []

    if yaw_deg != 0:
        direction = "to the right" if yaw_deg > 0 else "to the left"
        parts.append(f"{abs(yaw_deg)} degrees {direction}")

    if pitch_deg != 0:
        # In AI2-THOR, positive horizon = looking down; negative = looking up
        direction = "downward" if pitch_deg > 0 else "upward"
        parts.append(f"{abs(pitch_deg)} degrees {direction}")

    description = " and ".join(parts)
    return f"Note: The camera has rotated {description} from the original position."


def build_viewpoint_context_qualitative(yaw_deg: int, pitch_deg: int) -> str:
    """
    Variant B — Qualitative directional description.
    e.g. "Note: The camera is facing slightly to the right and slightly
          upward compared to the original position."
    """
    if yaw_deg == 0 and pitch_deg == 0:
        return ""

    def yaw_qualifier(deg: int) -> str:
        if abs(deg) <= 15:
            return "slightly"
        elif abs(deg) <= 30:
            return "moderately"
        else:
            return "considerably"

    def pitch_qualifier(deg: int) -> str:
        if abs(deg) <= 10:
            return "slightly"
        else:
            return "moderately"

    parts = []

    if yaw_deg != 0:
        qual = yaw_qualifier(yaw_deg)
        direction = "to the right" if yaw_deg > 0 else "to the left"
        parts.append(f"{qual} {direction}")

    if pitch_deg != 0:
        qual = pitch_qualifier(pitch_deg)
        direction = "downward" if pitch_deg > 0 else "upward"
        parts.append(f"{qual} {direction}")

    description = " and ".join(parts)
    return f"Note: The camera is facing {description} compared to the original position."
