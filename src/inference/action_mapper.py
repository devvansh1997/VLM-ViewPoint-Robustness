"""
Maps VLM text responses to AI2-THOR action primitives.

The prompt asks the VLM to reply with a single letter (A-H).
This module extracts that letter robustly and maps it to the
corresponding action string.
"""

import re

# Matches the prompt options exactly
ACTION_MAP = {
    "A": "MoveAhead",
    "B": "RotateLeft",
    "C": "RotateRight",
    "D": "LookUp",
    "E": "LookDown",
    "F": "PickupObject",
    "G": "PutObject",
    "H": "OpenObject",
}

VALID_LETTERS = set(ACTION_MAP.keys())


def map_response(response: str) -> tuple[str | None, str]:
    """
    Extract the action letter from a VLM response and map to AI2-THOR primitive.

    Handles common noisy outputs:
      - "A"           → "MoveAhead"
      - "A)"          → "MoveAhead"
      - "A) MoveAhead"→ "MoveAhead"
      - "The answer is A." → "MoveAhead"
      - "  a  "       → "MoveAhead"  (case-insensitive)

    Args:
        response: Raw text output from the VLM.

    Returns:
        (action, letter) where action is the AI2-THOR string or None if unparseable,
        and letter is the extracted letter or "" if none found.
    """
    if not response:
        return None, ""

    # Look for a standalone letter A-H (optionally followed by ) or .)
    match = re.search(r'\b([A-Ha-h])\b', response)
    if match:
        letter = match.group(1).upper()
        return ACTION_MAP.get(letter), letter

    # Fallback: first character if it's a valid letter
    first = response.strip()[0].upper()
    if first in VALID_LETTERS:
        return ACTION_MAP[first], first

    return None, ""


def is_valid_response(response: str) -> bool:
    """Returns True if the response maps to a known action."""
    action, _ = map_response(response)
    return action is not None
