"""
Abstract base class for all VLM wrappers.

Every model implements:
  - load()    — download weights and initialize model + processor
  - predict() — given a PIL image and prompt string, return the raw VLM response text

The inference loop only calls predict(); it never touches model internals.
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseVLM(ABC):

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and processor/tokenizer into memory.
        Called once before the inference loop begins.
        """
        ...

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> str:
        """
        Run a single inference call.

        Args:
            image:  PIL Image (RGB) — the pre-rendered frame.
            prompt: Full prompt string from prompt_builder.py.

        Returns:
            Raw text response from the VLM (ideally a single letter A–H,
            but the action_mapper handles noisy outputs).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={getattr(self, 'model_id', '?')})"
