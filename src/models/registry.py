"""
Model registry — maps CLI model name strings to (class, local_id, full_id).

Usage:
    model = load_model("qwen25vl", use_full=False)   # local small variant
    model = load_model("qwen25vl", use_full=True)    # HPC full variant
"""

from src.models.qwen25vl        import Qwen25VL,       LOCAL_MODEL_ID as QWEN_LOCAL,   FULL_MODEL_ID as QWEN_FULL
from src.models.internvl3       import InternVL3,      LOCAL_MODEL_ID as INTERN_LOCAL, FULL_MODEL_ID as INTERN_FULL
from src.models.gemma3          import Gemma3,         LOCAL_MODEL_ID as GEMMA_LOCAL,  FULL_MODEL_ID as GEMMA_FULL
from src.models.llava_onevision import LLaVAOneVision, LOCAL_MODEL_ID as LLAVA_LOCAL,  FULL_MODEL_ID as LLAVA_FULL

# Registry: name -> (class, local_model_id, full_model_id)
REGISTRY = {
    "qwen25vl":        (Qwen25VL,       QWEN_LOCAL,   QWEN_FULL),
    "internvl3":       (InternVL3,      INTERN_LOCAL, INTERN_FULL),
    "gemma3":          (Gemma3,         GEMMA_LOCAL,  GEMMA_FULL),
    "llava_onevision": (LLaVAOneVision, LLAVA_LOCAL,  LLAVA_FULL),
}

MODEL_NAMES = list(REGISTRY.keys())


def load_model(name: str, use_full: bool = False):
    """
    Instantiate and load a VLM by registry name.

    Args:
        name:     One of the keys in REGISTRY (e.g. "qwen25vl").
        use_full: If True, load the full HPC model. If False, load the local small variant.

    Returns:
        A loaded BaseVLM instance ready for inference.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {MODEL_NAMES}")

    cls, local_id, full_id = REGISTRY[name]
    model_id = full_id if use_full else local_id
    model = cls(model_id=model_id)
    model.load()
    return model
