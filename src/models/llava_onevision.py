"""
LLaVA-OneVision wrapper.

Local (small):  lmms-lab/llava-onevision-qwen2-0.5b-ov
HPC (full):     lmms-lab/llava-onevision-qwen2-7b-ov

The 0.5B checkpoint uses the older "llava" architecture (LlavaForConditionalGeneration)
while the 7B checkpoint uses the newer "llava_onevision" architecture
(LlavaOnevisionForConditionalGeneration).  We detect which to use at load time
by reading the model's config.json from the Hub.
"""

import torch
from PIL import Image

from src.models.base_vlm import BaseVLM

LOCAL_MODEL_ID = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
FULL_MODEL_ID  = "lmms-lab/llava-onevision-qwen2-7b-ov"


class LLaVAOneVision(BaseVLM):

    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def _is_old_arch(self) -> bool:
        """Check if model uses the old 'llava' architecture vs 'llava_onevision'."""
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        return model_type != "llava_onevision"

    def load(self) -> None:
        from transformers import AutoProcessor

        old_arch = self._is_old_arch()

        if old_arch:
            from transformers import LlavaForConditionalGeneration as ModelClass
            print(f"[LLaVAOneVision] Detected old 'llava' arch for {self.model_id}")
        else:
            from transformers import LlavaOnevisionForConditionalGeneration as ModelClass
            print(f"[LLaVAOneVision] Detected 'llava_onevision' arch for {self.model_id}")

        print(f"[LLaVAOneVision] Loading {self.model_id} ...")
        self.model = ModelClass.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"[LLaVAOneVision] Loaded.")

    def predict(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

        return response
