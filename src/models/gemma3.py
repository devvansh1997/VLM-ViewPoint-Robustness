"""
Gemma 3 wrapper.

Local (small):  google/gemma-3-4b-it   (4B)
HPC (full):     google/gemma-3-12b-it  (12B)

Vision encoder: SigLIP2
Language backbone: Gemma 3
Resolution handling: Pan-and-scan tiling for high-res images
"""

import torch
from PIL import Image

from src.models.base_vlm import BaseVLM

LOCAL_MODEL_ID = "google/gemma-3-4b-it"
FULL_MODEL_ID  = "google/gemma-3-12b-it"


class Gemma3(BaseVLM):

    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor

        print(f"[Gemma3] Loading {self.model_id} ...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,   # Gemma 3 is optimized for bfloat16
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"[Gemma3] Loaded.")

    def predict(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        response = self.processor.decode(
            generated[0], skip_special_tokens=True
        ).strip()

        return response
