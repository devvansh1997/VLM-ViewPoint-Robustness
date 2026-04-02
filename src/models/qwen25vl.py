"""
Qwen2.5-VL wrapper.

Local (small):  Qwen/Qwen2.5-VL-3B-Instruct
HPC (full):     Qwen/Qwen2.5-VL-7B-Instruct

Swap by passing model_id at construction time. Everything else is identical.
"""

import torch
from PIL import Image

from src.models.base_vlm import BaseVLM

LOCAL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
FULL_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen25VL(BaseVLM):

    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[Qwen25VL] Loading {self.model_id} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"[Qwen25VL] Loaded.")

    def predict(self, image: Image.Image, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

        return response
