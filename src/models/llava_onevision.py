"""
LLaVA-OneVision wrapper.

Local (small):  llava-hf/llava-onevision-qwen2-0.5b-ov-hf
HPC (full):     llava-hf/llava-onevision-qwen2-7b-ov-hf

We use the llava-hf Hub checkpoints which are properly converted to the
transformers-native LlavaOnevisionForConditionalGeneration format.
(The original lmms-lab checkpoints use an incompatible legacy weight layout.)
"""

import torch
from PIL import Image

from src.models.base_vlm import BaseVLM

LOCAL_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
FULL_MODEL_ID  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


class LLaVAOneVision(BaseVLM):

    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        print(f"[LLaVAOneVision] Loading {self.model_id} ...")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"[LLaVAOneVision] Loaded.")

    def predict(self, image: Image.Image, prompt: str) -> str:
        # Build a chat-style prompt with the image placeholder
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
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
