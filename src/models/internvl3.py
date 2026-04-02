"""
InternVL3 wrapper.

Local (small):  OpenGVLab/InternVL3-2B
HPC (full):     OpenGVLab/InternVL3-8B

Uses trust_remote_code=True (InternVL ships custom modeling code).
"""

import torch
from PIL import Image

from src.models.base_vlm import BaseVLM

LOCAL_MODEL_ID = "OpenGVLab/InternVL3-2B"
FULL_MODEL_ID  = "OpenGVLab/InternVL3-8B"

# InternVL image normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class InternVL3(BaseVLM):

    def __init__(self, model_id: str = LOCAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        print(f"[InternVL3] Loading {self.model_id} ...")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[InternVL3] Loaded.")

    def predict(self, image: Image.Image, prompt: str) -> str:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        pixel_values = self._preprocess_image(image)
        pixel_values = pixel_values.to(dtype=torch.float16, device=self.model.device)

        # InternVL3 uses a <image> token in the prompt
        full_prompt = f"<image>\n{prompt}"

        generation_config = dict(
            max_new_tokens=8,
            do_sample=False,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            full_prompt,
            generation_config,
        )

        return response.strip()

    def _preprocess_image(self, image: Image.Image):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        img = image.convert("RGB")
        return transform(img).unsqueeze(0)
