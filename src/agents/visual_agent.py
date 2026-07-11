"""
Visual Embedding Agent using SigLIP (via open_clip).
Owner: Truong Hoang Thong

Contract (docs/IMPLEMENTATION_PLAN.md §2.1):
  - payload = {"image": path | PIL.Image}  -> image embedding
  - payload = {"text": str}                -> text embedding
  - both modes go through the SAME loaded model, so image and text
    embeddings land in one joint space (required for cosine/inner-product
    search against the FAISS index built by Part 1).
  - output: np.ndarray, shape (1152,), dtype float32, L2-normalized.
"""

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from .base_agent import BaseAgent


class VisualAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP-384",
        pretrained: str = "webli",
        max_concurrent: int = 8,
        device: str | None = None,
    ):
        super().__init__("visual", max_concurrent)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    async def _run(self, payload: dict) -> np.ndarray:
        if "image" in payload:
            return self._encode_image(payload["image"])
        if "text" in payload:
            return self._encode_text(payload["text"])
        raise ValueError("payload must contain an 'image' or 'text' key")

    def _encode_image(self, image: str | Path | Image.Image) -> np.ndarray:
        img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        return self._normalize(features)

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return self._normalize(features)

    @staticmethod
    def _normalize(features: torch.Tensor) -> np.ndarray:
        vec = features.squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
