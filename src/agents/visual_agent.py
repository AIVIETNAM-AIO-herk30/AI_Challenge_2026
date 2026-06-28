"""
Visual Embedding Agent using SigLIP (via open_clip).
Owner: Truong Hoang Thong
"""

import numpy as np

from .base_agent import BaseAgent


class VisualAgent(BaseAgent):
    """
    Encodes images or text into SigLIP embedding vectors.

    TODO (Truong Hoang Thong):
    - Load open_clip model and preprocessor in __init__
    - Implement _run(payload):
        payload = {"image": path}  → image embedding (np.ndarray)
        payload = {"text": str}    → text embedding  (np.ndarray)
    - Normalize output to unit L2 norm (required for FAISS inner product search)
    """

    def __init__(
        self,
        model_name: str = "ViT-SO400M-14-SigLIP-384",
        pretrained: str = "webli",
        max_concurrent: int = 8,
    ):
        super().__init__("visual", max_concurrent)
        # TODO: load open_clip model

    async def _run(self, payload: dict) -> np.ndarray:
        raise NotImplementedError
