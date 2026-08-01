"""
Second vision encoder using BEiT-3, via timm — NOT transformers.
Owner: Team 1

M0 spike finding (deviation from the original plan, worth knowing before
touching this file): the community HF Hub checkpoints named for BEiT-3
image-text retrieval (e.g. Raghavan/beit3_base_patch16_384_coco_retrieval)
do NOT actually work with `transformers.AutoModel(trust_remote_code=True)`
— inspected the repo directly, it ships only a config.json + pytorch_model.bin,
no custom modeling code for trust_remote_code to execute, and transformers
has no native "beit3" architecture to fall back to (confirmed: KeyError in
CONFIG_MAPPING). The other retrieval-named repos (llnh/*, ThucPD/*) are raw
.pth checkpoints for Microsoft's original unilm/beit3 codebase (fairseq/
torchscale-based, not pip-installable) — not loadable via transformers or
timm either.

The one real, verified-working path: `timm` (already an installed
dependency via open_clip_torch) has first-class BEiT-3 support —
`timm.create_model("beit3_base_patch16_224.in22k_ft_in1k", ...)` loads and
runs. The catch: timm's BEiT-3 checkpoints are ImageNet-classification
checkpoints (BEiT-3's own pretraining + an ImageNet fine-tune), not a
retrieval-tuned image-text dual encoder — there is no text tower to load.

This is the plan's documented fallback (b): BEiT-3 runs in vision-only
mode. It still serves the "cascaded dual-encoder" purpose (a second,
differently-trained visual signal alongside SigLIP), but Team 2 cannot
embed a text query into this space — text-mode search only goes through
VisualAgent (SigLIP). `_run()` raises on a text payload rather than
silently returning a wrong vector, so a caller can't mistake this for a
real bug instead of a documented capability gap.
"""

from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from .base_agent import BaseAgent


class BEiT3Agent(BaseAgent):
    def __init__(
        self,
        model_name: str = "beit3_base_patch16_224.in22k_ft_in1k",
        max_concurrent: int = 4,
        device: str | None = None,
    ):
        super().__init__("beit3", max_concurrent)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_fp16 = self.device.startswith("cuda")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device).eval()
        if self._use_fp16:
            self.model = self.model.half()

        data_cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)

    async def _run(self, payload: dict) -> np.ndarray:
        if "image" in payload:
            return self._encode_image(payload["image"])
        if "text" in payload:
            raise ValueError(
                "BEiT3Agent has no text tower (vision-only checkpoint via timm, "
                "see module docstring) — text queries must go through VisualAgent instead"
            )
        raise ValueError("payload must contain an 'image' key")

    def _encode_image(self, image: str | Path | Image.Image) -> np.ndarray:
        img = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self._use_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            features = self.model(tensor)
        return self._normalize(features)

    @staticmethod
    def _normalize(features: torch.Tensor) -> np.ndarray:
        vec = features.squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
