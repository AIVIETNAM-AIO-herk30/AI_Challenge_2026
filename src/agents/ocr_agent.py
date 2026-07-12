"""
OCR Agent using Gemini Vision API.
Owner: Truong Hoang Thong / Team 1

Contract (docs/API_CONTRACT.md's working convention): output = extracted
text string, "" if none found (never None — downstream code should not
have to null-check this; it also satisfies the Elasticsearch schema's
asr_text/ocr_text typing directly).

Migrated from the old google-generativeai SDK (gemini-1.5-flash,
GOOGLE_API_KEY) to google-genai (gemini-2.0-flash, GEMINI_API_KEY) — the
old env var name didn't match what docker-compose.yml actually sets
(GEMINI_API_KEY), so this agent silently failed to find its key under
docker-compose.
"""

import os

from google import genai
from PIL import Image

from .base_agent import BaseAgent


class OCRAgent(BaseAgent):
    _PROMPT = (
        "Extract all visible text (signs, captions, labels, subtitles) in "
        "this image. Return only the raw text, no commentary or formatting. "
        "If there is no visible text, return an empty string."
    )

    def __init__(self, model_name: str = "gemini-3.5-flash", max_concurrent: int = 4):
        super().__init__("ocr", max_concurrent)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    async def _run(self, payload) -> str:
        image = payload if isinstance(payload, Image.Image) else Image.open(payload)
        response = self._client.models.generate_content(
            model=self._model_name, contents=[self._PROMPT, image]
        )
        return (response.text or "").strip()
