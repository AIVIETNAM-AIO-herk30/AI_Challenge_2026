"""
OCR Agent using Gemini Vision API.
Owner: Truong Hoang Thong

Contract (docs/IMPLEMENTATION_PLAN.md §2.3):
  output = extracted text string, "" if none found (never None — downstream
  code should not have to null-check this).

Phase 1 note (docs/IMPLEMENTATION_PLAN.md §3): calling this agent during
indexing is OPTIONAL — visual embeddings drive most of the baseline
accuracy, OCR mainly helps queries that reference on-screen text/signs.
Skip it if API budget or time is tight; VideoIndexer already treats a
missing OCRAgent as "no OCR for this run", not an error.
"""

import os

from PIL import Image

from .base_agent import BaseAgent

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency at runtime
    genai = None


class OCRAgent(BaseAgent):
    _PROMPT = (
        "Extract all visible text (signs, captions, labels, subtitles) in "
        "this image. Return only the raw text, no commentary or formatting. "
        "If there is no visible text, return an empty string."
    )

    def __init__(self, model_name: str = "gemini-1.5-flash", max_concurrent: int = 4):
        super().__init__("ocr", max_concurrent)
        if genai is None:
            raise RuntimeError("google-generativeai is not installed (see requirements.txt)")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    async def _run(self, payload) -> str:
        image = payload if isinstance(payload, Image.Image) else Image.open(payload)
        response = self._model.generate_content([self._PROMPT, image])
        return (response.text or "").strip()
