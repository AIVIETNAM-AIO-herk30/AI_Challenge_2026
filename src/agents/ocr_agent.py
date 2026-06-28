"""
OCR Agent using Gemini Vision API.
Owner: Truong Hoang Thong
"""

from .base_agent import BaseAgent


class OCRAgent(BaseAgent):
    """
    TODO (Truong Hoang Thong):
    - Initialize google.generativeai with GOOGLE_API_KEY
    - Implement _run(payload): payload is image path or bytes
    - Return extracted text string
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", max_concurrent: int = 4):
        super().__init__("ocr", max_concurrent)
        # TODO: setup Gemini client

    async def _run(self, payload) -> str:
        raise NotImplementedError
