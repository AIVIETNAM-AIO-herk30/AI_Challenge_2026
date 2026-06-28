"""
ASR Agent using Whisper (local inference).
Owner: Truong Hoang Thong
"""

from .base_agent import BaseAgent


class ASRAgent(BaseAgent):
    """
    TODO (Truong Hoang Thong):
    - Load whisper model once in __init__ (avoid reloading per request)
    - Implement _run(payload): payload is audio file path (.mp3/.wav/.mp4)
    - Return {"text": str, "segments": list[dict]}
    """

    def __init__(self, model_name: str = "large-v3", language: str = "vi", max_concurrent: int = 2):
        super().__init__("asr", max_concurrent)
        self.language = language
        # TODO: load whisper model

    async def _run(self, payload) -> dict:
        raise NotImplementedError
