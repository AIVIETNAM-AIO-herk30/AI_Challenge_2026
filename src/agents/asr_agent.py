"""
ASR Agent using Whisper (local inference).
Owner: Truong Hoang Thong

Contract (docs/IMPLEMENTATION_PLAN.md §2.3):
  output = {"text": str, "segments": [{"start": float, "end": float, "text": str}]}

Note (2026 dataset shift, docs/ARCHITECTURE.md §1): egocentric/wearable-camera
audio is noisier than 2025's broadcast TV — expect lower-confidence
transcripts and possibly empty segments on ambient-noise-only stretches.
That's expected; the caller (VideoIndexer) treats missing asr_text as null,
not an error.
"""

import whisper

from .base_agent import BaseAgent


class ASRAgent(BaseAgent):
    def __init__(self, model_name: str = "large-v3", language: str = "vi", max_concurrent: int = 2):
        super().__init__("asr", max_concurrent)
        self.language = language
        self.model = whisper.load_model(model_name)

    async def _run(self, payload) -> dict:
        result = self.model.transcribe(str(payload), language=self.language)
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
        return {"text": result.get("text", "").strip(), "segments": segments}
