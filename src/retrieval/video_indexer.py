"""
Video frame sampling and indexing pipeline.
Owner: Pham Viet Truong

Phase 1 (docs/IMPLEMENTATION_PLAN.md §3): fixed-FPS frame sampling via
decord, visual embedding through VisualAgent, optional ASR/OCR enrichment,
written into VectorStore per the metadata.parquet schema (§7.2).

DAKE-style adaptive keyframing (JPEG-size "steepness" scoring, no model
required) is the Phase 2 upgrade for this file — swap out _sample_frame_indices
once the fixed-FPS baseline is proven; nothing else in this class needs to
change since the rest of the pipeline works off whatever indices come back.
"""

import asyncio
import itertools
from pathlib import Path

import decord
import numpy as np
from PIL import Image

from ..agents.asr_agent import ASRAgent
from ..agents.ocr_agent import OCRAgent
from ..agents.visual_agent import VisualAgent
from .vector_store import VectorStore


class VideoIndexer:
    def __init__(
        self,
        visual_agent: VisualAgent,
        vector_store: VectorStore,
        asr_agent: ASRAgent | None = None,
        ocr_agent: OCRAgent | None = None,
        fps: float = 1.0,
        keyframe_dir: str | Path = "data/processed/keyframes",
        source_type: str = "surveillance",
    ):
        self._visual = visual_agent
        self._store = vector_store
        self._asr = asr_agent
        self._ocr = ocr_agent
        self._fps = fps
        self._keyframe_dir = Path(keyframe_dir)
        self._source_type = source_type
        # Monotonic id source for embedding_id (§2.2). Valid for a single
        # indexing run into a fresh store; if resuming into an existing
        # store, seed this from max(existing embedding_id) + 1 instead.
        self._next_id = itertools.count(len(vector_store))

    async def index_directory(self, video_dir: str | Path) -> None:
        video_dir = Path(video_dir)
        for video_path in sorted(video_dir.glob("*.mp4")):
            await self.index_video(video_path)

    async def index_video(self, video_path: Path) -> None:
        video_id = video_path.stem
        vr = decord.VideoReader(str(video_path))
        native_fps = vr.get_avg_fps()
        frame_indices = self._sample_frame_indices(len(vr), native_fps)

        asr_segments = await self._transcribe(video_path)

        out_dir = self._keyframe_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        embeddings: list[np.ndarray] = []
        metadata: list[dict] = []
        for frame_idx in frame_indices:
            timestamp_sec = frame_idx / native_fps
            image = Image.fromarray(vr[frame_idx].asnumpy())
            keyframe_path = out_dir / f"{frame_idx:08d}.jpg"
            image.save(keyframe_path, quality=90)

            visual_result = await self._visual.process({"image": image})
            if not visual_result.success:
                continue  # don't let one bad frame drop the whole video

            ocr_text = await self._extract_text(keyframe_path)
            asr_text = self._match_segment(asr_segments, timestamp_sec)

            embeddings.append(visual_result.output)
            metadata.append(
                {
                    "embedding_id": next(self._next_id),
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "keyframe_path": str(keyframe_path),
                    "asr_text": asr_text,
                    "ocr_text": ocr_text,
                    "source_type": self._source_type,
                }
            )

        if embeddings:
            self._store.add(np.stack(embeddings), metadata)

    def _sample_frame_indices(self, n_frames: int, native_fps: float) -> list[int]:
        step = max(1, round(native_fps / self._fps))
        return list(range(0, n_frames, step))

    async def _transcribe(self, video_path: Path) -> list[dict]:
        if self._asr is None:
            return []
        result = await self._asr.process(video_path)
        return result.output["segments"] if result.success else []

    async def _extract_text(self, keyframe_path: Path) -> str | None:
        if self._ocr is None:
            return None
        result = await self._ocr.process(keyframe_path)
        return result.output if result.success else None

    @staticmethod
    def _match_segment(segments: list[dict], timestamp_sec: float) -> str | None:
        for seg in segments:
            if seg["start"] <= timestamp_sec <= seg["end"]:
                return seg["text"]
        return None


async def _build_and_run(config: dict) -> None:
    """CLI entry point wiring: python -m src.retrieval.video_indexer"""
    agents_cfg = config["agents"]
    visual = VisualAgent(
        model_name=agents_cfg["visual"]["model"],
        pretrained=agents_cfg["visual"]["pretrained"],
        max_concurrent=agents_cfg["visual"].get("max_concurrent", 8),
    )
    # ASR/OCR are optional for the Phase 1 baseline (docs/IMPLEMENTATION_PLAN.md
    # §3) — comment out the relevant block in configs/config.yaml's `agents:`
    # section to skip either one and speed up indexing.
    asr = (
        ASRAgent(
            model_name=agents_cfg["asr"]["model"],
            language=agents_cfg["asr"].get("language", "vi"),
            max_concurrent=agents_cfg["asr"].get("max_concurrent", 2),
        )
        if "asr" in agents_cfg
        else None
    )
    ocr = (
        OCRAgent(
            model_name=agents_cfg["ocr"]["model"],
            max_concurrent=agents_cfg["ocr"].get("max_concurrent", 4),
        )
        if "ocr" in agents_cfg
        else None
    )

    store = VectorStore(
        embed_dim=config["retrieval"]["embed_dim"],
        nlist=config["retrieval"]["nlist"],
        nprobe=config["retrieval"]["nprobe"],
    )
    indexer = VideoIndexer(
        visual_agent=visual,
        vector_store=store,
        asr_agent=asr,
        ocr_agent=ocr,
        fps=config["data"]["frame_fps"],
    )
    await indexer.index_directory(config["data"]["video_dir"])
    store.save(config["data"]["embed_dir"])
    print(f"Indexed {len(store)} keyframes -> {config['data']['embed_dir']}")


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(_build_and_run(cfg))
